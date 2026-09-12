"""M5 FastAPI service: runtime chat plus read-only frozen evidence views."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ImportError as exc:  # pragma: no cover - installation is enforced by requirements
    raise RuntimeError("M5 API requires fastapi; install requirements.txt") from exc

from agent.api.security import ChecksumMismatch, build_read_model, project_trace
from agent.runtime_port import RuntimePort


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = Field(default=None, min_length=1, max_length=200)
    run_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    wait: str = Field(default="terminal", pattern=r"^(terminal|none)$")
    cancel: bool = False
    mode: str = Field(default="simulated", pattern=r"^(live|simulated)$")

    @field_validator("message")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    session_id: str
    status: str
    answer: str


class ErrorEnvelope(BaseModel):
    code: str
    message: str


def _principal(authorization: str | None, x_user_id: str | None) -> str:
    """Resolve an authenticated principal; body ownership is never consulted."""
    if authorization:
        kind, _, value = authorization.partition(" ")
        if kind.lower() == "bearer" and value.strip():
            return value.strip()
    if x_user_id and x_user_id.strip():
        return x_user_id.strip()
    raise HTTPException(status_code=401, detail={"code": "AUTH_REQUIRED", "message": "authentication required"})


def create_app(*, runtime: RuntimePort | None = None) -> FastAPI:
    service = runtime or RuntimePort()
    app = FastAPI(title="LLMproject M5 Runtime API", version="m5.api.v1")
    app.state.runtime = service

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "HTTP_ERROR", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=detail)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"code": "VALIDATION_ERROR", "message": "request schema validation failed"})

    def auth(authorization: str | None = Header(default=None), x_user_id: str | None = Header(default=None)) -> str:
        return _principal(authorization, x_user_id)

    def bundle_for(run_id: str, user_id: str):
        try:
            bundle = service.load_owned_bundle(run_id=run_id, user_id=user_id)
            from eval.harness import verify_bundle
            verified = verify_bundle(bundle)
            if not verified.get("ok"):
                raise ChecksumMismatch("frozen bundle checksum verification failed")
            return bundle
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "RUN_NOT_FOUND", "message": "run not found"})
        except PermissionError:
            raise HTTPException(status_code=403, detail={"code": "AUTH_OWNERSHIP_DENIED", "message": "run is not owned by authenticated principal"})
        except (ValueError, ChecksumMismatch):
            raise HTTPException(status_code=409, detail={"code": "TRACE_CHECKSUM_MISMATCH", "message": "frozen evidence failed checksum verification"})

    @app.post("/api/chat", response_model=ChatResponse, status_code=200,
              responses={401: {"model": ErrorEnvelope}, 403: {"model": ErrorEnvelope}, 404: {"model": ErrorEnvelope}, 409: {"model": ErrorEnvelope}, 422: {"model": ErrorEnvelope}})
    def chat(payload: ChatRequest, user_id: str = Depends(auth)) -> ChatResponse:
        try:
            if payload.cancel or payload.wait == "none":
                if payload.run_id:
                    try:
                        service.load_owned_bundle(run_id=payload.run_id, user_id=user_id)
                    except KeyError:
                        raise HTTPException(status_code=404, detail={"code": "RUN_NOT_FOUND", "message": "run not found"})
                    except PermissionError:
                        raise HTTPException(status_code=403, detail={"code": "AUTH_OWNERSHIP_DENIED", "message": "run is not owned by authenticated principal"})
                raise HTTPException(status_code=409, detail={"code": "OPERATION_NOT_SUPPORTED", "message": "synchronous runtime does not support cancellation or nonterminal wait"})
            if payload.mode == "simulated":
                result = service.chat(session_id=payload.session_id, user_id=user_id, message=payload.message, run_id=payload.run_id)
            else:
                result = service.interactive_chat(session_id=payload.session_id, user_id=user_id, message=payload.message, mode=payload.mode, run_id=payload.run_id)
            return ChatResponse(run_id=result.run_id, session_id=result.session_id, status=result.status, answer=result.answer)
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "RUN_NOT_FOUND", "message": "run not found"})
        except PermissionError:
            raise HTTPException(status_code=403, detail={"code": "AUTH_OWNERSHIP_DENIED", "message": "run is not owned by authenticated principal"})
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "INVALID_CHAT", "message": str(exc)})
        except RuntimeError as exc:
            code = getattr(exc, "code", "RUNTIME_FAILED")
            raise HTTPException(status_code=422, detail={"code": code, "message": "interactive runtime failed"})

    @app.get("/api/trace/{run_id}", responses={401: {"model": ErrorEnvelope}, 403: {"model": ErrorEnvelope}, 404: {"model": ErrorEnvelope}, 409: {"model": ErrorEnvelope}})
    def trace(run_id: str, user_id: str = Depends(auth)) -> dict[str, Any]:
        bundle = bundle_for(run_id, user_id)
        trace_ref = Path(bundle.trace.path)
        try:
            events = [json.loads(line) for line in trace_ref.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError):
            raise HTTPException(status_code=409, detail={"code": "TRACE_UNREADABLE", "message": "trace is not readable"})
        return {"run_id": bundle.run_id, "bundle_id": bundle.bundle_id, "checksum": bundle.trace.checksum,
                "event_count": len(events), "events": project_trace(events), "frozen": True}

    @app.get("/api/report/{run_id}", responses={401: {"model": ErrorEnvelope}, 403: {"model": ErrorEnvelope}, 404: {"model": ErrorEnvelope}, 409: {"model": ErrorEnvelope}})
    def report(run_id: str, user_id: str = Depends(auth)) -> dict[str, Any]:
        bundle = bundle_for(run_id, user_id)
        trace_ref = Path(bundle.trace.path)
        events = [json.loads(line) for line in trace_ref.read_text(encoding="utf-8").splitlines() if line.strip()]
        read_model = build_read_model(bundle, events)
        return {"report_version": "m5.report.v1", "run_id": bundle.run_id, "bundle_id": bundle.bundle_id,
                "status": "VERIFIED", "verified": True, "trace_checksum": bundle.trace.checksum,
                "event_count": len(events), "read_model": read_model,
                "counts": {key: len(value) for key, value in read_model.items()},
                "version_tuple": bundle.version_tuple.model_dump(by_alias=True)}

    # Keep a plural alias for clients that call the collection a reports API.
    app.add_api_route("/api/reports/{run_id}", report, methods=["GET"])

    def html_report(run_id: str, user_id: str) -> HTMLResponse:
        bundle = bundle_for(run_id, user_id)
        events = [json.loads(line) for line in Path(bundle.trace.path).read_text(encoding="utf-8").splitlines() if line.strip()]
        labels = ["Run", "Plan", "Task", "Attempt", "Tool", "Result", "Review", "Evidence"]
        read_model = build_read_model(bundle, events)
        counts = {label: len(read_model[label]) for label in labels}
        trace_json = html.escape(json.dumps(read_model, ensure_ascii=False, indent=2))
        body = "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Run " + html.escape(run_id) + "</title></head><body>"
        body += "<h1>Run " + html.escape(run_id) + "</h1><nav>" + " ".join("<span>" + label + "</span>" for label in labels) + "</nav>"
        body += "<p>" + " | ".join(label + ": " + str(counts[label]) for label in labels) + "</p><pre>" + trace_json + "</pre></body></html>"
        return HTMLResponse(body)

    def html_route(run_id: str, user_id: str = Depends(auth)) -> HTMLResponse:
        return html_report(run_id, user_id)

    app.add_api_route("/api/report/{run_id}/html", html_route, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/api/html/{run_id}", html_route, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/api/trace/{run_id}/html", html_route, methods=["GET"], response_class=HTMLResponse)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("app.fastapi_app:app", host="127.0.0.1", port=8000)


__all__ = ["ChatRequest", "ChatResponse", "ErrorEnvelope", "app", "create_app"]
