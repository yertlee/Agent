from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent.interactive_runtime import (
    ChatOpenAIStructuredProvider,
    IntentV1,
    LLMRuntimeConfig,
    ModelCallError,
    ModelClient,
    ModelResult,
    ResponseV1,
)
from agent.runtime_port import RuntimePort
from app.fastapi_app import create_app
from tests.r1.test_interactive_runtime import Provider, _db


def test_model_client_retry_policy_is_typed_and_bounded(tmp_path: Path):
    config = LLMRuntimeConfig.from_environment(db_path=tmp_path / "orders.db", env={"LLM_MAX_RETRIES": "2"})

    class RateThenValid:
        def __init__(self):
            self.calls = 0

        def __call__(self, _prompt, schema):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("429 rate limit")
            return ModelResult(IntentV1(intent="ORDER_READ", confidence=0.9))

    retryable = RateThenValid()
    ModelClient(config, retryable).complete("x", IntentV1)
    assert retryable.calls == 3

    class AuthFailure:
        def __init__(self):
            self.calls = 0

        def __call__(self, _prompt, _schema):
            self.calls += 1
            raise PermissionError("unauthorized")

    auth = AuthFailure()
    with pytest.raises(ModelCallError) as exc:
        ModelClient(config, auth).complete("x", IntentV1)
    assert exc.value.code == "MODEL_AUTH_ERROR" and auth.calls == 1


def test_chat_openai_provider_factory_is_structured_and_secret_safe(monkeypatch, tmp_path: Path):
    calls = {}

    class FakeChatOpenAI:
        with_usage = True

        def __init__(self, **kwargs):
            calls.update(kwargs)

        def with_structured_output(self, schema, **kwargs):
            calls["schema"] = schema
            calls.update(kwargs)
            return self

        def invoke(self, _prompt):
            parsed = {"schema_version": "response.v1", "message_code": "ORDER_FACTS_V1", "claim_fields": ["order_status"], "evidence_ref": "0" * 64}
            raw = SimpleNamespace(usage_metadata={"input_tokens": 11, "output_tokens": 7}, response_metadata={}) if self.with_usage else SimpleNamespace()
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatOpenAI))
    config = LLMRuntimeConfig.from_environment(db_path=tmp_path / "orders.db", env={
        "OPENAI_API_KEY": "test-secret-value", "OPENAI_MODEL": "mock-model",
        "OPENAI_BASE_URL": "https://mock.invalid", "LLM_TIMEOUT_SECONDS": "7", "LLM_MAX_RETRIES": "2",
    })
    provider = ChatOpenAIStructuredProvider(config)
    output = provider("facts", ResponseV1)
    assert output.output["message_code"] == "ORDER_FACTS_V1"
    assert output.usage.input_tokens == 11 and output.usage.output_tokens == 7 and output.usage.available is True
    assert calls == {"model": "mock-model", "base_url": "https://mock.invalid", "timeout": 7.0, "max_retries": 0, "schema": ResponseV1, "method": "function_calling", "include_raw": True}
    assert "test-secret-value" not in repr(config.evidence())

    FakeChatOpenAI.with_usage = False
    no_usage = ChatOpenAIStructuredProvider(config)("facts", ResponseV1)
    assert no_usage.usage.available is False


def test_fastapi_rejects_harness_modes_and_dispatches_live_without_fallback(tmp_path: Path):
    class ProbePort(RuntimePort):
        def __init__(self):
            self.seen = []

        def chat(self, **_kwargs):
            raise AssertionError("legacy chat must not receive live mode")

        def interactive_chat(self, **kwargs):
            self.seen.append(kwargs["mode"])
            return SimpleNamespace(run_id="probe-run", session_id="probe-session", status="FAILED", answer="model failed")

    service = ProbePort()
    client = TestClient(create_app(runtime=service))
    headers = {"X-User-Id": "alice"}
    assert client.post("/api/chat", json={"message": "x", "mode": "fault"}, headers=headers).status_code == 422
    response = client.post("/api/chat", json={"message": "x", "mode": "live"}, headers=headers)
    assert response.status_code == 200 and service.seen == ["live"]


def test_interactive_owned_bundle_report_and_continuation(tmp_path: Path):
    service = RuntimePort(
        artifact_root=tmp_path / "port",
        interactive_runtime=__import__("agent.interactive_runtime", fromlist=["InteractiveRuntime"]).InteractiveRuntime(
            db_path=_db(tmp_path), provider=Provider(), artifact_root=tmp_path / "interactive"
        ),
    )
    first = service.interactive_chat(session_id=None, user_id="alice", message="x", mode="live")
    assert service.load_owned_bundle(run_id=first.run_id, user_id="alice").run_id == first.run_id
    second = service.interactive_chat(session_id=None, user_id="alice", message="x", mode="live", run_id=first.run_id)
    assert second.run_id != first.run_id and second.session_id == first.session_id
    assert service.load_owned_bundle(run_id=second.run_id, user_id="alice").run_id == second.run_id
    app_client = TestClient(create_app(runtime=service))
    report = app_client.get(f"/api/report/{second.run_id}", headers={"X-User-Id": "alice"})
    trace = app_client.get(f"/api/trace/{second.run_id}", headers={"X-User-Id": "alice"})
    assert report.status_code == 200 and report.json()["verified"] is True
    assert trace.status_code == 200 and trace.json()["frozen"] is True
