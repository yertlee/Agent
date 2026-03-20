from __future__ import annotations

import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI


def _load_env_file() -> None:
    # project_root/.env (public repo 不包含该文件；只在本地开发时可用）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or line.startswith("@env") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        return


_load_env_file()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def build_chat_model(*, temperature: float = 0.1, max_tokens: int = 800, tags: Optional[list[str]] = None) -> ChatOpenAI:
    model = ChatOpenAI(
        model=_env("OPENAI_MODEL", "gpt-4o"),
        api_key=_env("OPENAI_API_KEY"),
        base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if tags:
        return model.with_config(tags=tags)
    return model


def as_configurable(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    return config or {}
