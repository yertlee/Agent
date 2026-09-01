import hashlib
from pathlib import Path

import pytest

from agent.config import (
    CONFIG_INVALID,
    CONFIG_MISSING,
    ConfigError,
    compute_config_hash,
    config_evidence,
    load_config,
)


def _env(tmp_path: Path) -> dict[str, str]:
    return {"ECOMMERCE_DB_PATH": str(tmp_path / "ecommerce.db")}


def test_required_config_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config({})
    assert exc.value.code == CONFIG_MISSING
    assert exc.value.layer == "contract"
    assert exc.value.action == "fail_run"


@pytest.mark.parametrize("field,value", [("LANGSMITH_TRACING", "maybe"), ("LOGISTICS_TIMEOUT_SECONDS", "ten"), ("LOGISTICS_MODE", "live")])
def test_invalid_values_fail_closed(tmp_path: Path, field: str, value: str) -> None:
    env = _env(tmp_path)
    env[field] = value
    with pytest.raises(ConfigError) as exc:
        load_config(env)
    assert exc.value.code == CONFIG_INVALID


def test_secret_values_are_not_in_hash_or_evidence(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env["OPENAI_API_KEY"] = "sentinel-secret-value"
    config = load_config(env)
    digest = compute_config_hash(config)
    evidence = repr(config_evidence(config))
    assert "sentinel-secret-value" not in digest
    assert "sentinel-secret-value" not in evidence
    assert config.secret_references["OPENAI_API_KEY"] == {"present": True}


def test_hash_only_changes_for_secret_presence_and_non_secret_values(tmp_path: Path) -> None:
    present = _env(tmp_path)
    present["OPENAI_API_KEY"] = "one-secret"
    changed_secret = _env(tmp_path)
    changed_secret["OPENAI_API_KEY"] = "another-secret"
    assert compute_config_hash(load_config(present)) == compute_config_hash(load_config(changed_secret))
    assert compute_config_hash(load_config(_env(tmp_path))) != compute_config_hash(load_config(present))
    base = load_config(_env(tmp_path))
    changed_mode = _env(tmp_path)
    changed_mode["LOGISTICS_CACHE_TTL_MINUTES"] = "31"
    assert compute_config_hash(base) != compute_config_hash(load_config(changed_mode))
