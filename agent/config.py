"""Config schema / validator / hash（M0，plan step 1）。

依据 module-guide-06 §5（config_hash：canonical JSON SHA-256，秘密值排除、
秘密引用名纳入）与 module-guide-03 §5（CONFIG_MISSING / CONFIG_INVALID，
layer=contract、action=fail_run）。对应 RISK-P0-02 的关闭单元。

硬性边界：
- 本模块只接受 env 映射（默认 os.environ），**绝不读取 .env 文件内容**
  （.env.example 仅用于字段名盘点，见 reports/startup_report.yaml）；
- secret 原值只保留"引用名 + 是否存在"，永不读取/记录/哈希其值；
- 物流模式固定为本地 SIMULATED（唯一合法值），运行不产生网络调用；
- 任何非法值 fail closed：抛 ConfigError（CONFIG_MISSING / CONFIG_INVALID），
  绝不带病启动。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

CONFIG_VERSION = "m0-2026-09-01"

# 03 §5 中央错误目录（canonical，只引用不自造同义码）
CONFIG_MISSING = "CONFIG_MISSING"
CONFIG_INVALID = "CONFIG_INVALID"
_ERROR_LAYER = "contract"
_ERROR_ACTION = "fail_run"

# 物流执行模式唯一合法值
LOGISTICS_MODE_FIXED = "SIMULATED"


class ConfigError(Exception):
    """配置错误（fail closed）。

    code 仅取 03 §5 中央错误目录的 CONFIG_MISSING / CONFIG_INVALID；
    layer=contract、action=fail_run（必须终止运行，不允许降级继续）。
    """

    def __init__(self, code: str, field: str, message: str) -> None:
        self.code = code
        self.layer = _ERROR_LAYER
        self.action = _ERROR_ACTION
        self.field = field
        self.message = message
        super().__init__(f"{code}: {field}: {message}")

    def envelope(self) -> Dict[str, Any]:
        """03 §5 ErrorEnvelope 形状的安全结构（不含任何 secret 值）。"""
        return {
            "code": self.code,
            "layer": self.layer,
            "action": self.action,
            "message_key": "config_error",
            "retryable": False,
            "retry_after_ms": None,
            "details": {"field": self.field},
        }


@dataclass(frozen=True)
class ConfigField:
    name: str
    kind: str = "str"                      # str | int | bool | path
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[tuple] = None
    secret: bool = False                   # 只记引用名 + 存在性，值永不进入 hash/日志
    deprecated: bool = False               # 弃用字段：非必填、不参与运行
    deprecation_reason: str = ""


# 非敏感运行字段与 secret 引用字段
FIELD_SPECS: tuple = (
    ConfigField("OPENAI_API_KEY", secret=True),
    ConfigField("OPENAI_BASE_URL", default="https://api.openai.com/v1"),
    ConfigField("OPENAI_MODEL", default="gpt-4o"),
    ConfigField("LANGSMITH_TRACING", kind="bool", default=False),
    ConfigField("LANGSMITH_ENDPOINT", default="https://api.smith.langchain.com"),
    ConfigField("LANGSMITH_API_KEY", secret=True),
    ConfigField("LANGSMITH_PROJECT", default=""),
    ConfigField("ECOMMERCE_DB_PATH", kind="path", required=True),
    # 唯一合法值 SIMULATED
    ConfigField("LOGISTICS_MODE", enum=(LOGISTICS_MODE_FIXED,), default=LOGISTICS_MODE_FIXED),
    ConfigField("LOGISTICS_TIMEOUT_SECONDS", kind="int", default=10),
    ConfigField("LOGISTICS_CACHE_TTL_MINUTES", kind="int", default=30),
    ConfigField("LOGISTICS_MIN_QUERY_INTERVAL_MINUTES", kind="int", default=30),
    ConfigField("LOGISTICS_CACHE_DIR", default="./runtime/logistics_cache"),
    ConfigField("RAG_EMBEDDING_MODEL", default="BAAI/bge-small-zh-v1.5"),
)

SECRET_REFERENCE_NAMES = tuple(f.name for f in FIELD_SPECS if f.secret)
DEPRECATED_FIELDS = tuple(f.name for f in FIELD_SPECS if f.deprecated)


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off", ""):
        return False
    raise ValueError(f"not a boolean: {raw!r}")


def _parse_int(raw: str) -> int:
    return int(raw.strip())


def _normalize(raw: str) -> str:
    return raw.strip()


@dataclass(frozen=True)
class AgentConfig:
    """校验通过的非敏感配置快照。

    - non_secret：字段名 -> 解析后的值（不含任何 secret 值）；
    - secret_references：引用名 -> {"present": bool}（绝不含值）；
    - config_version：M0 配置版本标识（06 §5 版本独立记录）。
    """

    non_secret: Dict[str, Any]
    secret_references: Dict[str, Dict[str, bool]]
    config_version: str = CONFIG_VERSION
    fields_used: tuple = ()
    fields_deprecated: tuple = ()

    def db_path(self) -> str:
        return str(self.non_secret["ECOMMERCE_DB_PATH"])


def canonical_json(payload: Any) -> str:
    """06 §5 canonical JSON：UTF-8、排序 key、无空白、固定数值。"""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_config_hash(config: AgentConfig) -> str:
    """config_hash = SHA-256(canonical JSON(config_input))。

    输入包含：config_version、非敏感运行参数值、secret 引用名与其存在性
    布尔。**绝不包含任何 secret 原值**（06 §5）。
    """
    payload = {
        "config_version": config.config_version,
        "non_secret": dict(config.non_secret),
        "secret_references": config.secret_references,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_config(env: Optional[Mapping[str, str]] = None) -> AgentConfig:
    """按 schema 校验并返回配置快照；任何缺失/类型错/非法枚举 fail closed。

    env 缺省为 os.environ。本函数不读取任何文件（包括 .env）。
    """
    source: Mapping[str, str] = os.environ if env is None else env
    non_secret: Dict[str, Any] = {}
    secret_references: Dict[str, Dict[str, bool]] = {}
    fields_used: list = []
    fields_deprecated: list = []

    for spec in FIELD_SPECS:
        raw = source.get(spec.name)
        raw = _normalize(raw) if isinstance(raw, str) else raw

        if spec.secret:
            # secret：只登记引用名与存在性；值为 None 时也不读取
            present = raw is not None and raw != ""
            secret_references[spec.name] = {"present": bool(present)}
            if spec.deprecated:
                fields_deprecated.append(spec.name)
            continue  # 原值到此为止，不进入 non_secret / hash / 日志

        if raw is None or raw == "":
            if spec.required:
                raise ConfigError(CONFIG_MISSING, spec.name, "required config is missing")
            if spec.kind == "bool":
                non_secret[spec.name] = bool(spec.default)
            else:
                non_secret[spec.name] = spec.default
            if spec.deprecated:
                fields_deprecated.append(spec.name)
            else:
                fields_used.append(spec.name)
            continue

        # 弃用字段：非必填；即使提供了值也只做最宽松的类型容忍并标记不使用
        if spec.deprecated:
            if spec.kind == "int":
                try:
                    _parse_int(raw)
                except ValueError as e:
                    raise ConfigError(CONFIG_INVALID, spec.name, str(e)) from e
            fields_deprecated.append(spec.name)
            # 值不进入运行参数（not_used）：仅校验可解析性后丢弃
            continue

        try:
            if spec.kind == "int":
                value: Any = _parse_int(raw)
            elif spec.kind == "bool":
                value = _parse_bool(raw)
            else:
                value = raw
        except ValueError as e:
            raise ConfigError(CONFIG_INVALID, spec.name, f"invalid value for type {spec.kind}: {e}") from e

        if spec.enum is not None and value not in spec.enum:
            raise ConfigError(
                CONFIG_INVALID,
                spec.name,
                f"value {value!r} not in allowed set {list(spec.enum)}",
            )
        non_secret[spec.name] = value
        fields_used.append(spec.name)

    config = AgentConfig(
        non_secret=non_secret,
        secret_references=secret_references,
        fields_used=tuple(fields_used),
        fields_deprecated=tuple(fields_deprecated),
    )
    return config


def config_evidence(config: AgentConfig) -> Dict[str, Any]:
    """供报告使用的安全证据结构（无任何 secret 值、无行内容）。"""
    return {
        "config_version": config.config_version,
        "config_hash_sha256": compute_config_hash(config),
        "hash_recipe": "SHA-256(canonical JSON: UTF-8, sort_keys, separators=(',',':'), ensure_ascii=False); "
        "input=config_version + non_secret values + secret reference names/presence; secret raw values excluded",
        "fields_used": list(config.fields_used),
        "fields_deprecated": [
            {"name": n, "reason": "not_used_by_owner_decision", "owner": "user", "date": "2026-09-01"}
            for n in config.fields_deprecated
        ],
        "secret_references": config.secret_references,
        "logistics_mode": LOGISTICS_MODE_FIXED,
    }
