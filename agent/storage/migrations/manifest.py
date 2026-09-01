"""Migration manifest 记录格式（M0，plan step 3）。

06 §6 第 4 点：每个迁移记录 migration id、输入/输出 schema、checksum、
owner、日期与证据。M0 为空 skeleton：只定义记录结构与校验，不含任何
实际迁移；实际建表/回填迁移属 M1。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

OWNER = "m0-implementation-model"

# M0 冻结基线（startup_report.yaml 实测；dry-run preflight 的输入口径）
M0_BASELINE_SCHEMA_FINGERPRINT = (
    "06f42075507ad219492f8bbc624f41dd6b1d1e88ec5fe7bcb8452d8c4af96f41"
)
M0_BASELINE_COUNTS = {"orders": 20, "aftersales_tickets": 12}

ALLOWED_STATUS = ("planned", "preflight_ok", "applied", "failed", "rolled_back")


@dataclass
class MigrationRecord:
    """单条迁移记录（06 §6 第 4 点字段不可缺省）。"""

    migration_id: str
    title: str
    input_schema_fingerprint: str
    output_schema_fingerprint: str
    checksum: str
    owner: str
    date: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: str = "planned"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.status not in ALLOWED_STATUS:
            raise ValueError(f"invalid migration status: {self.status}")
        return d


def compute_migration_checksum(payload: Dict[str, Any]) -> str:
    """迁移内容 checksum：canonical JSON（排序 key、无空白）SHA-256。"""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_m0_manifest(evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """M0 空 manifest：无迁移条目，仅冻结输入 schema 口径与协议约束。"""
    payload = {
        "milestone": "M0",
        "owner": OWNER,
        "migrations": [],
        "input_schema_fingerprint": M0_BASELINE_SCHEMA_FINGERPRINT,
        "baseline_counts": M0_BASELINE_COUNTS,
        "protocol_constraints": [
            "preflight -> backup -> single transaction -> postflight（06 §6）",
            "禁止 INSERT OR IGNORE 或等效掩盖；冲突/坏数据/计数不符必须失败并保留诊断",
            "M0 内任何迁移/写操作只允许在隔离副本 dry-run，源库零写入",
        ],
        "evidence": evidence or {},
    }
    payload["manifest_checksum"] = compute_migration_checksum(
        {k: v for k, v in payload.items() if k != "manifest_checksum"}
    )
    return payload


def validate_record(record: MigrationRecord) -> List[str]:
    """记录完整性校验（缺任一 06 §6 必填字段即报错，不静默补默认）。"""
    problems: List[str] = []
    required = (
        "migration_id",
        "title",
        "input_schema_fingerprint",
        "output_schema_fingerprint",
        "checksum",
        "owner",
        "date",
    )
    d = record.to_dict()
    for key in required:
        if not d.get(key):
            problems.append(f"missing required field: {key}")
    if record.status not in ALLOWED_STATUS:
        problems.append(f"invalid status: {record.status}")
    return problems
