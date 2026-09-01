"""Backup / restore for the ecommerce SQLite database（M0，plan step 4）。

关闭 RISK-P0-01（源库无备份）的最小单元：源库只读校验 → 备份副本 → SHA-256
→ 恢复到隔离路径 → 逐表计数/完整性检查（runbook §4.3 任务 4、§4.4 不变量；
module-guide-06 §6 迁移协议）。

硬性边界：
- 源库零写入：以只读 URI（mode=ro）打开，仅用于 sqlite3 backup API 复制；
- 输出只包含计数、checksum、完整性结果，绝不打印/记录任何行内容
  （orders/aftersales_tickets 含 phone_last4/recipient/address 等 PII，RISK-P1-06）；
- M0 基线断言：orders=20、aftersales_tickets=12（与源库一致的恢复副本才算成功）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

OWNER = "m0-implementation-model"
MILESTONE = "M0"

# M0 冻结基线（startup_report.yaml 实测，只读 COUNT(*)）
EXPECTED_BASELINE_COUNTS: Dict[str, int] = {
    "orders": 20,
    "aftersales_tickets": 12,
}


class BackupError(RuntimeError):
    """备份/恢复校验失败（触发 runbook §4.5 stop condition 时抛出）。"""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ro_connect(db_path: str | Path) -> sqlite3.Connection:
    """只读连接（源库/校验目标均零写入）。"""
    uri = "file:" + Path(db_path).resolve().as_posix() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_table_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    """逐表 COUNT(*)（仅用户表；不含任何行数据）。"""
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]
    return {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in tables}


def _integrity_check(conn: sqlite3.Connection) -> str:
    return str(conn.execute("PRAGMA integrity_check").fetchone()[0])


def _fk_violations(conn: sqlite3.Connection) -> int:
    return len(conn.execute("PRAGMA foreign_key_check").fetchall())


def create_backup(
    source: str | Path,
    output: str | Path,
    expected_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """创建备份副本并自校验。源库只读打开，零写入。"""
    source = Path(source)
    output = Path(output)
    if not source.exists():
        raise BackupError(f"source database not found: {source}")
    if source.resolve() == output.resolve():
        raise BackupError("output path must differ from source (source db is never written)")
    output.parent.mkdir(parents=True, exist_ok=True)

    src = _ro_connect(source)
    try:
        src_integrity = _integrity_check(src)
        if src_integrity != "ok":
            raise BackupError(f"source integrity_check failed: {src_integrity}")
        source_counts = get_table_counts(src)
        # sqlite3 backup API：ro 源 → 副本，复制期间源库不可被写入。
        dst = sqlite3.connect(str(output))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    backup_sha256 = sha256_file(output)
    chk = _ro_connect(output)
    try:
        restore_integrity = _integrity_check(chk)
        restore_counts = get_table_counts(chk)
        fk_violations = _fk_violations(chk)
    finally:
        chk.close()

    if restore_integrity != "ok":
        raise BackupError(f"backup integrity_check failed: {restore_integrity}")
    if restore_counts != source_counts:
        raise BackupError(
            f"backup table counts mismatch source: source={source_counts} backup={restore_counts}"
        )
    if fk_violations:
        raise BackupError(f"backup foreign_key_check violations: {fk_violations}")
    if expected_counts is not None:
        for table, expected in expected_counts.items():
            if restore_counts.get(table) != expected:
                raise BackupError(
                    f"baseline count mismatch for {table}: expected={expected} "
                    f"actual={restore_counts.get(table)}（stop condition：计数不符）"
                )

    return {
        "kind": "backup",
        "milestone": MILESTONE,
        "owner": OWNER,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "backup_path": str(output),
        "source_sha256": sha256_file(source),
        "backup_sha256": backup_sha256,
        "source_counts": source_counts,
        "backup_counts": restore_counts,
        "integrity_check": restore_integrity,
        "foreign_key_violations": fk_violations,
        "source_write_mode": "read_only_uri",
    }


def restore_backup(
    backup_path: str | Path,
    restore_path: str | Path,
    expected_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """恢复到隔离路径并逐表校验（恢复副本与备份/源库计数一致才算通过）。"""
    backup_path = Path(backup_path)
    restore_path = Path(restore_path)
    if not backup_path.exists():
        raise BackupError(f"backup file not found: {backup_path}")
    if restore_path.resolve() == backup_path.resolve():
        raise BackupError("restore path must differ from backup file")
    restore_path.parent.mkdir(parents=True, exist_ok=True)

    backup_sha256 = sha256_file(backup_path)
    shutil.copyfile(backup_path, restore_path)
    restore_sha256 = sha256_file(restore_path)
    if restore_sha256 != backup_sha256:
        raise BackupError(
            f"restored file checksum mismatch: backup={backup_sha256} restore={restore_sha256}"
        )

    conn = _ro_connect(restore_path)
    try:
        integrity = _integrity_check(conn)
        counts = get_table_counts(conn)
        fk_violations = _fk_violations(conn)
    finally:
        conn.close()

    if integrity != "ok":
        raise BackupError(f"restored db integrity_check failed: {integrity}")
    if fk_violations:
        raise BackupError(f"restored db foreign_key_check violations: {fk_violations}")
    if expected_counts is not None:
        for table, expected in expected_counts.items():
            if counts.get(table) != expected:
                raise BackupError(
                    f"restored baseline count mismatch for {table}: expected={expected} "
                    f"actual={counts.get(table)}（stop condition：计数不符）"
                )

    return {
        "kind": "restore",
        "milestone": MILESTONE,
        "owner": OWNER,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backup_path": str(backup_path),
        "backup_sha256": backup_sha256,
        "restore_path": str(restore_path),
        "restore_sha256": restore_sha256,
        "restore_counts": counts,
        "integrity_check": integrity,
        "foreign_key_violations": fk_violations,
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.storage.backup",
        description="ecommerce.db 备份/恢复演练（源库零写入；输出不含行内容）",
    )
    parser.add_argument("--source", required=True, help="源数据库路径（只读打开）")
    parser.add_argument("--output", required=True, help="备份输出路径")
    parser.add_argument("--restore-to", default=None, help="恢复演练目标路径（隔离路径）")
    parser.add_argument(
        "--no-baseline-assert",
        action="store_true",
        help="跳过 M0 基线计数断言（orders=20 / aftersales_tickets=12）",
    )
    parser.add_argument("--report", default=None, help="结构化报告输出路径（JSON）")
    args = parser.parse_args(argv)

    expected = None if args.no_baseline_assert else dict(EXPECTED_BASELINE_COUNTS)
    try:
        backup_report = create_backup(args.source, args.output, expected_counts=expected)
        restore_report = None
        if args.restore_to:
            restore_report = restore_backup(
                args.output, args.restore_to, expected_counts=expected
            )
    except BackupError as e:
        print(f"BACKUP_FAILED: {e}", file=sys.stderr)
        return 1

    report = {"backup": backup_report, "restore": restore_report}
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
