from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from .logistics_types import now_timestamp, parse_timestamp


@dataclass
class CacheLookupResult:
    cache_key: str
    cache_hit: bool
    snapshot: Optional[Dict[str, Any]]
    raw_payload_ref: str
    last_query_at: str
    ttl_minutes: int
    should_throttle_query: bool
    file_path: str


class FileLogisticsCache:
    def __init__(
        self,
        *,
        base_dir: Optional[str] = None,
        ttl_minutes: Optional[int] = None,
        min_query_interval_minutes: int = 30,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        configured_dir = base_dir or os.getenv("LOGISTICS_CACHE_DIR") or str(project_root / "runtime" / "logistics_cache")
        self.base_dir = Path(configured_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_minutes = int(ttl_minutes or os.getenv("LOGISTICS_CACHE_TTL_MINUTES") or 30)
        self.min_query_interval_minutes = max(
            int(os.getenv("LOGISTICS_MIN_QUERY_INTERVAL_MINUTES") or min_query_interval_minutes),
            30,
        )

    def build_cache_key(self, carrier_code: str, tracking_no: str) -> str:
        return f"{(carrier_code or '').strip().lower()}:{(tracking_no or '').strip()}"

    def cache_meta(self, lookup: CacheLookupResult) -> Dict[str, Any]:
        return {
            "cache_key": lookup.cache_key,
            "cache_hit": lookup.cache_hit,
            "last_query_at": lookup.last_query_at,
            "ttl_minutes": lookup.ttl_minutes,
        }

    def lookup(self, carrier_code: str, tracking_no: str) -> CacheLookupResult:
        cache_key = self.build_cache_key(carrier_code, tracking_no)
        path = self._path_for(cache_key)
        record = self._read_record(path)
        last_query_at = str((record or {}).get("last_query_at") or "")
        snapshot = deepcopy((record or {}).get("snapshot")) if isinstance((record or {}).get("snapshot"), dict) else None

        if snapshot and self._is_fresh(str((record or {}).get("fetched_at") or "")):
            snapshot["source"] = "cache"
            snapshot["raw_payload_ref"] = str(path)
            return CacheLookupResult(
                cache_key=cache_key,
                cache_hit=True,
                snapshot=snapshot,
                raw_payload_ref=str(path),
                last_query_at=last_query_at,
                ttl_minutes=self.ttl_minutes,
                should_throttle_query=False,
                file_path=str(path),
            )

        return CacheLookupResult(
            cache_key=cache_key,
            cache_hit=False,
            snapshot=None,
            raw_payload_ref=str(path) if record else "",
            last_query_at=last_query_at,
            ttl_minutes=self.ttl_minutes,
            should_throttle_query=self._should_throttle_query(last_query_at),
            file_path=str(path),
        )

    def save_success(
        self,
        *,
        cache_key: str,
        snapshot: Dict[str, Any],
        raw_payload: Optional[Dict[str, Any]],
        source_name: str,
    ) -> str:
        path = self._path_for(cache_key)
        fetched_at = str(snapshot.get("fetched_at") or now_timestamp())
        stored_snapshot = dict(snapshot)
        stored_snapshot["raw_payload_ref"] = str(path)
        record = {
            "cache_key": cache_key,
            "source_name": source_name,
            "last_status": "success",
            "last_query_at": fetched_at,
            "fetched_at": fetched_at,
            "ttl_minutes": self.ttl_minutes,
            "snapshot": stored_snapshot,
            "raw_payload": raw_payload if isinstance(raw_payload, dict) else None,
            "last_error": None,
        }
        self._write_record(path, record)
        return str(path)

    def save_error(
        self,
        *,
        cache_key: str,
        source_name: str,
        error: Dict[str, Any],
        raw_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        path = self._path_for(cache_key)
        existing = self._read_record(path) or {}
        record = {
            "cache_key": cache_key,
            "source_name": source_name,
            "last_status": "error",
            "last_query_at": now_timestamp(),
            "fetched_at": str(existing.get("fetched_at") or ""),
            "ttl_minutes": self.ttl_minutes,
            "snapshot": existing.get("snapshot"),
            "raw_payload": raw_payload if isinstance(raw_payload, dict) else existing.get("raw_payload"),
            "last_error": dict(error or {}),
        }
        self._write_record(path, record)
        return str(path)

    def _path_for(self, cache_key: str) -> Path:
        digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
        return self.base_dir / f"{digest}.json"

    def _read_record(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_record(self, path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def _is_fresh(self, fetched_at: str) -> bool:
        timestamp = parse_timestamp(fetched_at)
        if timestamp is None:
            return False
        return (timestamp + timedelta(minutes=self.ttl_minutes)) > datetime.now()

    def _should_throttle_query(self, last_query_at: str) -> bool:
        timestamp = parse_timestamp(last_query_at)
        if timestamp is None:
            return False
        return (timestamp + timedelta(minutes=self.min_query_interval_minutes)) > datetime.now()
