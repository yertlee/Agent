"""M3 registry admission and manifest projection over the M2 registry."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from .m2_registry import Registry as M2Registry
from .m2_registry import ToolSpec


M3ToolSpec = ToolSpec


class M3Registry:
    """M3 view over the canonical M2 Registry; no weaker duplicate spec model."""
    def __init__(self, specs: Optional[list[M3ToolSpec]] = None, *, canonical: M2Registry | None = None):
        self.canonical = canonical or M2Registry(specs or [])

    def register(self, spec: M3ToolSpec) -> None:
        self.canonical.register(spec)

    def get(self, tool_ref: str) -> M3ToolSpec:
        return self.canonical.get(tool_ref)

    def require_capability(self, tool_ref: str, capability_ref: str) -> M3ToolSpec:
        spec = self.get(tool_ref)
        if not capability_ref or spec.capability_ref != capability_ref:
            raise PermissionError("M3 capability is not explicitly authorized")
        return spec

    def refs(self) -> tuple[str, ...]:
        return self.canonical.refs()

    def as_manifest(self) -> list[dict[str, Any]]:
        return [self.canonical.get(key).model_dump(exclude={"callable"}) for key in self.refs()]


def _from_m2(spec: ToolSpec) -> M3ToolSpec:
    return spec


def build_m3_registry() -> M3Registry:
    """Admit M2 tools plus the first M3 Product port.

    product/read@v1 is intentionally admitted here, not retrofitted into the
    M2 module, and has a deterministic in-process catalog callable.
    """
    from .m2_registry import build_m2_registry

    catalog = {
        "SKU-FIXTURE-001": {"name": "fixture product", "price": 10.0, "stock": 8, "attributes": {"category": "fixture"}},
        "SKU-FIXTURE-002": {"name": "fixture accessory", "price": 5.0, "stock": 0, "attributes": {"category": "fixture"}},
    }

    def product_read(*, sku: str) -> dict[str, Any]:
        item = catalog.get(sku)
        if item is None:
            return {"success": False, "code": "ORDER_NOT_FOUND", "message": "product not found", "data": None}
        return {"success": True, "code": "OK", "data": ProductFact(fact_id=f"product_{sku}", entity_id=sku, source=FactSource.SIMULATOR, version="product.v1", observed_at=datetime.now(timezone.utc), sku=sku, attributes=item["attributes"], price=item["price"], stock=item["stock"]).model_dump(mode="json")}

    from .domain.facts import FactSource
    from .domain.m3_facts import ProductFact
    canonical = build_m2_registry()
    canonical.register(ToolSpec(tool_ref="product/get@v1", capability_ref="product/read@v1", owner="product-agent", args_schema="product.read.v1", result_schema="product.result.v1", risk="READ", implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("ORDER_NOT_FOUND", "DATA_MISSING"), callable=product_read))
    return M3Registry(canonical=canonical)


__all__ = ["M3Registry", "M3ToolSpec", "build_m3_registry"]
