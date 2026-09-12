"""R5 canonical contracts: product fact, tool schema extensions, error mapping.

R5 reuses the canonical M2 ToolSpec/Registry/Executor chain.  New argument
schemas are admitted through :class:`R5ToolSpec`, a strict subclass that only
extends the schema table; no M-stage schema is weakened.  Unknown product
catalog misses map to the canonical ``DATA_MISSING`` code instead of borrowing
``ORDER_NOT_FOUND`` from the order domain.
"""
from __future__ import annotations

from typing import Any, Mapping

from pydantic import Field

from .domain.facts import DataQuality, DomainFactBase, EntityType, FactSource
from .m2_registry import RESERVED_CONTEXT_KEYS, ToolSpec


R5_PRODUCT_SOURCE_VERSION = "self_built_product_catalog.v1"
R5_CATALOG_VERSION = "r5.catalog.v1"


class R5ProductFact(DomainFactBase):
    """Versioned product fact with explicit source, currency and quality.

    ``price`` is a canonical decimal string; ``stock``/``price`` are ``None``
    when the catalog does not carry them (never defaulted, never guessed).
    """

    entity_type: EntityType = EntityType.PRODUCT
    source: FactSource = FactSource.SIMULATOR
    version: str = R5_CATALOG_VERSION

    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    price: str | None = None
    currency: str = "CNY"
    stock: int | None = None
    listing_status: str = "ACTIVE"


class R5ToolSpec(ToolSpec):
    """ToolSpec with R5 argument schemas appended; existing schemas unchanged.

    Pydantic treats underscore class attributes as private attrs, so the R5
    extension table lives at module level and only schemas declared here
    override the inherited candidate-args validation.
    """

    def validate_candidate_args(self, args: Mapping[str, Any]) -> None:
        schema = _R5_ARG_SCHEMAS.get(self.args_schema)
        if schema is None:
            return super().validate_candidate_args(args)
        if not isinstance(args, Mapping):
            raise ValueError("tool args must be an object")
        if RESERVED_CONTEXT_KEYS.intersection(args):
            raise ValueError("candidate args cannot override trusted invocation context")
        required, allowed = schema
        keys = set(args)
        if not required.issubset(keys) or not keys.issubset(allowed):
            raise ValueError(f"arguments do not match {self.args_schema}")


_R5_ARG_SCHEMAS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "aftersales.eligibility.v1": (
        frozenset({"order_id", "phone_last4", "service"}),
        frozenset({"order_id", "phone_last4", "service"}),
    ),
}


def product_fact_from_row(row: Mapping[str, Any], *, catalog_version: str = R5_CATALOG_VERSION) -> R5ProductFact:
    """Build a validated R5ProductFact from a products table row."""
    price_raw = row.get("price")
    stock_raw = row.get("stock")
    price: str | None = None
    stock: int | None = None
    quality = DataQuality.FRESH
    if price_raw is None:
        quality = DataQuality.MISSING
    else:
        price = str(price_raw)
    if stock_raw is None:
        quality = DataQuality.MISSING
    else:
        stock = int(stock_raw)
    attributes: dict[str, Any] = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
    return R5ProductFact(
        fact_id=f"product_{row['sku']}",
        entity_id=str(row["sku"]),
        source=FactSource.SIMULATOR,
        version=catalog_version,
        data_quality=quality,
        observed_at=row["observed_at"],
        sku=str(row["sku"]),
        name=str(row["name"]),
        attributes=attributes,
        price=price,
        currency=str(row.get("currency") or "CNY"),
        stock=stock,
        listing_status=str(row.get("listing_status") or "ACTIVE"),
    )


def product_read_error(code: str, message: str, **details: Any) -> dict[str, Any]:
    """Canonical failure payload shaped like the other tool callables."""
    return {"success": False, "code": code, "message": message, "details": details, "data": None}


__all__ = [
    "R5_CATALOG_VERSION",
    "R5_PRODUCT_SOURCE_VERSION",
    "R5ProductFact",
    "R5ToolSpec",
    "product_fact_from_row",
    "product_read_error",
]
