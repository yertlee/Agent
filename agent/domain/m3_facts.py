"""M3 typed product and adapter result facts."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from .facts import DataQuality, DomainFactBase, EntityType, FactSource


class ProductFact(DomainFactBase):
    entity_type: EntityType = EntityType.PRODUCT
    sku: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    price: float = Field(ge=0)
    stock: int = Field(ge=0)


__all__ = ["ProductFact"]
