"""Stable import surface for M1 schema/repository consumers."""
from .migrations.m1 import M1_SCHEMA_VERSION, apply_m1_schema, schema_tables
from .repositories import M1Repository

__all__ = ["M1_SCHEMA_VERSION", "apply_m1_schema", "schema_tables", "M1Repository"]
