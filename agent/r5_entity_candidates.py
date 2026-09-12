"""R5 deterministic entity candidate extraction.

Structured identifiers (order ids, tracking numbers, SKUs, phone last-four)
are found by deterministic rules; the model's job is to judge their semantic
role (which candidate is the current target).  Acting on an ambiguous text
without an explicit model choice is not safe, so the runtime forces a
clarification instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

# Kinds where a second distinct value in one message is a real ambiguity.
# A carrier is included here because two explicitly named carriers must never
# be collapsed into one by a deterministic fill rule.
AMBIGUITY_KINDS = ("order_id", "tracking_no", "carrier_code", "sku", "service")

_ORDER_ID = re.compile(r"(?<!\d)(20\d{9,11})(?!\d)")
_SKU = re.compile(r"\bSKU-[A-Za-z0-9]{2,}\b", re.IGNORECASE)
_TRACKING = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{0,4}\d{10,20})(?![A-Za-z0-9])")
_PHONE = re.compile(r"(?:尾号|后四位|手机号后四位|手机尾号)[^\d]{0,4}(\d{4})")
# Carrier values are accepted only when they are explicitly marked in the
# request (or are one of the short, unambiguous carrier codes next to a
# logistics context word).  We never infer a carrier from a tracking number.
# These are the carrier values used by the repository-owned snapshots.  A
# Chinese name is normalized only when its canonical repository value is
# known; an arbitrary name is left unresolved and therefore cannot be used to
# invent a lookup parameter.
_CARRIER_MARKED = re.compile(
    r"(?:承运商|承运方|快递公司|物流公司|carrier(?:_code)?)"
    r"\s*(?:是|为|=|:|：)?\s*"
    r"([A-Za-z][A-Za-z0-9_-]{1,31}|"
    r"圆通(?:速递|快递)?|中通(?:快递)?|韵达(?:快递)?|"
    r"顺丰(?:速运|快递)?|京东(?:物流|快递)?)",
    re.IGNORECASE,
)
_KNOWN_CARRIER_CODES = frozenset({"yuantong", "zhongtong", "yunda"})
_CARRIER_ALIASES = {
    "圆通": "yuantong",
    "圆通速递": "yuantong",
    "圆通快递": "yuantong",
    "中通": "zhongtong",
    "中通快递": "zhongtong",
    "韵达": "yunda",
    "韵达快递": "yunda",
}
_SERVICE_WORDS = (("退款", "refund"), ("退货", "return"), ("换货", "exchange"))
# Words that make a logistics reading plausible in the message.
_LOGISTICS_CONTEXT = ("运单", "快递", "物流", "tracking", "单号", "追踪", "寄件", "签收")


@dataclass(frozen=True)
class EntityCandidate:
    kind: str
    value: str
    span: tuple[int, int]
    rule: str


def extract_candidates(text: str) -> list[EntityCandidate]:
    """Return deterministic candidates in text order (deduplicated by kind+value)."""
    found: list[EntityCandidate] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, value: str, span: tuple[int, int], rule: str) -> None:
        key = (kind, value)
        if key in seen:
            return
        seen.add(key)
        found.append(EntityCandidate(kind=kind, value=value, span=span, rule=rule))

    for match in _ORDER_ID.finditer(text):
        add("order_id", match.group(1), match.span(1), "order_id_regex")
    for match in _SKU.finditer(text):
        add("sku", match.group(0).upper(), match.span(0), "sku_regex")
    for match in _PHONE.finditer(text):
        add("phone_last4", match.group(1), match.span(1), "phone_hint")
    # Carrier names/codes are extracted only from explicit user text.  This is
    # evidence of what the user stated, not a lookup or a fallback based on a
    # tracking number.  Marked values are kept as supplied (lower-cased for
    # ASCII codes); well-known Chinese aliases map to the canonical code used
    # by the demo snapshot.
    for match in _CARRIER_MARKED.finditer(text):
        raw = match.group(1)
        value = _CARRIER_ALIASES.get(raw, raw.lower() if raw.isascii() else raw)
        add("carrier_code", value, match.span(1), "carrier_explicit_marker")
    if any(token in text for token in _LOGISTICS_CONTEXT):
        lowered = text.lower()
        for code in sorted(_KNOWN_CARRIER_CODES):
            for match in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])", lowered):
                add("carrier_code", code, match.span(0), "carrier_code_token")
        for alias, code in sorted(_CARRIER_ALIASES.items(), key=lambda item: (-len(item[0]), item[0])):
            index = text.find(alias)
            if index >= 0:
                add("carrier_code", code, (index, index + len(alias)), "carrier_name_token")
    for word, service in _SERVICE_WORDS:
        index = text.find(word)
        if index >= 0:
            add("service", service, (index, index + len(word)), "service_keyword")
    # Tracking numbers are only plausible when a logistics context word is present;
    # this avoids treating amounts or order ids as tracking numbers.
    if any(token in text for token in _LOGISTICS_CONTEXT):
        for match in _TRACKING.finditer(text):
            value = match.group(1)
            if _ORDER_ID.fullmatch(value):
                continue
            add("tracking_no", value, match.span(1), "tracking_regex")
    return found


def candidates_by_kind(candidates: list[EntityCandidate]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for candidate in candidates:
        out.setdefault(candidate.kind, [])
        if candidate.value not in out[candidate.kind]:
            out[candidate.kind].append(candidate.value)
    return out


def ambiguous_kinds(candidates: list[EntityCandidate]) -> dict[str, list[str]]:
    """Kinds with more than one distinct value (only ambiguity-worthy kinds)."""
    grouped = candidates_by_kind(candidates)
    return {kind: values for kind, values in grouped.items() if kind in AMBIGUITY_KINDS and len(values) > 1}


def single_values(candidates: list[EntityCandidate]) -> dict[str, str]:
    grouped = candidates_by_kind(candidates)
    return {kind: values[0] for kind, values in grouped.items() if len(values) == 1}


__all__ = ["AMBIGUITY_KINDS", "EntityCandidate", "ambiguous_kinds", "candidates_by_kind", "extract_candidates", "single_values"]
