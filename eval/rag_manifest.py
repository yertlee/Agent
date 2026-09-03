"""Deterministic, dependency-light KB manifest and hybrid retrieval.

The runtime RAG implementation is deliberately not imported here.  This
module is the versioned evaluation boundary: source/chunk IDs are derived from
canonical text, retrieval uses a fixed k=5 RRF protocol, and retrieved
instructions are represented as data only.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent.domain.objects import canonical_json, sha256_json

TOP_K = 5
DEFAULT_RRF_CONSTANT = 60


def rrf_score(*, dense_rank: int | None = None, sparse_rank: int | None = None, rrf_constant: int = DEFAULT_RRF_CONSTANT) -> float:
    """Canonical reciprocal-rank fusion score (ranks are one-based)."""
    if rrf_constant < 0: raise ValueError("rrf_constant must be non-negative")
    return sum(1.0 / (rrf_constant + rank) for rank in (dense_rank, sparse_rank) if rank is not None and rank > 0)


def normalize_text(value: str) -> str:
    """Canonical text form used for source and chunk hashes."""
    value = unicodedata.normalize("NFKC", str(value)).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")).strip()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KBSource:
    source_id: str
    source_uri: str
    content_hash: str
    text: str = field(repr=False)


@dataclass(frozen=True)
class KBChunk:
    chunk_id: str
    source_id: str
    source_uri: str
    section: str
    text: str
    text_hash: str
    source_hash: str
    conflict_group: str | None = None
    claim_key: str | None = None


@dataclass(frozen=True)
class KBManifest:
    kb_version: str
    manifest_version: str
    sources: tuple[KBSource, ...]
    chunks: tuple[KBChunk, ...]
    embedding_model: str
    tokenizer: str
    builder_version: str
    top_k: int = TOP_K
    rrf_constant: int = DEFAULT_RRF_CONSTANT
    candidate_k: int = 20
    score_threshold: float = 0.0
    checksum: str = ""

    def __post_init__(self) -> None:
        if self.top_k != TOP_K:
            raise ValueError("M4 RAG protocol fixes top_k=5")
        if self.rrf_constant < 0 or self.candidate_k < TOP_K:
            raise ValueError("invalid deterministic RRF parameters")
        object.__setattr__(self, "checksum", self.compute_checksum())

    def projection(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "kb_version": self.kb_version,
            "embedding_model": self.embedding_model,
            "tokenizer": self.tokenizer,
            "builder_version": self.builder_version,
            "top_k": self.top_k,
            "rrf_constant": self.rrf_constant,
            "candidate_k": self.candidate_k,
            "score_threshold": self.score_threshold,
            "sources": [{"source_id": s.source_id, "source_uri": s.source_uri, "content_hash": s.content_hash} for s in self.sources],
            "chunks": [{"chunk_id": c.chunk_id, "source_id": c.source_id, "source_uri": c.source_uri,
                        "section": c.section, "text_hash": c.text_hash, "source_hash": c.source_hash,
                        "conflict_group": c.conflict_group, "claim_key": c.claim_key} for c in self.chunks],
        }

    def compute_checksum(self) -> str:
        return sha256_json(self.projection())

    def as_dict(self, include_text: bool = False) -> dict[str, Any]:
        out = self.projection()
        if include_text:
            out["chunks"] = [dict(item, text=c.text) for item, c in zip(out["chunks"], self.chunks)]
        return out


def _split_source(source_id: str, uri: str, text: str) -> list[tuple[str, str]]:
    """Split markdown by headings and paragraphs while retaining section."""
    section = "root"
    paragraphs: list[tuple[str, str]] = []
    buf: list[str] = []
    def flush() -> None:
        nonlocal buf
        value = normalize_text("\n".join(buf))
        if value:
            paragraphs.append((section, value))
        buf = []
    for line in text.splitlines():
        match = re.match(r"^\s*#{1,6}\s+(.*?)\s*$", line)
        if match:
            flush(); section = normalize_text(match.group(1)) or section
        elif not line.strip():
            flush()
        else:
            buf.append(line)
    flush()
    if not paragraphs and normalize_text(text):
        paragraphs = [(section, normalize_text(text))]
    return paragraphs


def _source_metadata(path: Path, raw: str) -> tuple[str, dict[str, Any]]:
    """Read optional frontmatter or adjacent ``.meta.json`` metadata."""
    body = raw
    metadata: dict[str, Any] = {}
    if raw.lstrip().startswith("---"):
        lines = raw.splitlines()
        try:
            end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
            metadata = dict(__import__("yaml").safe_load("\n".join(lines[1:end])) or {})
            body = "\n".join(lines[end + 1:])
        except (StopIteration, TypeError, ValueError):
            pass
    sidecar = path.with_suffix(".meta.json")
    if sidecar.is_file():
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping): metadata.update(loaded)
    return body, metadata


def build_kb_manifest(kb_dir: str | Path, *, kb_version: str = "kb.m4.v1",
                      manifest_version: str = "kb-manifest.m4.v1",
                      embedding_model: str = "deterministic-dense.v1", tokenizer: str = "unicode-nfkc.v1",
                      builder_version: str = "kb-builder.m4.v1", rrf_constant: int = DEFAULT_RRF_CONSTANT,
                      candidate_k: int = 20, score_threshold: float = 0.0) -> KBManifest:
    root = Path(kb_dir)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise FileNotFoundError(f"no KB sources under {root}")
    sources: list[KBSource] = []
    chunks: list[KBChunk] = []
    seen_chunk: dict[tuple[str, str, str], int] = {}
    for path in files:
        raw = path.read_text(encoding="utf-8")
        body, metadata = _source_metadata(path, raw)
        normalized = normalize_text(body)
        source_id = _hash(str(path.relative_to(root)).replace("\\", "/"))[:24]
        source_hash = _hash(normalized)
        uri = str(path.relative_to(root)).replace("\\", "/")
        sources.append(KBSource(source_id, uri, source_hash, normalized))
        for section, body in _split_source(source_id, uri, normalized):
            text_hash = _hash(body)
            key = (source_id, section, text_hash)
            ordinal = seen_chunk.get(key, 0)
            seen_chunk[key] = ordinal + 1
            chunk_id = _hash(f"{source_id}\n{section}\n{text_hash}\n{ordinal}")[:32]
            chunks.append(KBChunk(chunk_id, source_id, uri, section, body, text_hash, source_hash,
                                  str(metadata.get("conflict_group")) if metadata.get("conflict_group") is not None else None,
                                  str(metadata.get("claim_key")) if metadata.get("claim_key") is not None else None))
    return KBManifest(kb_version, manifest_version, tuple(sources), tuple(chunks), embedding_model,
                      tokenizer, builder_version, TOP_K, rrf_constant, candidate_k, score_threshold)


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    source_id: str
    source_uri: str
    text: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    injection_data_only: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    status: str
    hits: tuple[RetrievalHit, ...]
    conflict_set: tuple[str, ...] = ()
    injection_data_only: bool = True
    query: str = ""
    top_k: int = TOP_K
    safe_tool_args: Mapping[str, Any] = field(default_factory=dict)


def _terms(value: str) -> set[str]:
    text = normalize_text(value).lower()
    words = set(re.findall(r"[\w\u4e00-\u9fff]+", text))
    for word in list(words):
        if len(word) >= 4 and re.search(r"[\u4e00-\u9fff]", word):
            words.update(word[i:i + 2] for i in range(len(word) - 1))
    return {x for x in words if x}


def _sparse_score(query: str, text: str) -> float:
    q, t = _terms(query), _terms(text)
    return len(q & t) / max(len(q), 1)


def _dense_score(query: str, text: str) -> float:
    """Deterministic dense-like cosine over hashed character n-grams.

    This is a reproducible local stand-in for an embedding adapter; production
    embeddings are still versioned in ``embedding_model`` and produce a new
    manifest when changed.
    """
    def vector(value: str) -> list[float]:
        compact = normalize_text(value).lower()
        grams = [compact[i:i + 3] for i in range(max(0, len(compact) - 2))] or [compact]
        vec = [0.0] * 64
        for gram in grams:
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:2], "big") % 64] += 1.0 if digest[2] & 1 else -1.0
        return vec
    a, b = vector(query), vector(text)
    den = (sum(x * x for x in a) * sum(x * x for x in b)) ** 0.5
    return max(0.0, sum(x * y for x, y in zip(a, b)) / den) if den else 0.0


class HybridRetriever:
    """Deterministic dense/sparse adapter; dense scores may be supplied by a model."""
    def __init__(self, manifest: KBManifest):
        self.manifest = manifest
        self._by_id = {c.chunk_id: c for c in manifest.chunks}

    def retrieve(self, query: str, *, dense: Sequence[Any] | Mapping[str, float] | None = None,
                 sparse: Sequence[Any] | Mapping[str, float] | None = None,
                 top_k: int = TOP_K, score_threshold: float | None = None,
                 tool_args: Mapping[str, Any] | None = None) -> RetrievalResult:
        if top_k != TOP_K:
            raise ValueError("M4 RAG protocol fixes top_k=5")
        threshold = self.manifest.score_threshold if score_threshold is None else score_threshold
        def ranked(values: Sequence[Any] | Mapping[str, float] | None, fallback: str) -> list[tuple[str, float]]:
            if values is None:
                values = ({c.chunk_id: _dense_score(query, f"{c.section}\n{c.text}") for c in self.manifest.chunks}
                          if fallback == "dense" else {c.chunk_id: _sparse_score(query, f"{c.section}\n{c.text}") for c in self.manifest.chunks})
            if isinstance(values, Mapping):
                minimum = 0.25 if fallback == "dense" else 0.0
                rows = [(str(k), float(v)) for k, v in values.items() if str(k) in self._by_id and (float(v) >= minimum if fallback == "dense" else float(v) > minimum)]
                return sorted(rows, key=lambda x: (-x[1], x[0]))[: self.manifest.candidate_k]
            rows: list[tuple[str, float]] = []
            for index, item in enumerate(values):
                if isinstance(item, Mapping):
                    cid = str(item.get("chunk_id")); score = float(item.get("score", 0.0))
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    cid, score = str(item[0]), float(item[1])
                else:
                    cid, score = str(item), float(len(values) - index)
                if cid in self._by_id and score > 0.0: rows.append((cid, score))
            return sorted(rows, key=lambda x: (-x[1], x[0]))[: self.manifest.candidate_k]
        dense_rows, sparse_rows = ranked(dense, "dense"), ranked(sparse, "sparse")
        d_rank = {cid: i for i, (cid, _) in enumerate(dense_rows, 1)}
        s_rank = {cid: i for i, (cid, _) in enumerate(sparse_rows, 1)}
        ids = set(d_rank) | set(s_rank)
        scored = [(cid, rrf_score(dense_rank=d_rank.get(cid), sparse_rank=s_rank.get(cid), rrf_constant=self.manifest.rrf_constant)) for cid in ids]
        scored = sorted(scored, key=lambda x: (-x[1], x[0]))
        hits: list[RetrievalHit] = []
        for cid, score in scored:
            if score < threshold: continue
            chunk = self._by_id[cid]
            # Potential prompt injection is a property of evidence data, never
            # an executable instruction.  No caller arguments are accepted here.
            injection = bool(re.search(r"(?i)(ignore\s+(previous|all)\s+instructions|system\s+prompt|tool\s+call)", chunk.text))
            hits.append(RetrievalHit(cid, chunk.source_id, chunk.source_uri, chunk.text, score, d_rank.get(cid), s_rank.get(cid), injection))
            if len(hits) == TOP_K: break
        groups = {self._by_id[h.chunk_id].conflict_group for h in hits if self._by_id[h.chunk_id].conflict_group}
        claims = {self._by_id[h.chunk_id].claim_key for h in hits if self._by_id[h.chunk_id].claim_key}
        conflict = tuple(sorted(str(g) for g in groups)) if len(groups) > 1 or len(claims) > 1 else ()
        status = "no_hits" if not hits else ("conflict" if conflict else ("strong" if hits[0].dense_rank is not None and hits[0].sparse_rank is not None else "weak"))
        # ``tool_args`` is echoed only as an immutable observational projection;
        # evidence text can never mutate or supply execution arguments.
        return RetrievalResult(status, tuple(hits), conflict, True, query, TOP_K, dict(tool_args or {}))


def hybrid_retrieve(manifest: KBManifest, query: str, **kwargs: Any) -> RetrievalResult:
    return HybridRetriever(manifest).retrieve(query, **kwargs)


def write_kb_manifest(manifest: KBManifest, path: str | Path) -> str:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.as_dict(include_text=True), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(target)


def load_kb_manifest(path: str | Path) -> KBManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = tuple(KBSource(str(x["source_id"]), str(x["source_uri"]), str(x["content_hash"]), str(x.get("text", ""))) for x in raw.get("sources", ()))
    chunks = tuple(KBChunk(str(x["chunk_id"]), str(x["source_id"]), str(x["source_uri"]), str(x.get("section", "root")),
                          str(x.get("text", "")), str(x["text_hash"]), str(x["source_hash"]), x.get("conflict_group"), x.get("claim_key"))
                   for x in raw.get("chunks", ()))
    manifest = KBManifest(str(raw["kb_version"]), str(raw["manifest_version"]), sources, chunks,
                          str(raw["embedding_model"]), str(raw["tokenizer"]), str(raw["builder_version"]),
                          int(raw.get("top_k", TOP_K)), int(raw.get("rrf_constant", DEFAULT_RRF_CONSTANT)),
                          int(raw.get("candidate_k", 20)), float(raw.get("score_threshold", 0.0)))
    if raw.get("checksum") and str(raw["checksum"]) != manifest.checksum:
        raise ValueError("KB manifest checksum mismatch")
    return manifest


__all__ = ["DEFAULT_RRF_CONSTANT", "KBChunk", "KBManifest", "KBSource", "HybridRetriever", "RetrievalHit", "RetrievalResult", "TOP_K", "build_kb_manifest", "hybrid_retrieve", "load_kb_manifest", "normalize_text", "rrf_score", "write_kb_manifest"]
