"""Shared R3 policy/product RAG runtime and deterministic evaluator primitives.

This module is intentionally self-contained so the runtime and evaluator use the
same corpus, candidate generation, fusion, reranking, and safety decisions.  It
does not call an LLM.  Dense retrieval uses sentence-transformers when the
configured model is locally available; an unavailable model is reported as an
explicit engineering failure rather than replaced by a fake embedding.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TOP_K = 5
DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_K = 60
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
GENERIC_STOPWORDS = {
    "本", "资料", "明确", "标记", "为", "project", "authored", "demo", "corpus", "不", "代表",
    "任何", "外部", "平台", "当前", "有效", "规则", "演示", "场景", "商品", "项目", "自建",
}


class R3Error(RuntimeError):
    """Base class for explicit R3 engineering failures."""


class DenseModelUnavailable(R3Error):
    """The required local sentence-transformers model cannot be loaded."""


class CorpusIntegrityError(R3Error):
    """The frozen source or manifest checksum does not match."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.replace("\r\n", "\n").replace("\r", "\n").split())


def tokenize(value: str) -> list[str]:
    lowered = value.lower()
    for stopword in GENERIC_STOPWORDS:
        lowered = lowered.replace(stopword, " ")
    output: list[str] = []
    for token in TOKEN_RE.findall(lowered):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            output.extend(token[i : i + 2] for i in range(len(token) - 1))
        else:
            output.append(token)
    return output


def _parse_day(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    document_type: str
    version: str
    effective_from: date
    effective_to: date | None
    scope: str
    claim_key: str
    path: str
    checksum: str

    def active_at(self, as_of: date) -> bool:
        return self.effective_from <= as_of and (self.effective_to is None or as_of <= self.effective_to)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    version: str
    text: str
    text_hash: str
    locator: str
    effective_from: date
    effective_to: date | None
    scope: str
    claim_key: str

    def active_at(self, as_of: date) -> bool:
        return self.effective_from <= as_of and (self.effective_to is None or as_of <= self.effective_to)


@dataclass(frozen=True)
class Manifest:
    manifest_version: str
    corpus_version: str
    builder_version: str
    tokenizer_version: str
    embedding_model: str
    reranker_version: str
    top_k: int
    candidate_k: int
    rrf_k: int
    bm25_min_score: float
    dense_min_score: float
    min_lexical_overlap: int
    min_query_coverage: float
    sources: tuple[SourceRecord, ...]
    chunks: tuple[ChunkRecord, ...]
    manifest_checksum: str

    @property
    def version_tuple(self) -> tuple[str, ...]:
        return (
            self.manifest_version,
            self.corpus_version,
            self.builder_version,
            self.tokenizer_version,
            self.embedding_model,
            self.reranker_version,
            str(self.top_k),
            str(self.candidate_k),
            str(self.rrf_k),
            str(self.bm25_min_score),
            str(self.dense_min_score),
            str(self.min_lexical_overlap),
            str(self.min_query_coverage),
            self.manifest_checksum,
        )


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    source_id: str
    version: str
    chunk_id: str
    text_hash: str
    locator: str
    score: float
    effective_from: date
    effective_to: date | None


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalHit:
    chunk: ChunkRecord
    score: float
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0


@dataclass
class RetrievalResult:
    status: str
    query: str
    mode: str
    hits: list[RetrievalHit] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    conflict_set: list[str] = field(default_factory=list)
    failure_code: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.status == "ANSWERED"


def _source_projection(source: SourceRecord) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "document_type": source.document_type,
        "version": source.version,
        "effective_from": source.effective_from.isoformat(),
        "effective_to": source.effective_to.isoformat() if source.effective_to else None,
        "scope": source.scope,
        "claim_key": source.claim_key,
        "path": source.path,
        "checksum": source.checksum,
    }


def _chunk_projection(chunk: ChunkRecord) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "version": chunk.version,
        "text_hash": chunk.text_hash,
        "locator": chunk.locator,
        "effective_from": chunk.effective_from.isoformat(),
        "effective_to": chunk.effective_to.isoformat() if chunk.effective_to else None,
        "scope": chunk.scope,
        "claim_key": chunk.claim_key,
    }


def _manifest_checksum(raw: Mapping[str, Any], sources: Sequence[SourceRecord], chunks: Sequence[ChunkRecord]) -> str:
    projection = {
        "manifest_version": raw["manifest_version"],
        "corpus_version": raw["corpus_version"],
        "builder_version": raw["builder_version"],
        "tokenizer_version": raw["tokenizer_version"],
        "embedding_model": raw["embedding_model"],
        "reranker_version": raw["reranker_version"],
        "top_k": int(raw.get("top_k", TOP_K)),
        "candidate_k": int(raw.get("candidate_k", DEFAULT_CANDIDATE_K)),
        "rrf_k": int(raw.get("rrf_k", DEFAULT_RRF_K)),
        "bm25_min_score": float(raw.get("bm25_min_score", 0.5)),
        "dense_min_score": float(raw.get("dense_min_score", 0.30)),
        "min_lexical_overlap": int(raw.get("min_lexical_overlap", 1)),
        "min_query_coverage": float(raw.get("min_query_coverage", 0.30)),
        "sources": [_source_projection(s) for s in sources],
        "chunks": [_chunk_projection(c) for c in chunks],
    }
    return sha256_text(json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _split_source(text: str) -> list[tuple[str, str]]:
    """Split on markdown headings/paragraphs without using query-specific rules."""
    sections: list[tuple[str, str]] = []
    section = "document"
    buf: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("#"):
            if normalize_text("\n".join(buf)):
                sections.append((section, normalize_text("\n".join(buf))))
            section = line.lstrip("# ").strip() or "document"
            buf = []
        elif line.strip():
            buf.append(line.strip())
        elif buf:
            sections.append((section, normalize_text("\n".join(buf))))
            buf = []
    if normalize_text("\n".join(buf)):
        sections.append((section, normalize_text("\n".join(buf))))
    return sections


def load_manifest(path: str | Path, *, verify: bool = True) -> Manifest:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    sources: list[SourceRecord] = []
    chunks: list[ChunkRecord] = []
    for item in raw["sources"]:
        source_path = root / item["path"]
        content = source_path.read_text(encoding="utf-8")
        content_checksum = sha256_text(normalize_text(content))
        if verify and content_checksum != item["checksum"]:
            raise CorpusIntegrityError(f"SOURCE_CHECKSUM_MISMATCH:{item['source_id']}")
        source = SourceRecord(
            source_id=item["source_id"],
            document_type=item["document_type"],
            version=item["version"],
            effective_from=_parse_day(item["effective_from"]) or date.min,
            effective_to=_parse_day(item.get("effective_to")),
            scope=item["scope"],
            claim_key=item["claim_key"],
            path=item["path"],
            checksum=item["checksum"],
        )
        sources.append(source)
        for ordinal, (section, text) in enumerate(_split_source(content)):
            text_hash = sha256_text(normalize_text(text))
            chunk_id = sha256_text(f"{source.source_id}|{source.version}|{section}|{text_hash}|{ordinal}")[:24]
            chunks.append(ChunkRecord(
                chunk_id=chunk_id,
                source_id=source.source_id,
                version=source.version,
                text=text,
                text_hash=text_hash,
                locator=f"{source.path}#{section}:{ordinal}",
                effective_from=source.effective_from,
                effective_to=source.effective_to,
                scope=source.scope,
                claim_key=source.claim_key,
            ))
    checksum = _manifest_checksum(raw, sources, chunks)
    if verify and raw.get("manifest_checksum") != checksum:
        raise CorpusIntegrityError("MANIFEST_CHECKSUM_MISMATCH")
    if int(raw.get("top_k", TOP_K)) != TOP_K:
        raise CorpusIntegrityError("TOP_K_MUST_BE_5")
    return Manifest(
        manifest_version=raw["manifest_version"], corpus_version=raw["corpus_version"],
        builder_version=raw["builder_version"], tokenizer_version=raw["tokenizer_version"],
        embedding_model=raw["embedding_model"], reranker_version=raw["reranker_version"],
        top_k=TOP_K, candidate_k=int(raw.get("candidate_k", DEFAULT_CANDIDATE_K)),
        rrf_k=int(raw.get("rrf_k", DEFAULT_RRF_K)),
        bm25_min_score=float(raw.get("bm25_min_score", 0.5)),
        dense_min_score=float(raw.get("dense_min_score", 0.30)),
        min_lexical_overlap=int(raw.get("min_lexical_overlap", 1)),
        min_query_coverage=float(raw.get("min_query_coverage", 0.30)),
        sources=tuple(sources),
        chunks=tuple(chunks), manifest_checksum=checksum,
    )


class BM25Index:
    def __init__(self, chunks: Sequence[ChunkRecord]) -> None:
        self.chunks = tuple(chunks)
        self.tokens = [tokenize(c.text) for c in self.chunks]
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        self.df: dict[str, int] = {}
        for toks in self.tokens:
            for token in set(toks):
                self.df[token] = self.df.get(token, 0) + 1

    def score(self, query: str, chunk: ChunkRecord) -> float:
        try:
            idx = next(i for i, c in enumerate(self.chunks) if c.chunk_id == chunk.chunk_id)
        except StopIteration:
            return 0.0
        q = tokenize(query)
        tf: dict[str, int] = {}
        for token in self.tokens[idx]:
            tf[token] = tf.get(token, 0) + 1
        n = len(self.chunks)
        dl = len(self.tokens[idx])
        value = 0.0
        for token in q:
            if token not in tf:
                continue
            idf = math.log(1.0 + (n - self.df.get(token, 0) + 0.5) / (self.df.get(token, 0) + 0.5))
            value += idf * (tf[token] * 2.0) / (tf[token] + 1.5 * (0.75 + 0.25 * dl / max(self.avgdl, 1.0)))
        return value

    def rank(self, query: str, chunks: Sequence[ChunkRecord] | None = None) -> list[tuple[ChunkRecord, float]]:
        pool = tuple(chunks or self.chunks)
        return sorted(((c, self.score(query, c)) for c in pool), key=lambda x: (-x[1], x[0].chunk_id))


class DenseIndex:
    def __init__(self, chunks: Sequence[ChunkRecord], model_name: str) -> None:
        self.chunks = tuple(chunks)
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, local_files_only=True)
            vectors = self.model.encode([c.text for c in self.chunks], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            self.vectors = vectors
        except Exception as exc:  # explicit engineering status, never a fake dense result
            raise DenseModelUnavailable(f"DENSE_MODEL_UNAVAILABLE:{model_name}:{type(exc).__name__}") from exc

    def rank(self, query: str, chunks: Sequence[ChunkRecord] | None = None) -> list[tuple[ChunkRecord, float]]:
        import numpy as np
        selected = tuple(chunks or self.chunks)
        query_vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        index = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        scored = [(c, float(np.dot(query_vector, self.vectors[index[c.chunk_id]]))) for c in selected]
        return sorted(scored, key=lambda x: (-x[1], x[0].chunk_id))


class GenericCoverageReranker:
    """A generic deterministic reranker; it has no corpus/case-specific mapping."""

    version = "coverage-reranker.v1"

    def rerank(self, query: str, hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
        q = set(tokenize(query))
        def key(hit: RetrievalHit) -> tuple[float, str]:
            t = set(tokenize(hit.chunk.text))
            coverage = len(q & t) / max(len(q), 1)
            score = 0.55 * max(hit.rrf_score, 0.0) + 0.30 * coverage + 0.15 * max(hit.bm25_score, 0.0)
            return score, hit.chunk.chunk_id
        output = []
        for hit in hits:
            qtokens = set(tokenize(query))
            coverage = len(qtokens & set(tokenize(hit.chunk.text))) / max(len(qtokens), 1)
            output.append(replace(hit, score=0.55 * max(hit.rrf_score, 0.0) + 0.30 * coverage + 0.15 * max(hit.bm25_score, 0.0)))
        return sorted(output, key=lambda h: (-h.score, h.chunk.chunk_id))


class R3Retriever:
    def __init__(self, manifest: Manifest, *, dense: DenseIndex | None = None, threshold: float = 0.02) -> None:
        self.manifest = manifest
        self.bm25 = BM25Index(manifest.chunks)
        self.dense = dense
        self.threshold = threshold
        self.reranker = GenericCoverageReranker()

    @classmethod
    def from_manifest(cls, path: str | Path, *, require_dense: bool = False) -> "R3Retriever":
        manifest = load_manifest(path)
        dense: DenseIndex | None = None
        try:
            dense = DenseIndex(manifest.chunks, manifest.embedding_model)
        except DenseModelUnavailable:
            if require_dense:
                raise
        return cls(manifest, dense=dense)

    def _evidence(self, hits: Sequence[RetrievalHit]) -> list[EvidenceRef]:
        return [EvidenceRef(
            evidence_id=f"ev-{h.chunk.chunk_id}", source_id=h.chunk.source_id, version=h.chunk.version,
            chunk_id=h.chunk.chunk_id, text_hash=h.chunk.text_hash, locator=h.chunk.locator,
            score=float(h.score), effective_from=h.chunk.effective_from, effective_to=h.chunk.effective_to,
        ) for h in hits]

    def retrieve(self, query: str, *, as_of: date, mode: str = "hybrid_reranker", top_k: int = TOP_K) -> RetrievalResult:
        trace: dict[str, Any] = {"query": query, "mode": mode, "as_of": as_of.isoformat(), "top_k": top_k, "version_tuple": self.manifest.version_tuple, "rewrite": {"status": "NOT_APPLIED"}}
        if top_k != TOP_K:
            return RetrievalResult("FAILED", query, mode, failure_code="TOP_K_MUST_BE_5", trace=trace)
        active = [c for c in self.manifest.chunks if c.active_at(as_of)]
        all_bm = self.bm25.rank(query, self.manifest.chunks)[: self.manifest.candidate_k]
        relevant_old = [c for c, s in all_bm if s > 0]
        query_tokens = set(tokenize(query))
        def strong_bm25(chunk: ChunkRecord, score: float) -> bool:
            coverage = len(query_tokens & set(tokenize(chunk.text))) / max(len(query_tokens), 1)
            return score >= self.manifest.bm25_min_score and coverage >= self.manifest.min_query_coverage

        active_bm = self.bm25.rank(query, active)[: self.manifest.candidate_k]
        active_relevant = [c for c, s in active_bm if strong_bm25(c, s)]
        qualified_old = [c for c, s in all_bm if strong_bm25(c, s) and not c.active_at(as_of)]
        if qualified_old and not active_relevant:
            return RetrievalResult("STALE_ONLY", query, mode, trace={**trace, "candidate_count": len(relevant_old)})
        if mode == "no_retrieval":
            return RetrievalResult("NO_HITS", query, mode, trace=trace)
        if mode == "bm25":
            ranked = [(c, s) for c, s in self.bm25.rank(query, active)[: self.manifest.candidate_k]]
            top_score = ranked[0][1] if ranked else 0.0
            ranked = [(c, s) for c, s in ranked if s >= self.manifest.bm25_min_score and s >= top_score * 0.50]
            hits = [RetrievalHit(c, s, bm25_score=s) for c, s in ranked]
        elif mode == "dense":
            if self.dense is None:
                return RetrievalResult("FAILED", query, mode, failure_code="DENSE_MODEL_UNAVAILABLE", trace=trace)
            ranked = [(c, s) for c, s in self.dense.rank(query, active)[: self.manifest.candidate_k] if s >= self.manifest.dense_min_score]
            hits = [RetrievalHit(c, s, dense_score=s) for c, s in ranked]
        else:
            if self.dense is None and mode in {"hybrid_reranker", "hybrid_no_reranker"}:
                return RetrievalResult("FAILED", query, mode, failure_code="DENSE_MODEL_UNAVAILABLE", trace=trace)
            bm = self.bm25.rank(query, active)[: self.manifest.candidate_k]
            de = self.dense.rank(query, active)[: self.manifest.candidate_k] if self.dense else []
            qtokens = set(tokenize(query))
            bm = [(c, s) for c, s in bm if s >= self.manifest.bm25_min_score and len(qtokens & set(tokenize(c.text))) >= self.manifest.min_lexical_overlap]
            de = [(c, s) for c, s in de if s >= self.manifest.dense_min_score]
            bm_pos = {c.chunk_id: (i + 1, s) for i, (c, s) in enumerate(bm)}
            de_pos = {c.chunk_id: (i + 1, s) for i, (c, s) in enumerate(de)}
            chunks = {c.chunk_id: c for c in active}
            fused: list[RetrievalHit] = []
            for cid, chunk in chunks.items():
                rrf = (1.0 / (self.manifest.rrf_k + bm_pos[cid][0]) if cid in bm_pos else 0.0) + (1.0 / (self.manifest.rrf_k + de_pos[cid][0]) if cid in de_pos else 0.0)
                fused.append(RetrievalHit(chunk, rrf, bm_pos.get(cid, (0, 0.0))[1], de_pos.get(cid, (0, 0.0))[1], rrf))
            hits = sorted(fused, key=lambda h: (-h.rrf_score, h.chunk.chunk_id))[: self.manifest.candidate_k]
            if mode == "hybrid_reranker":
                hits = self.reranker.rerank(query, hits)
        qtokens = set(tokenize(query))
        hits = [
            h for h in hits
            if h.bm25_score >= self.manifest.bm25_min_score
            or (h.dense_score >= self.manifest.dense_min_score and len(qtokens & set(tokenize(h.chunk.text))) >= self.manifest.min_lexical_overlap + 1)
        ][:TOP_K]
        if not hits or hits[0].score < self.threshold:
            return RetrievalResult("WEAK_EVIDENCE" if hits else "NO_HITS", query, mode, hits=hits, evidence=self._evidence(hits), trace={**trace, "candidate_count": len(hits)})
        groups: dict[str, list[RetrievalHit]] = {}
        for hit in hits:
            groups.setdefault(hit.chunk.claim_key, []).append(hit)
        conflicts = [
            key for key, values in groups.items()
            if len({(v.chunk.source_id, v.chunk.version) for v in values}) > 1
            and len({v.chunk.text_hash for v in values}) > 1
        ]
        evidence = self._evidence(hits)
        if conflicts:
            return RetrievalResult("CONFLICT", query, mode, hits=hits, evidence=evidence, conflict_set=conflicts, trace={**trace, "candidate_count": len(hits)})
        claims = [Claim(claim_id=f"claim-{h.chunk.chunk_id}", text=h.chunk.text, evidence_ids=(f"ev-{h.chunk.chunk_id}",)) for h in hits[:1]]
        return RetrievalResult("ANSWERED", query, mode, hits=hits, evidence=evidence, claims=claims, trace={**trace, "candidate_count": len(hits), "claim_evidence_valid": True})


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def mutate_chunk_ids(manifest: Manifest, seed: int = 17) -> Manifest:
    """Return a deterministic ID-remapped/order-shuffled mutation for audit tests."""
    rng = random.Random(seed)
    mapping = {c.chunk_id: sha256_text(f"mutation:{seed}:{c.chunk_id}")[:24] for c in manifest.chunks}
    chunks = [replace(c, chunk_id=mapping[c.chunk_id]) for c in manifest.chunks]
    rng.shuffle(chunks)
    return replace(manifest, chunks=tuple(chunks), manifest_checksum=sha256_text(f"mutated:{manifest.manifest_checksum}:{seed}"))


def build_claim_evidence(result: RetrievalResult) -> bool:
    if not result.claims:
        return False
    evidence_ids = {e.evidence_id for e in result.evidence}
    return all(set(claim.evidence_ids) <= evidence_ids for claim in result.claims)
