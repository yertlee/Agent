"""R3.5 retrieval runtime with isolated, auditable baseline components.

The evaluator and any future caller use this module.  Each mode owns its
candidate provider; metadata time filtering never invokes a second retriever.
This module has no case, gold, or answer-specific rules and never calls an
external model.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from .r3_rag import (
    BM25Index,
    Claim,
    DenseIndex,
    DenseModelUnavailable,
    EvidenceRef,
    GenericCoverageReranker,
    Manifest,
    RetrievalHit,
    RetrievalResult,
    R3Error,
    _parse_day,
    load_manifest,
    sha256_text,
    tokenize,
)

R35_MODES = (
    "no_retrieval", "random", "bm25", "dense", "hybrid_no_reranker",
    "hybrid_reranker", "oracle", "remove_version_filter",
    "remove_abstention", "mutation",
)
R35_COMPONENT_MATRIX = {
    "no_retrieval": {"corpus_access": False, "candidate_provider": "none", "metadata_time_filter": False, "abstention": "fixed_refusal", "reranker": None},
    "random": {"corpus_access": True, "candidate_provider": "seeded_random", "metadata_time_filter": True, "abstention": False, "reranker": None},
    "bm25": {"corpus_access": True, "candidate_provider": "bm25", "metadata_time_filter": True, "abstention": True, "reranker": None},
    "dense": {"corpus_access": True, "candidate_provider": "dense", "metadata_time_filter": True, "abstention": True, "reranker": None},
    "hybrid_no_reranker": {"corpus_access": True, "candidate_provider": "bm25+dense+rrf", "metadata_time_filter": True, "abstention": True, "reranker": None},
    "hybrid_reranker": {"corpus_access": True, "candidate_provider": "bm25+dense+rrf", "metadata_time_filter": True, "abstention": True, "reranker": "coverage-reranker.v1"},
    "oracle": {"corpus_access": False, "candidate_provider": "gold", "metadata_time_filter": "gold", "abstention": "gold", "reranker": None},
    "remove_version_filter": {"corpus_access": True, "candidate_provider": "bm25+dense+rrf", "metadata_time_filter": False, "abstention": True, "reranker": "coverage-reranker.v1"},
    "remove_abstention": {"corpus_access": True, "candidate_provider": "bm25+dense+rrf", "metadata_time_filter": True, "abstention": False, "reranker": "coverage-reranker.v1"},
    "mutation": {"corpus_access": True, "candidate_provider": "bm25+dense+rrf", "metadata_time_filter": True, "abstention": True, "reranker": "coverage-reranker.v1"},
}


class R35ManifestError(R3Error):
    """R3.5 strategy metadata is missing or does not match its checksum."""


def _strategy_projection(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "r35_version": raw["r35_version"],
        "evidence_threshold": float(raw["evidence_threshold"]),
        "time_strategy": raw["time_strategy"],
        "abstention_version": raw["abstention_version"],
        "component_matrix": raw["component_matrix"],
    }


def strategy_checksum(raw: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(_strategy_projection(raw), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class R35Manifest:
    base: Manifest
    r35_version: str
    evidence_threshold: float
    time_strategy: str
    abstention_version: str
    component_matrix: Mapping[str, Mapping[str, Any]]
    strategy_checksum: str

    @property
    def chunks(self):
        return self.base.chunks

    @property
    def sources(self):
        return self.base.sources

    @property
    def top_k(self) -> int:
        return self.base.top_k

    @property
    def candidate_k(self) -> int:
        return self.base.candidate_k

    @property
    def rrf_k(self) -> int:
        return self.base.rrf_k

    @property
    def version_tuple(self) -> tuple[str, ...]:
        return (*self.base.version_tuple, self.r35_version, str(self.evidence_threshold), self.time_strategy, self.abstention_version, self.strategy_checksum)


def load_r35_manifest(path: str | Path, *, verify: bool = True) -> R35Manifest:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {"r35_version", "evidence_threshold", "time_strategy", "abstention_version", "component_matrix", "strategy_checksum"}
    if not required <= set(raw):
        raise R35ManifestError("R35_STRATEGY_METADATA_MISSING")
    if verify and raw["strategy_checksum"] != strategy_checksum(raw):
        raise R35ManifestError("R35_STRATEGY_CHECKSUM_MISMATCH")
    base = load_manifest(path, verify=verify)
    matrix = raw["component_matrix"]
    if set(matrix) != set(R35_COMPONENT_MATRIX) or any(matrix[k] != R35_COMPONENT_MATRIX[k] for k in R35_COMPONENT_MATRIX):
        raise R35ManifestError("R35_COMPONENT_MATRIX_MISMATCH")
    return R35Manifest(
        base=base,
        r35_version=str(raw["r35_version"]),
        evidence_threshold=float(raw["evidence_threshold"]),
        time_strategy=str(raw["time_strategy"]),
        abstention_version=str(raw["abstention_version"]),
        component_matrix=matrix,
        strategy_checksum=str(raw["strategy_checksum"]),
    )


def mutate_manifest(manifest: R35Manifest, *, seed: int = 17) -> tuple[R35Manifest, dict[str, str]]:
    rng = random.Random(seed)
    mapping = {c.chunk_id: sha256_text(f"r35-mutation:{seed}:{c.chunk_id}")[:24] for c in manifest.chunks}
    chunks = [replace(c, chunk_id=mapping[c.chunk_id]) for c in manifest.chunks]
    rng.shuffle(chunks)
    base = replace(manifest.base, chunks=tuple(chunks), manifest_checksum=sha256_text(f"r35-mutated:{manifest.base.manifest_checksum}:{seed}"))
    return replace(manifest, base=base, r35_version=f"{manifest.r35_version}.mutation.{seed}"), mapping


class R35Retriever:
    """R3.5 shared runtime; mode choice controls which providers may run."""

    def __init__(self, manifest: R35Manifest, *, seed: int = 71) -> None:
        self.manifest = manifest
        self.seed = seed
        self._bm25: BM25Index | None = None
        self._dense: DenseIndex | None = None
        self._reranker = GenericCoverageReranker()

    @classmethod
    def from_manifest(cls, path: str | Path, *, seed: int = 71) -> "R35Retriever":
        return cls(load_r35_manifest(path), seed=seed)

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.manifest.chunks)
        return self._bm25

    @property
    def dense(self) -> DenseIndex:
        if self._dense is None:
            self._dense = DenseIndex(self.manifest.chunks, self.manifest.base.embedding_model)
        return self._dense

    def _trace(self, query: str, mode: str, as_of: date, calls: list[str], **extra: Any) -> dict[str, Any]:
        return {
            "query": query,
            "mode": mode,
            "as_of": as_of.isoformat(),
            "top_k": self.manifest.top_k,
            "version_tuple": self.manifest.version_tuple,
            "strategy_checksum": self.manifest.strategy_checksum,
            "component_matrix": self.manifest.component_matrix.get(mode, {}),
            "components_called": list(calls),
            "evidence_ids": [],
            "scores": [],
            "provider_called": False,
            "provider_returned": False,
            "model": None,
            "latency_ms": 0,
            "schema_error": None,
            "grounding_pass": None,
            "error": None,
            **extra,
        }

    def _evidence(self, hits: Sequence[RetrievalHit]) -> list[EvidenceRef]:
        return [EvidenceRef(
            evidence_id=f"ev-{h.chunk.chunk_id}", source_id=h.chunk.source_id, version=h.chunk.version,
            chunk_id=h.chunk.chunk_id, text_hash=h.chunk.text_hash, locator=h.chunk.locator,
            score=float(h.score), effective_from=h.chunk.effective_from, effective_to=h.chunk.effective_to,
        ) for h in hits]

    def _metadata_active(self, as_of: date, enabled: bool) -> list[Any]:
        return [c for c in self.manifest.chunks if not enabled or c.active_at(as_of)]

    def _bm25_candidates(self, query: str, pool: Sequence[Any], calls: list[str]) -> list[tuple[Any, float]]:
        calls.append("bm25")
        ranked = self.bm25.rank(query, pool)[: self.manifest.candidate_k]
        top = ranked[0][1] if ranked else 0.0
        return [(c, s) for c, s in ranked if s >= self.manifest.base.bm25_min_score and s >= top * 0.50]

    def _dense_candidates(self, query: str, pool: Sequence[Any], calls: list[str]) -> list[tuple[Any, float]]:
        calls.append("dense")
        return [(c, s) for c, s in self.dense.rank(query, pool)[: self.manifest.candidate_k] if s >= self.manifest.base.dense_min_score]

    def _hybrid_candidates(self, query: str, pool: Sequence[Any], calls: list[str]) -> list[RetrievalHit]:
        bm = self._bm25_candidates(query, pool, calls)
        de = self._dense_candidates(query, pool, calls)
        bm_pos = {c.chunk_id: (i + 1, s) for i, (c, s) in enumerate(bm)}
        de_pos = {c.chunk_id: (i + 1, s) for i, (c, s) in enumerate(de)}
        chunks = {c.chunk_id: c for c, _ in bm + de}
        hits = []
        calls.append("rrf")
        for cid, chunk in chunks.items():
            rrf = (1.0 / (self.manifest.rrf_k + bm_pos[cid][0]) if cid in bm_pos else 0.0) + (1.0 / (self.manifest.rrf_k + de_pos[cid][0]) if cid in de_pos else 0.0)
            hits.append(RetrievalHit(chunk, rrf, bm_pos.get(cid, (0, 0.0))[1], de_pos.get(cid, (0, 0.0))[1], rrf))
        hits = sorted(hits, key=lambda h: (-h.rrf_score, h.chunk.chunk_id))[: self.manifest.candidate_k]
        if self._mode_rerank:
            calls.append("reranker")
            return self._reranker.rerank(query, hits)
        return hits

    _mode_rerank = False

    def _old_qualified(self, query: str, as_of: date, mode: str, calls: list[str]) -> bool:
        """Use only the selected candidate provider to classify stale evidence."""
        old = [c for c in self.manifest.chunks if not c.active_at(as_of)]
        if not old or mode not in {"bm25", "dense", "hybrid_no_reranker", "hybrid_reranker", "remove_abstention", "mutation"}:
            return False
        if mode == "bm25":
            return bool(self._bm25_candidates(query, old, calls))
        if mode == "dense":
            return bool(self._dense_candidates(query, old, calls))
        return bool(self._hybrid_candidates(query, old, calls))

    def retrieve(self, query: str, *, as_of: date, mode: str = "hybrid_reranker", top_k: int = 5, seed: int | None = None) -> RetrievalResult:
        if mode not in R35_MODES:
            return RetrievalResult("FAILED", query, mode, failure_code="UNKNOWN_R35_MODE", trace={"mode": mode, "query": query})
        calls: list[str] = []
        if top_k != self.manifest.top_k:
            return RetrievalResult("FAILED", query, mode, failure_code="TOP_K_MUST_BE_5", trace=self._trace(query, mode, as_of, calls))
        trace = self._trace(query, mode, as_of, calls)
        if mode == "no_retrieval":
            return RetrievalResult("NO_HITS", query, mode, trace=trace)
        if mode == "oracle":
            return RetrievalResult("FAILED", query, mode, failure_code="ORACLE_REQUIRES_GOLD", trace=trace)

        time_filter = bool(self.manifest.component_matrix[mode]["metadata_time_filter"])
        pool = self._metadata_active(as_of, time_filter)
        if mode == "random":
            calls.append("seeded_random")
            rng = random.Random(self.seed if seed is None else seed)
            pool = list(pool)
            rng.shuffle(pool)
            hits = [RetrievalHit(c, 1.0) for c in pool[: self.manifest.top_k]]
        elif mode == "bm25":
            hits = [RetrievalHit(c, s, bm25_score=s) for c, s in self._bm25_candidates(query, pool, calls)]
        elif mode == "dense":
            try:
                hits = [RetrievalHit(c, s, dense_score=s) for c, s in self._dense_candidates(query, pool, calls)]
            except DenseModelUnavailable as exc:
                trace["components_called"] = calls
                trace["error"] = str(exc)
                return RetrievalResult("FAILED", query, mode, failure_code="DENSE_MODEL_UNAVAILABLE", trace=trace)
        else:
            self._mode_rerank = mode in {"hybrid_reranker", "remove_version_filter", "remove_abstention", "mutation"}
            try:
                hits = self._hybrid_candidates(query, pool, calls)
            except DenseModelUnavailable as exc:
                trace["components_called"] = calls
                trace["error"] = str(exc)
                return RetrievalResult("FAILED", query, mode, failure_code="DENSE_MODEL_UNAVAILABLE", trace=trace)

        hits = list(hits[: self.manifest.top_k])
        trace["scores"] = [{"chunk_id": h.chunk.chunk_id, "score": h.score, "bm25": h.bm25_score, "dense": h.dense_score, "rrf": h.rrf_score} for h in hits]
        evidence = self._evidence(hits)
        trace["evidence_ids"] = [e.evidence_id for e in evidence]

        # The random baseline is deliberately a retrieval-only diagnostic.  It
        # must expose random mistakes to the metrics and never turn them into a
        # safety abstention through conflict, weak-score, or stale checks.
        if mode == "random":
            trace["components_called"] = calls
            trace["conflict_set"] = []
            trace["abstention_applied"] = False
            claims = [Claim(f"claim-{hits[0].chunk.chunk_id}", hits[0].chunk.text, (evidence[0].evidence_id,))] if hits else []
            return RetrievalResult("ANSWERED" if hits else "NO_HITS", query, mode, hits=hits, evidence=evidence, claims=claims, trace=trace)

        if not hits:
            if mode == "remove_abstention":
                trace["abstention_applied"] = False
                return RetrievalResult("ANSWERED", query, mode, evidence=evidence, trace=trace)
            trace["components_called"] = calls
            try:
                stale = self._old_qualified(query, as_of, mode, calls) if time_filter else False
            except DenseModelUnavailable as exc:
                trace["error"] = str(exc)
                return RetrievalResult("FAILED", query, mode, failure_code="DENSE_MODEL_UNAVAILABLE", trace=trace)
            status = "STALE_ONLY" if stale else "NO_HITS"
            trace["components_called"] = calls
            return RetrievalResult(status, query, mode, evidence=evidence, trace=trace)

        groups: dict[str, list[RetrievalHit]] = {}
        for hit in hits:
            groups.setdefault(hit.chunk.claim_key, []).append(hit)
        conflicts = [key for key, values in groups.items() if len({(v.chunk.source_id, v.chunk.version) for v in values}) > 1 and len({v.chunk.text_hash for v in values}) > 1]
        if mode == "remove_abstention":
            status = "ANSWERED"
        elif conflicts:
            status = "CONFLICT"
        elif time_filter and not any(h.chunk.active_at(as_of) for h in hits) and self._old_qualified(query, as_of, mode, calls):
            status = "STALE_ONLY"
        elif max(h.score for h in hits) < self.manifest.evidence_threshold:
            status = "WEAK_EVIDENCE"
        else:
            status = "ANSWERED"
        claims = [Claim(f"claim-{hits[0].chunk.chunk_id}", hits[0].chunk.text, (evidence[0].evidence_id,))] if status == "ANSWERED" and hits else []
        trace["components_called"] = calls
        trace["conflict_set"] = conflicts
        trace["abstention_applied"] = bool(self.manifest.component_matrix[mode]["abstention"] is True)
        return RetrievalResult(status, query, mode, hits=hits, evidence=evidence, claims=claims, conflict_set=conflicts, trace=trace)


def remap_gold_ids(gold_ids: Sequence[str], mapping: Mapping[str, str]) -> list[str]:
    return [mapping.get(cid, cid) for cid in gold_ids]
