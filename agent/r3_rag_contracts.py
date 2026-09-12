"""Frozen R3 contract surface.

The implementation lives in :mod:`agent.r3_rag`; these explicit exports make
the EvidenceRef/Claim/version/status contract importable without exposing the
legacy runtime retriever.
"""

from .r3_rag import Claim, EvidenceRef, Manifest, RetrievalHit, RetrievalResult, SourceRecord, ChunkRecord

__all__ = ["Claim", "EvidenceRef", "Manifest", "RetrievalHit", "RetrievalResult", "SourceRecord", "ChunkRecord"]
