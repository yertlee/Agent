"""R3 shared runtime entry point used by the evaluator and tests."""

from .r3_rag import DenseIndex, DenseModelUnavailable, GenericCoverageReranker, R3Retriever, load_manifest

__all__ = ["DenseIndex", "DenseModelUnavailable", "GenericCoverageReranker", "R3Retriever", "load_manifest"]
