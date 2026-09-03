"""Evaluation package (local + deterministic M4 evaluator boundary)."""

from .rag_manifest import (KBChunk, KBManifest, KBSource, HybridRetriever, RetrievalHit,
                           RetrievalResult, build_kb_manifest, hybrid_retrieve, load_kb_manifest, normalize_text, rrf_score, write_kb_manifest)
from .datasets import (DatasetCase, DatasetManifest, case_runtime_mapping, load_dataset_manifest, make_manifest,
                       make_safety10_manifest, validate_dataset_manifest, validate_dev72,
                       validate_safety10, validate_test20_metadata, validate_no_leakage, validate_split_manifests)
from .evaluator import (CaseStatus, EvaluatorRegistry, MetricResult, default_registry,
                        evaluate_cases, paired_bootstrap, wilson_interval)
from .comparator import (BaselineComparator, ComparableRun, ComparisonResult, compare_runs)
from .report import ReportGenerator

__all__ = ["KBChunk", "KBManifest", "KBSource", "HybridRetriever", "RetrievalHit", "RetrievalResult",
           "build_kb_manifest", "hybrid_retrieve", "load_kb_manifest", "normalize_text", "rrf_score", "write_kb_manifest", "DatasetCase", "DatasetManifest",
           "case_runtime_mapping", "load_dataset_manifest", "make_manifest", "make_safety10_manifest", "validate_dataset_manifest",
           "validate_dev72", "validate_safety10", "validate_test20_metadata", "validate_no_leakage", "validate_split_manifests", "CaseStatus",
           "EvaluatorRegistry", "MetricResult", "default_registry", "evaluate_cases", "paired_bootstrap",
           "wilson_interval", "BaselineComparator", "ComparableRun", "ComparisonResult", "compare_runs",
           "ReportGenerator"]
