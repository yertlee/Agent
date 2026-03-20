from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_core.prompts import ChatPromptTemplate

from .llm import build_chat_model
from .prompts import RETRIEVAL_REWRITE_SYSTEM_PROMPT, build_rewrite_user_prompt
from .state import RetrievalEvidence
from .langsmith_utils import traceable


try:
    import faiss  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("未安装 faiss-cpu，无法构建规则检索索引。") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("未安装 sentence-transformers，无法执行规则检索。") from e


KB_DIR_NAME = os.path.join("docs", "kb")
MAX_CHUNK_CHARS = 500
MIN_CHUNK_CHARS = 50


@dataclass
class KBChunk:
    text: str
    source: str
    title: str
    chunk_id: str


_INDEX: Optional[faiss.IndexFlatIP] = None
_EMB_DIM: Optional[int] = None
_MODEL: Optional[SentenceTransformer] = None
_CHUNKS: List[KBChunk] = []


def _get_project_root() -> str:
    # project_root/.../agent/rag_retriever.py -> project_root
    return str(Path(__file__).resolve().parents[1])


def _load_model() -> SentenceTransformer:
    global _MODEL, _EMB_DIM
    if _MODEL is not None:
        return _MODEL

    model_name = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    model = SentenceTransformer(model_name)
    _MODEL = model
    test_vec = model.encode(["规则检索测试"], convert_to_numpy=True)
    _EMB_DIM = int(test_vec.shape[1])
    return model


def _iter_markdown_files(kb_dir: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(kb_dir):
        if name.lower().endswith(".md"):
            files.append(os.path.join(kb_dir, name))
    return sorted(files)


def _split_markdown_to_chunks(path: str) -> List[KBChunk]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(path)
    lines = content.splitlines()
    chunks: List[KBChunk] = []
    current_title = ""
    paragraph_lines: List[str] = []
    chunk_idx = 0

    def flush(lines_buffer: List[str], title: str, start_idx: int) -> int:
        local_idx = start_idx
        paragraph = "\n".join(ln.strip() for ln in lines_buffer).strip()
        if not paragraph:
            return local_idx

        buffer: List[str] = []
        buffer_len = 0

        def emit(buf: List[str], idx: int) -> int:
            if not buf:
                return idx
            body = "\n".join(buf).strip()
            if not body:
                return idx
            title_label = title or "通用规则"
            chunks.append(
                KBChunk(
                    text=f"标题：{title_label}\n内容：{body}",
                    source=filename,
                    title=title_label,
                    chunk_id=f"{filename}#{idx}",
                )
            )
            return idx + 1

        for line in paragraph.splitlines():
            line = line.strip()
            if not line:
                continue
            if buffer and buffer_len + len(line) + 1 > MAX_CHUNK_CHARS:
                local_idx = emit(buffer, local_idx)
                buffer = []
                buffer_len = 0
            buffer.append(line)
            buffer_len += len(line) + 1

        if buffer_len >= MIN_CHUNK_CHARS or not chunks:
            local_idx = emit(buffer, local_idx)
        return local_idx

    for raw in lines:
        line = raw.rstrip("\n")
        header_match = re.match(r"^\s*(#{1,6})\s+(.*)$", line)
        if header_match:
            if paragraph_lines:
                chunk_idx = flush(paragraph_lines, current_title, chunk_idx)
                paragraph_lines = []
            current_title = header_match.group(2).strip() or current_title
            continue
        if not line.strip():
            if paragraph_lines:
                chunk_idx = flush(paragraph_lines, current_title, chunk_idx)
                paragraph_lines = []
            continue
        paragraph_lines.append(line)

    if paragraph_lines:
        flush(paragraph_lines, current_title, chunk_idx)
    return chunks


def build_index(force_rebuild: bool = False) -> None:
    global _INDEX, _CHUNKS, _EMB_DIM
    if _INDEX is not None and _CHUNKS and not force_rebuild:
        return

    kb_dir = os.path.join(_get_project_root(), KB_DIR_NAME)
    if not os.path.isdir(kb_dir):
        raise FileNotFoundError(f"未找到知识库目录：{kb_dir}")

    md_files = _iter_markdown_files(kb_dir)
    if not md_files:
        raise FileNotFoundError(f"知识库目录 {kb_dir} 下没有 markdown 文件。")

    all_chunks: List[KBChunk] = []
    for path in md_files:
        all_chunks.extend(_split_markdown_to_chunks(path))
    if not all_chunks:
        raise RuntimeError("知识库切块结果为空。")

    model = _load_model()
    emb = model.encode([chunk.text for chunk in all_chunks], convert_to_numpy=True, show_progress_bar=False)
    emb = emb.astype("float32")
    faiss.normalize_L2(emb)

    dim = emb.shape[1]
    _EMB_DIM = dim
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    _INDEX = index
    _CHUNKS = all_chunks


def _ensure_index() -> Tuple[faiss.IndexFlatIP, List[KBChunk]]:
    if _INDEX is None or not _CHUNKS:
        build_index(force_rebuild=False)
    assert _INDEX is not None
    return _INDEX, _CHUNKS


def _extract_query_terms(query: str) -> List[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", query or "").strip().lower()
    if not text:
        return []

    parts = [p for p in text.split() if p]
    terms = set(parts)
    for part in parts:
        if len(part) >= 4:
            for idx in range(0, len(part) - 1):
                terms.add(part[idx : idx + 2])
    return [term for term in terms if len(term) >= 2]


def _lexical_overlap_ratio(query: str, text: str) -> float:
    terms = _extract_query_terms(query)
    if not terms:
        return 0.0
    normalized = (text or "").lower()
    hits = sum(1 for term in terms if term in normalized)
    return hits / max(len(terms), 1)


def _dedupe_hits(raw_hits: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    deduped: List[Dict[str, object]] = []
    for hit in raw_hits:
        key = (hit.get("chunk_id"), hit.get("source"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
    return deduped


def _summarize_text(text: str, max_chars: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "…"


@traceable(name="agent_v3_policy_query_rewrite")
def rewrite_query(original_query: str, user_input: str = "") -> str:
    original = (original_query or "").strip()
    if not original:
        return original

    llm = build_chat_model(temperature=0.0, max_tokens=80, tags=["query-rewrite"])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RETRIEVAL_REWRITE_SYSTEM_PROMPT),
            ("human", "{user_prompt}"),
        ]
    )
    chain = prompt | llm
    try:
        result = chain.invoke({"user_prompt": build_rewrite_user_prompt(original, user_input)})
        rewritten = (result.content or "").strip()
        return rewritten or original
    except Exception:
        return original


@traceable(name="agent_v3_policy_retrieve")
def retrieve_policy_evidence(
    query: str,
    *,
    top_k: int = 4,
    rewritten_from: str = "",
    score_threshold: Optional[float] = None,
    lexical_threshold: Optional[float] = None,
) -> List[RetrievalEvidence]:
    query = (query or "").strip()
    if not query:
        return []

    index, chunks = _ensure_index()
    model = _load_model()

    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(q_emb)

    k = min(max(top_k, 1), len(chunks))
    scores, idxs = index.search(q_emb, k)
    score_threshold = score_threshold if score_threshold is not None else float(os.getenv("RAG_SCORE_THRESHOLD", "0.2"))
    lexical_threshold = lexical_threshold if lexical_threshold is not None else float(os.getenv("RAG_LEXICAL_THRESHOLD", "0.05"))

    raw_hits: List[Dict[str, object]] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0 or float(score) < score_threshold:
            continue
        chunk = chunks[int(idx)]
        lexical_ratio = _lexical_overlap_ratio(query, chunk.text)
        if lexical_ratio < lexical_threshold:
            continue
        raw_hits.append(
            {
                "query_used": query,
                "rewritten_from": rewritten_from,
                "score": float(score),
                "source": chunk.source,
                "title": chunk.title,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "evidence_summary": _summarize_text(chunk.text),
            }
        )

    deduped = _dedupe_hits(raw_hits)
    return [RetrievalEvidence(**hit) for hit in deduped]


def retrieve(query: str, top_k: int = 3) -> List[Dict[str, object]]:
    evidence = retrieve_policy_evidence(query, top_k=top_k)
    return [item.model_dump() for item in evidence]


def build_citation_text(evidence: List[RetrievalEvidence], max_items: int = 2) -> str:
    blocks: List[str] = []
    for idx, item in enumerate(evidence[:max_items], start=1):
        blocks.append(f"[参考{idx}] 《{item.title}》：{item.evidence_summary}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    build_index(force_rebuild=False)
    while True:
        user_query = input("问题（回车退出）: ").strip()
        if not user_query:
            break
        hits = retrieve_policy_evidence(user_query, top_k=4)
        print(f"\n命中 {len(hits)} 条：")
        for idx, hit in enumerate(hits, start=1):
            print(f"\n[{idx}] {hit.source} - {hit.title} score={hit.score:.3f}")
            print(hit.text[:400] + ("..." if len(hit.text) > 400 else ""))
        print("\n" + "-" * 60 + "\n")
