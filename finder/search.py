"""Two-stage search: CLIP + FAISS retrieval, then Claude Vision reranking."""
from __future__ import annotations

import time

from .embeddings import embed_text
from .index import PhotoIndex
from .rerank import rerank


def search(index: PhotoIndex, query: str, clip_k: int = 10, final_k: int = 3, use_rerank: bool = True) -> dict:
    t0 = time.perf_counter()
    candidates = index.search(embed_text(query), k=clip_k)
    stage1_ms = (time.perf_counter() - t0) * 1000

    if not use_rerank:
        return {"candidates": candidates, "results": candidates[:final_k], "stage1_ms": stage1_ms, "stage2_s": 0.0}

    t1 = time.perf_counter()
    results = rerank(query, candidates, final_k=final_k)
    return {"candidates": candidates, "results": results, "stage1_ms": stage1_ms,
            "stage2_s": time.perf_counter() - t1}
