"""
Search engine: encodes a query and retrieves the top-k most similar records.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from data.loader import Record, SearchIndex
from models.embedder import embed_query


@dataclass(frozen=True, slots=True)
class SearchResult:
    rank: int
    similarity: float          # cosine similarity in [0, 1]
    similarity_pct: float      # pre-computed percentage
    question: str
    answer: str
    source: str
    title: str


def search(
    model: SentenceTransformer,
    index: SearchIndex,
    query: str,
    top_k: int,
) -> list[SearchResult]:
    """Return up to `top_k` results for `query`, sorted by similarity."""
    vec = embed_query(model, query).astype(np.float32)
    faiss.normalize_L2(vec)

    similarities, indices = index.faiss_index.search(vec, top_k)

    results: list[SearchResult] = []
    for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), start=1):
        if idx == -1:
            continue
        rec: Record = index.records[idx]
        results.append(
            SearchResult(
                rank=rank,
                similarity=float(sim),
                similarity_pct=float(sim) * 100,
                question=rec.question,
                answer=rec.answer,
                source=rec.source,
                title=rec.title,
            )
        )
    return results
