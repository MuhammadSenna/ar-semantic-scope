

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import faiss
import numpy as np
import streamlit as st
from datasets import load_dataset

from config import DATASET_ID, DATASET_SPLIT, FILTER_HAS_ANSWER
from models.embedder import embed


@dataclass(frozen=True, slots=True)
class Record:
    text: str
    question: str
    answer: str
    source: str
    title: str


@dataclass
class SearchIndex:
    faiss_index: faiss.IndexIDMap
    records: list[Record]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_records(hf_split) -> list[Record]:
    return [
        Record(
            text=row["text"],
            question=row["question"],
            answer=row["answer"],
            source=row["source"],
            title=row["title"],
        )
        for row in hf_split
    ]


def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexIDMap:
    norm = deepcopy(embeddings).astype(np.float32)
    faiss.normalize_L2(norm)
    ids = np.arange(len(norm), dtype=np.int64)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(norm.shape[1]))
    index.add_with_ids(norm, ids)
    return index


# ── Public API (cached) ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_records() -> list[Record]:
    """Download (or load from cache) and return filtered records."""
    ds = load_dataset(DATASET_ID)
    if FILTER_HAS_ANSWER:
        ds = ds.filter(lambda x: x["has_answer"] is True)
    return _build_records(ds[DATASET_SPLIT])


@st.cache_data(show_spinner=False)
def build_search_index(_model, records: list[Record]) -> SearchIndex:
    """
    Create embeddings for all records and build a FAISS index.
    `_model` has a leading underscore so Streamlit skips hashing it
    (SentenceTransformer is not hashable).
    """
    texts = [r.text for r in records]
    embeddings = embed(_model, texts)
    return SearchIndex(
        faiss_index=_build_faiss_index(embeddings),
        records=records,
    )
