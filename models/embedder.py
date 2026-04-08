

import streamlit as st
from sentence_transformers import SentenceTransformer
from config import MODEL_ID


@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    """Load and cache the multilingual sentence encoder."""
    return SentenceTransformer(MODEL_ID)


def embed(model: SentenceTransformer, texts: list[str]) -> "np.ndarray":
    """Encode a list of texts; returns a float32 array (N, dim)."""
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def embed_query(model: SentenceTransformer, query: str) -> "np.ndarray":
    """Encode a single query string; returns shape (1, dim)."""
    import numpy as np
    vec = model.encode(query, convert_to_numpy=True)
    return vec.reshape(1, -1).astype(np.float32)
