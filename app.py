"""
app.py — Streamlit entry point.

"""

import streamlit as st

import config
from data.loader import load_records, build_search_index
from models.embedder import load_embedder
from ui import search_page, stats_page
from utils.formatting import inject_css

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── App header ─────────────────────────────────────────────────────────────────
st.title("🔍 البحث الدلالي في الأسئلة والأجوبة العربية")
st.title("Arabic QnA Semantic Search")
st.caption(
    "Uses multilingual sentence embeddings + FAISS to find semantically similar Q&A pairs."
)

# ── Load resources ─────────────────────────────────────────────────────────────
with st.spinner("Loading model and dataset — this only happens once…"):
    try:
        model = load_embedder()
        records = load_records()
        index = build_search_index(model, records)
    except Exception as exc:
        st.error(f"Failed to load resources: {exc}")
        st.info(
            "Check your internet connection and available disk space, "
            "then reload the page."
        )
        st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider(
        "Number of results",
        min_value=config.MIN_TOP_K,
        max_value=config.MAX_TOP_K,
        value=config.DEFAULT_TOP_K,
    )
    st.divider()
    show_stats = st.checkbox("Show dataset statistics")

# ── Main content ───────────────────────────────────────────────────────────────
if show_stats:
    stats_page.render(records)
    st.divider()

search_page.render(model, index, top_k=top_k)
