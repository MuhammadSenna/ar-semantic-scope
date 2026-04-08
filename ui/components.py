"""
Reusable UI components.  Each function renders one logical piece of the UI
and has no side-effects beyond the Streamlit elements it creates.
"""

import streamlit as st

from search.engine import SearchResult
from utils.formatting import rtl_block, similarity_badge


def result_card(result: SearchResult) -> None:
    """Render a single search result as a two-column card."""
    col_text, col_score = st.columns([4, 1])

    with col_text:
        st.markdown(
            f"""
            <div class="q-card">
                <strong>السؤال — Question #{result.rank}</strong>
                {rtl_block(result.question)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="a-card">
                <strong>الإجابة — Answer</strong>
                {rtl_block(result.answer)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("المصدر — Source & details"):
            st.write(f"**Source:** {result.source}")
            st.write(f"**Title:** {result.title}")

    with col_score:
        st.markdown(
            f"<div style='margin-top:20px;text-align:center'>"
            f"{similarity_badge(result.similarity_pct)}"
            f"<br><small>similarity</small></div>",
            unsafe_allow_html=True,
        )

    st.divider()


def example_query_buttons(examples: list[str]) -> str | None:
    """
    Render one button per example query.
    Returns the clicked example text, or None if nothing was clicked.
    """
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True):
            return example
    return None
