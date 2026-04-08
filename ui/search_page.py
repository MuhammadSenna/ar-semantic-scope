"""
Search page: query input, example buttons, result rendering.
"""

import streamlit as st

from config import EXAMPLE_QUERIES, DEFAULT_TOP_K
from data.loader import SearchIndex
from search.engine import search
from ui.components import result_card, example_query_buttons


def render(model, index: SearchIndex, top_k: int = DEFAULT_TOP_K) -> None:
    st.subheader("البحث — Search")

    # ── Example queries ────────────────────────────────────────────────────────
    st.caption("أمثلة — try an example:")
    clicked = example_query_buttons(EXAMPLE_QUERIES)
    if clicked:
        st.session_state["query"] = clicked

    # ── Search input ───────────────────────────────────────────────────────────
    query: str = st.text_input(
        label="اكتب سؤالك — enter your question:",
        value=st.session_state.get("query", ""),
        placeholder="مثال: ما السبب في صغر الأسنان؟",
        key="query_input",
    )

    # Sync session state so example buttons can pre-fill the box
    if query:
        st.session_state["query"] = query

    # ── Search trigger: explicit button only ───────────────────────────────────
    if not st.button("بحث — Search", type="primary"):
        return

    query = query.strip()
    if not query:
        st.warning("يرجى إدخال سؤال للبحث — please enter a question.")
        return

    with st.spinner("جاري البحث…"):
        results = search(model, index, query, top_k)

    if not results:
        st.warning("لم يتم العثور على نتائج — no results found.")
        return

    st.success(f"تم العثور على {len(results)} نتيجة — {len(results)} result(s) found")
    st.markdown("---")
    for result in results:
        result_card(result)
