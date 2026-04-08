"""
CSS injection and text formatting utilities.
Keeping styles here avoids scattering raw HTML strings across UI modules.
"""

import streamlit as st

_CSS = """
<style>
.rtl {
    direction: rtl;
    text-align: right;
    font-family: 'Tahoma', 'Arial Unicode MS', sans-serif;
    font-size: 16px;
    line-height: 1.7;
    padding: 10px 14px;
    border-radius: 6px;
}
.q-card {
    border-left: 4px solid #1f77b4;
    background-color: var(--background-color, #41a5f0);
    padding: 14px;
    margin: 8px 0;
    border-radius: 6px;
}
.a-card {
    border-left: 4px solid #2ca02c;
    background-color: var(--background-color, #45cc45);
    padding: 14px;
    margin: 8px 0;
    border-radius: 6px;
}
.sim-badge {
    display: inline-block;
    background-color: #c7a330;
    color: #fff;
    padding: 4px 12px;
    border-radius: 14px;
    font-weight: 600;
    font-size: 14px;
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def rtl_block(text: str, extra_class: str = "") -> str:
    """Wrap `text` in a right-to-left styled div."""
    return f'<div class="rtl {extra_class}">{text}</div>'


def similarity_badge(pct: float) -> str:
    return f'<span class="sim-badge">{pct:.1f}%</span>'
