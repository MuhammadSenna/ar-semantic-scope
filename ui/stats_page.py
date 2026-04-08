"""
Statistics page: metrics and source distribution chart.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data.loader import Record
from data.stats import compute_stats


def render(records: list[Record]) -> None:
    st.subheader("إحصائيات البيانات — Dataset Statistics")

    stats = compute_stats(records)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total questions", stats.total)
    col2.metric("Avg question length", f"{stats.avg_question_words:.1f} words")
    col3.metric("Avg answer length", f"{stats.avg_answer_words:.1f} words")

    st.subheader("توزيع المصادر — Source Distribution")

    source_df = pd.DataFrame(
        stats.source_counts.items(), columns=["Source", "Count"]
    ).sort_values("Count", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(3, len(source_df) * 0.45)))
    ax.barh(source_df["Source"], source_df["Count"], color="#1f77b4")
    ax.set_xlabel("Count")
    ax.set_ylabel("Source")
    ax.set_title("Data sources")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
