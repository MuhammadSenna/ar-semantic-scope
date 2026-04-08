"""
Pure-Python statistics helpers.  No Streamlit imports here — rendering
lives in ui/stats_page.py.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from data.loader import Record


@dataclass
class DatasetStats:
    total: int
    avg_question_words: float
    avg_answer_words: float
    source_counts: dict[str, int]


def compute_stats(records: list[Record]) -> DatasetStats:
    q_lens = [len(r.question.split()) for r in records]
    a_lens = [len(r.answer.split()) for r in records]
    sources = Counter(r.source for r in records)
    return DatasetStats(
        total=len(records),
        avg_question_words=float(np.mean(q_lens)),
        avg_answer_words=float(np.mean(a_lens)),
        source_counts=dict(sources.most_common()),
    )
