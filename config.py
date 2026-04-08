"""
Central configuration for the Arabic QnA Semantic Search app.
Edit here to change model, dataset, or UI defaults.
"""

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET_ID = "sadeem-ai/arabic-qna"
DATASET_SPLIT = "train"
FILTER_HAS_ANSWER = True

# ── Search ─────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 10

# ── UI ─────────────────────────────────────────────────────────────────────────
PAGE_TITLE = "Arabic QnA Semantic Search"
PAGE_ICON = "🔍"

EXAMPLE_QUERIES = [
    "ما السبب في صغر الأسنان",
    "كيف أتعلم البرمجة",
    "ما هي فوائد الرياضة",
    "كيف أحسن من صحتي",
]
