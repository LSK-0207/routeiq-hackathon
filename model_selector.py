"""
Maps a predicted task category to a specific Fireworks model ID.

ROUTING STRATEGY (updated after cost + benchmark research):

  Model pricing on Fireworks (per 1M tokens):
    minimax-m3:      $0.30 input / $1.20 output   -- cheap general-purpose
    kimi-k2p7-code:  $0.95 input / $4.00 output   -- expensive, code-specialist

  Benchmark summary:
    - kimi-k2p7-code scores higher on coding (62.0 Kimi Code Bench v2 vs ~59
      SWE-Bench Pro for minimax-m3) and deep reasoning (HLE, GPQA-Diamond)
    - minimax-m3 is accurate enough for NLP/factual tasks at 3x lower cost

  Decision rules applied:
    CODE tasks (debugging, generation): kimi-k2p7-code first -- its benchmark
      advantage is worth the cost premium; a wrong code answer is much less
      salvageable than a wrong NLP answer.
    REASONING tasks (math, logic): kimi-k2p7-code first -- it has a measurable
      edge on deep reasoning benchmarks and thinking-token efficiency.
    NLP tasks (factual, sentiment, NER, summarization): minimax-m3 first --
      plenty capable, and at 3x lower cost it's the clear choice.
    LOCAL model: retained for factual, sentiment, NER -- the local Llama 3.2
      3B handles these reliably and costs zero tokens.

ALLOWED_MODELS is read fresh from the environment at call time (never
hardcoded), per the harness rules -- this file only defines *preferences*
among whatever the harness actually provides. If a preferred model isn't in
the runtime ALLOWED_MODELS list, select_model() falls back gracefully to the
first available model rather than crashing.

Gemma variants (gemma-4-*) are placed LAST in every category -- Gemma is
an "on-demand" model on Fireworks requiring manual deployment before it's
callable. Its availability in the grading environment is unknown. It is kept
in the list purely as a bonus fallback in case it happens to be deployed.
"""

import os

# Categories where a small bundled local model (2-3B, 4-bit quantized) can
# reliably get the right answer at zero token cost. Deliberately conservative:
# wrong local answers cost an accuracy slot just like wrong paid answers.
# Math, logic, summarization, and code stay Fireworks-only.
LOCAL_ELIGIBLE_CATEGORIES = {
    "factual_knowledge",
    "sentiment_classification",
    "named_entity_recognition",
    "text_summarization",
}

# Per-category ordered preference list.
# First entry = primary model. Remaining entries = retry candidates.
CATEGORY_MODEL_PREFERENCE = {
    # NLP categories: minimax-m3 first (3x cheaper, accurate enough for NLP)
    "factual_knowledge": [
        "minimax-m3", "gemma-4-31b-it-nvfp4",
    ],
    "sentiment_classification": [
        "minimax-m3", "gemma-4-31b-it-nvfp4",
    ],
    "named_entity_recognition": [
        "minimax-m3", "gemma-4-31b-it-nvfp4",
    ],
    "text_summarization": [
        "minimax-m3", "gemma-4-31b-it-nvfp4",
    ],
    # Reasoning categories: minimax-m3 first to reduce output tokens from
    # internal reasoning chains.
    "mathematical_reasoning": [
        "minimax-m3", "gemma-4-31b-it-nvfp4",
    ],
    "logical_reasoning": [
        "minimax-m3", "gemma-4-31b-it-nvfp4",
    ],
    # Code categories: minimax-m3 first to save tokens, Kimi as powerful fallback
    "code_debugging": [
        "minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it-nvfp4",
    ],
    "code_generation": [
        "minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it-nvfp4",
    ],
}

# Absolute last-resort fallback if a category is missing from the table above.
DEFAULT_PREFERENCE = ["minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it-nvfp4"]


def get_allowed_models() -> list[str]:
    """
    Reads ALLOWED_MODELS fresh from the environment every call -- never
    cached, never hardcoded, per the harness rule that exact model IDs are
    published on launch day and must be read at runtime.
    """
    raw = os.environ["ALLOWED_MODELS"]
    return [m.strip() for m in raw.split(",") if m.strip()]


def select_model(category: str, exclude: list[str] | None = None) -> str:
    """
    Returns the best available model ID for a given category.

    Parameters:
        category: one of the 8 category labels from router.py's classifier.
        exclude: model IDs to skip (used for retry -- e.g. if the first
                 choice's output already failed validation once).

    Returns:
        A model ID string guaranteed to be present in the runtime
        ALLOWED_MODELS list.

    Raises:
        RuntimeError if no model in the preference chain (or the fallback
        default) is actually present in ALLOWED_MODELS -- this should only
        happen if the harness publishes a completely different model set
        than expected, in which case failing loudly is better than silently
        calling an unauthorized model (which invalidates the submission).

    NOTE on matching: real Fireworks model IDs are typically full paths
    like "accounts/fireworks/models/minimax-m3", not the bare
    "minimax-m3" name alone. This function matches on SUFFIX, not exact
    equality, so the bare names in CATEGORY_MODEL_PREFERENCE correctly
    match full-path IDs from ALLOWED_MODELS regardless of which form the
    harness actually publishes -- this was found and fixed after testing
    showed exact-match against a real full-path ALLOWED_MODELS value
    silently fell through to the fallback branch every time.
    """
    exclude = exclude or []
    allowed = get_allowed_models()
    preference = CATEGORY_MODEL_PREFERENCE.get(category, DEFAULT_PREFERENCE)

    def _matches(short_name: str, full_id: str) -> bool:
        return full_id == short_name or full_id.endswith("/" + short_name)

    for short_name in preference:
        for full_id in allowed:
            if _matches(short_name, full_id) and full_id not in exclude:
                return full_id

    # Preference chain exhausted -- fall back to any allowed model not yet excluded
    for full_id in allowed:
        if full_id not in exclude:
            return full_id

    raise RuntimeError(
        f"No available model for category '{category}'. "
        f"ALLOWED_MODELS={allowed}, excluded={exclude}"
    )