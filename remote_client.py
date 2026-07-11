"""
Fireworks AI client for Track 1.

CRITICAL: all three of these come from the harness at runtime, never
hardcoded and never defaulted -- using os.environ[...] (not os.getenv with
a fallback) so a missing var fails loudly and immediately rather than
silently producing a broken request.

  FIREWORKS_API_KEY   - provided by the harness, not your own key
  FIREWORKS_BASE_URL   - the harness's proxy; calls that bypass this are
                         not recorded and score zero tokens
  ALLOWED_MODELS       - comma-separated list, published on launch day
                         (read via model_selector.get_allowed_models(),
                         not directly in this file)
"""

import httpx
import os
import re
from router import wants_detailed_response, category_has_length_constraint

FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
FIREWORKS_BASE_URL = os.environ["FIREWORKS_BASE_URL"]

# Lowered from the old 90.0s used in the earlier hybrid-routing design.
# The reveal document specifies a 30-second per-request ceiling; this
# leaves headroom for main.py's retry logic to still land inside that
# window if the first attempt is slow.
REQUEST_TIMEOUT_SECONDS = 20.0

# Strategy 2: Categories whose prompts are always self-contained instructions
# (e.g. "Classify as Positive/Negative/Neutral: ..."). No system message needed;
# adding one would be pure redundant token overhead.
_ZERO_PROMPT_CATEGORIES = {"sentiment_classification", "named_entity_recognition"}

# Strategy 4: Code categories need a minimal format guard so the model doesn't
# wrap its output in prose explanation. Trimmed from the old ~30-token CONCISE
# prompt to ~10 tokens -- only the semantically necessary clause is kept.
_CODE_CATEGORIES = {"code_generation", "code_debugging"}
_CODE_SYSTEM_PROMPT = "Output only code. No explanation, no preamble."

# Strategy 1: For all remaining categories, inline a 5-token instruction directly
# into the user turn rather than paying a separate system-role message tax.
_CONCISE_PREFIX = "Output only the answer. "

# Per-category output token caps.
# Goal: minimize output tokens for cheap-but-short tasks (sentiment → 1 word),
# while preserving enough budget for code generation and detailed summaries.
# The harness counts output tokens against the efficiency score, so shaving
# even 50 tokens per NLP task across a large batch is meaningful.
# Increased from strict plan values to provide a generous safety margin against truncation.
_CATEGORY_MAX_TOKENS = {
    "sentiment_classification": 50,      # Safely raised to prevent cutting off if the model rambles
    "named_entity_recognition": 150,     # Short entity list
    "factual_knowledge": 256,            # Standard budget for one/few sentences
    "mathematical_reasoning": 256,       # Numbers with potential short explanations
    "logical_reasoning": 256,            # Short answers or logic conclusions
    "text_summarization": 400,           # Generous fallback floor for summaries
    "code_debugging": 750,               # Code + brief explanation
    "code_generation": 750,              # Short to medium function body
}

_CATEGORY_TEMPERATURE = {
    "sentiment_classification": 0.0,
    "mathematical_reasoning":   0.0,
    "logical_reasoning":        0.0,
    "named_entity_recognition": 0.0,
    "factual_knowledge":        0.0,
    "text_summarization":       0.3,
    "code_generation":          0.2,
    "code_debugging":           0.2,
}

_WORD_COUNT_RE = re.compile(r'\b(\d+)\s*(?:word|words)\b', re.IGNORECASE)

def _max_tokens_for(prompt: str, category: str) -> int:
    # Word-count-constrained task: allocate 4x the requested word count to ensure no truncation
    m = _WORD_COUNT_RE.search(prompt)
    if m:
        calculated = max(100, int(m.group(1)) * 4)
        if category in {"code_generation", "code_debugging"}:
            # For code tasks, a word match (like "32-word architecture") might be a false positive,
            # so only use it to INCREASE the cap, never reduce it below the category baseline.
            return max(_CATEGORY_MAX_TOKENS.get(category, 750), calculated)
        return calculated
        
    # Genuinely detailed/open-ended task (no word count specified):
    if wants_detailed_response(prompt):
        return 800
        
    # Length-constrained by category (e.g. text_summarization without explicit word count):
    if category_has_length_constraint(category):
        return 400
        
    return _CATEGORY_MAX_TOKENS.get(category, 512)


async def call_fireworks(prompt: str, model_id: str, category: str = "") -> dict:
    """
    Calls a Fireworks model through the harness's proxy URL.

    Parameters:
        prompt: the task's prompt text.
        model_id: a model ID string that MUST already be a member of the
                   runtime ALLOWED_MODELS list -- this function does not
                   validate that itself; the caller (main.py, via
                   model_selector.select_model()) is responsible for that.
        category: the task's classified category, used only to decide
                   whether the conciseness system prompt should defer to
                   an explicit format/length constraint (e.g.
                   text_summarization tasks often specify exact length).

    Returns:
        dict with keys: text, tokens_in, tokens_out, cost
        (cost is informational only -- the harness's own judging proxy is
        the authoritative token count, not this client-side estimate)
    """
    # ── System-prompt tax reduction (Strategies 1, 2, 4) ──────────────────
    #
    # ZERO system message (Strategy 2):
    #   a) Self-describing NLP categories: the user prompt already contains
    #      the full instruction ("Classify as Positive/Negative/Neutral: ...").
    #   b) Length/format-specified tasks: the prompt itself specifies the
    #      output format, so any system instruction is redundant overhead.
    #
    # TRIMMED system message (Strategy 4):
    #   Code categories keep a minimal format guard (~10 tokens) so the model
    #   doesn't wrap code in prose. This is the only remaining system message.
    #
    # INLINE PREFIX (Strategy 1):
    #   All other categories prepend a 5-token phrase into the user turn
    #   instead of a 30-token system role message.
    if wants_detailed_response(prompt) or category_has_length_constraint(category):
        # Strategy 3 fix: detailed responses still need language guard
        messages = [{"role": "user", "content": "Answer in English. " + prompt}]
    elif category in _ZERO_PROMPT_CATEGORIES:
        # Strategy 2: zero overhead -- no system message, no prefix.
        messages = [{"role": "user", "content": prompt}]
    elif category in _CODE_CATEGORIES:
        # Strategy 4: lean system message only where format genuinely matters.
        messages = [
            {"role": "system", "content": _CODE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
    else:
        # Strategy 1: 5-token inline prefix replaces the old 30-token system role.
        messages = [{"role": "user", "content": _CONCISE_PREFIX + prompt}]

    max_tokens = _max_tokens_for(prompt, category)
    
    # If Kimi is used (either as primary for code, or as a fallback for NLP), 
    # it needs enough token budget for its internal reasoning chain.
    # Enforce a generous floor so it doesn't inherit a tiny NLP micro-cap and truncate.
    if "kimi" in model_id.lower():
        max_tokens = max(max_tokens, 750)

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": _CATEGORY_TEMPERATURE.get(category, 0.1),
    }

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{FIREWORKS_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )

        response.raise_for_status()
        data = response.json()

    message = data["choices"][0].get("message", {})
    text = (message.get("content") or "").strip()
    
    usage = data.get("usage", {})
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)

    return {
        "text": text,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model_used": model_id,
    }
