import pickle
import os
import re
from functools import lru_cache


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "router_model.pkl")
_classifier_pipeline = None


def _load_classifier():
    global _classifier_pipeline
    if _classifier_pipeline is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"router_model.pkl not found at {_MODEL_PATH}. "
                f"Run: python train_classifier.py"
            )
        with open(_MODEL_PATH, "rb") as f:
            _classifier_pipeline = pickle.load(f)
    return _classifier_pipeline


def classify_category(prompt: str) -> str:
    """
    Classifies a task prompt into one of the 8 Track 1 capability
    categories using the trained sklearn pipeline (TF-IDF + Logistic
    Regression). This is NOT an LLM -- it produces zero tokens and runs in
    under a millisecond, so it costs nothing against the token-efficiency
    score and doesn't touch the "all inference must go through Fireworks"
    rule at all (it never generates text, only classifies).

    Parameters:
        prompt: the task's prompt text.

    Returns:
        One of: "factual_knowledge", "mathematical_reasoning",
        "sentiment_classification", "text_summarization",
        "named_entity_recognition", "code_debugging", "logical_reasoning",
        "code_generation".
    """
    clf = _load_classifier()
    proba = clf.predict_proba([prompt])[0]
    best_idx = proba.argmax()
    best_prob = proba[best_idx]
    predicted = clf.classes_[best_idx]
    
    # If confidence is low, fall back to factual_knowledge as the safe default
    if best_prob < 0.60:
        return "factual_knowledge"
    return predicted


# Categories where the task explicitly constrains output length/format
# (e.g. "summarize in one sentence", "list 3 bullet points"). Used by
# remote_client.py to decide whether the conciseness system prompt should
# defer to the task's own instructions.
LENGTH_CONSTRAINED_CATEGORIES = {"text_summarization"}


def category_has_length_constraint(category: str) -> bool:
    return category in LENGTH_CONSTRAINED_CATEGORIES


# Detects when the prompt itself explicitly requests a long/detailed
# answer, independent of category -- e.g. a factual_knowledge task that
# says "explain in detail" should not have its answer artificially
# shortened by a conciseness system prompt.
LENGTH_OVERRIDE_REGEX = re.compile(
    r"\b(detailed|comprehensive|exhaustive|thorough|extensive|elaborate|"
    r"in.depth|step.by.step|long.form|full.explanation|"
    r"\d+[\s-]*(word|words)\b)",
    re.IGNORECASE
)


def wants_detailed_response(prompt: str) -> bool:
    """Returns True if the prompt explicitly requests a long/detailed answer."""
    return bool(LENGTH_OVERRIDE_REGEX.search(prompt))
