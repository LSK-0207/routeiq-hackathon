"""
RouteIQ Track 1 -- batch entry point.

Contract (from the hackathon reveal document):
  1. Read tasks from /input/tasks.json on startup
  2. Write results to /output/results.json before exiting
  3. Exit code 0 on success, non-zero on failure
  4. Maximum runtime: 10 minutes for the entire batch
  5. Response time per task: under 30 seconds
  6. Container must be ready within 60 seconds of start
  7. All responses must be in English
  8. Only models in ALLOWED_MODELS are permitted

This file intentionally is NOT a FastAPI server -- there is no HTTP
request/response cycle being scored. It runs once, processes every task,
writes one JSON file, and exits.
"""

import asyncio
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher

from router import classify_category
from model_selector import select_model, LOCAL_ELIGIBLE_CATEGORIES
from remote_client import call_fireworks
import local_model_client

INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json")

# How many tasks to process concurrently. Bounded via a semaphore so we
# don't hammer the harness's proxy with unlimited parallel requests, while
# still finishing well inside the 10-minute total ceiling for larger
# batches.
MAX_CONCURRENT_TASKS = 5

# Detects genuinely wrong-language output (e.g. a model responding
# entirely in Chinese, Japanese, Korean, Cyrillic, Arabic, Devanagari,
# Thai, or Hebrew script) without flagging normal English/code content
# that happens to contain isolated non-ASCII characters -- math symbols
# (\u00b2, \u00b0, \u03c0), accented proper nouns (caf\u00e9), currency signs, or
# unicode identifiers in code are all legitimate and must NOT be flagged.
#
# A naive "ratio of any character above ASCII 127" check was tested and
# found to false-positive on exactly the kind of short, correct answers
# these categories organically produce -- "O(n\u00b2)" (a completely standard
# Big-O answer), "42\u00b0C", and "x\u00b2 + y\u00b2 = r\u00b2" all triggered a false
# "non-English" flag under that approach, which would have wasted a retry
# call (and its tokens) on a perfectly correct answer. This version checks
# specifically for non-Latin SCRIPT BLOCKS rather than any non-ASCII code
# point, and requires a meaningful proportion of the text to be in such a
# script before flagging -- a single isolated symbol can never trigger it.
_NON_LATIN_SCRIPT_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0x0400, 0x04FF),   # Cyrillic
    (0x0600, 0x06FF),   # Arabic
    (0x0900, 0x097F),   # Devanagari
    (0x0E00, 0x0E7F),   # Thai
    (0x0590, 0x05FF),   # Hebrew
]


def _looks_like_english(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    script_chars = sum(
        1 for ch in stripped
        if any(lo <= ord(ch) <= hi for lo, hi in _NON_LATIN_SCRIPT_RANGES)
    )
    # Require both a minimum count AND a minimum proportion, so a single
    # stray character in a long response can't trigger this, and a very
    # short response needs a real concentration of non-Latin script (not
    # just one symbol) to be flagged.
    if script_chars < 3:
        return True
    return (script_chars / len(stripped)) < 0.30


def _is_acceptable_output(response: str, prompt: str, category: str = "") -> bool:
    text = response.strip()
    if len(text) < 1:
        return False

    refusal_phrases = [
        "i cannot", "i can't", "i'm not able", "i am unable",
        "i don't have the ability", "as an ai, i cannot", "as an ai",
        "i'm sorry", "i apologize", "i can't help with that",
        "is illegal", "unauthorized access", "against my safety guidelines",
        "i am programmed to be a helpful", "i cannot fulfill this request",
        "i cannot provide instructions", "i cannot assist with"
    ]
    text_lower = text.lower()
    if any(phrase in text_lower[:200] for phrase in refusal_phrases):
        return False

    compare_len = min(len(prompt), 300)
    echo_score = SequenceMatcher(
        None, text_lower[:compare_len], prompt.lower()[:compare_len]
    ).ratio()
    if echo_score > 0.85:
        return False

    if not _looks_like_english(text):
        return False

    if category == "sentiment_classification":
        allowed = [w for w in ["positive", "negative", "neutral", "mixed"] if w in prompt.lower()]
        if allowed and not any(w in text_lower for w in allowed):
            return False
    elif category == "mathematical_reasoning":
        if not any(char.isdigit() for char in text):
            return False
    elif category == "named_entity_recognition":
        if not any(char.isupper() for char in text):
            return False
    elif category in ("code_generation", "code_debugging"):
        code_markers = ["def ", "class ", "return", "select ", "from ", "{", "}", "(", ")", "```"]
        if not any(m in text_lower for m in code_markers):
            return False

    return True


async def process_task(task: dict, semaphore: asyncio.Semaphore) -> dict:
    """
    Processes a single task end to end: classify -> select model -> call ->
    validate -> (retry once on a different model if needed) -> return.

    Never raises -- any failure results in a best-effort answer (even if
    it's an error placeholder) so that one bad task cannot take down the
    entire batch and cause a total-loss non-zero exit.
    """
    task_id = task["task_id"]
    prompt = task["prompt"]

    async with semaphore:
        try:
            category = classify_category(prompt)
        except Exception as e:
            print(f"[{task_id}] classification failed: {e}", file=sys.stderr)
            category = "factual_knowledge"  # safe generic default

        # ── Local-model-first path (zero Fireworks tokens if it works) ──
        # Only attempted for categories a small bundled model can plausibly
        # get right (see model_selector.LOCAL_ELIGIBLE_CATEGORIES), and
        # only if a model was actually bundled and loaded successfully.
        # A correct local answer is the single best outcome for ranking:
        # full accuracy credit, zero tokens. A local answer that fails
        # validation falls through to the normal Fireworks flow below --
        # it never blocks or delays that path, just adds one extra
        # (free, local) attempt first.
        if category in LOCAL_ELIGIBLE_CATEGORIES:
            try:
                local_result = await local_model_client.call_local_model(prompt)
            except Exception as e:
                print(f"[{task_id}] local model call raised: {e}", file=sys.stderr)
                local_result = None

            if local_result and _is_acceptable_output(local_result["text"], prompt, category):
                return {"task_id": task_id, "answer": local_result["text"]}
            elif local_result:
                print(f"[{task_id}] local answer failed validation, falling back to Fireworks", file=sys.stderr)
            # local_result is None -> either no model bundled, or it
            # genuinely errored; either way, fall through silently to
            # Fireworks below, exactly as if this category weren't
            # local-eligible at all.

        tried_models = []
        last_result = None

        # Try the preferred model, then one retry with a different model
        # from the category's preference chain if the first attempt's
        # output fails validation or the call itself errors.
        for attempt in range(2):
            try:
                model_id = select_model(category, exclude=tried_models)
            except RuntimeError as e:
                print(f"[{task_id}] no available model: {e}", file=sys.stderr)
                break

            tried_models.append(model_id)

            try:
                result = await call_fireworks(prompt, model_id, category=category)
                last_result = result
                if _is_acceptable_output(result["text"], prompt, category):
                    return {"task_id": task_id, "answer": result["text"]}
                print(f"[{task_id}] output failed validation on {model_id}, retrying", file=sys.stderr)
            except Exception as e:
                print(f"[{task_id}] call to {model_id} failed: {e}", file=sys.stderr)

        # Both attempts exhausted -- return the best answer we have, even
        # if it didn't pass validation, rather than an empty string. A
        # non-empty best-effort answer has a chance at the accuracy gate;
        # an empty one has none.
        if last_result and last_result["text"].strip():
            return {"task_id": task_id, "answer": last_result["text"]}

        return {"task_id": task_id, "answer": ""}


async def run_batch(tasks: list[dict]) -> list[dict]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    coroutines = [process_task(task, semaphore) for task in tasks]
    return await asyncio.gather(*coroutines)


def main() -> int:
    start_time = time.monotonic()

    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"FATAL: could not read {INPUT_PATH}: {e}", file=sys.stderr)
        return 1

    if not isinstance(tasks, list) or not tasks:
        print(f"FATAL: {INPUT_PATH} did not contain a non-empty JSON array", file=sys.stderr)
        return 1

    print(f"Loaded {len(tasks)} tasks from {INPUT_PATH}")

    try:
        results = asyncio.run(run_batch(tasks))
    except Exception as e:
        print(f"FATAL: batch processing crashed: {e}", file=sys.stderr)
        return 1

    # Self-validate before writing: every task_id from the input must
    # appear exactly once in the output, and the whole thing must be
    # valid JSON -- both are explicit "scores zero" failure modes per the
    # rules, worth catching here rather than discovering at judging time.
    input_ids = {t["task_id"] for t in tasks}
    output_ids = {r["task_id"] for r in results}
    if input_ids != output_ids:
        missing = input_ids - output_ids
        extra = output_ids - input_ids
        print(f"FATAL: task_id mismatch. missing={missing}, extra={extra}", file=sys.stderr)
        return 1

    try:
        serialized = json.dumps(results, ensure_ascii=False, indent=2)
        json.loads(serialized)  # round-trip check
    except Exception as e:
        print(f"FATAL: results failed JSON self-validation: {e}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(serialized)

    elapsed = time.monotonic() - start_time
    print(f"Wrote {len(results)} results to {OUTPUT_PATH} in {elapsed:.1f}s")

    local_model_client.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())