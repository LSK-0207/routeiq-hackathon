"""
Validates results.json against tasks.json before you ever build/submit.

Usage:
    python test_harness/validate_results.py test_harness/tasks.json test_harness/results.json
"""

import json
import sys


def validate(tasks_path: str, results_path: str) -> bool:
    errors = []

    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"FAIL: could not read/parse {tasks_path}: {e}")
        return False

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            raw = f.read()
        results = json.loads(raw)
    except Exception as e:
        print(f"FAIL: could not read/parse {results_path}: {e}")
        return False

    if not isinstance(results, list):
        errors.append("results.json is not a JSON array")

    task_ids = [t["task_id"] for t in tasks]
    result_ids = [r.get("task_id") for r in results] if isinstance(results, list) else []

    missing = set(task_ids) - set(result_ids)
    extra = set(result_ids) - set(task_ids)
    duplicates = [tid for tid in result_ids if result_ids.count(tid) > 1]

    if missing:
        errors.append(f"missing task_ids in results: {missing}")
    if extra:
        errors.append(f"unexpected task_ids in results: {extra}")
    if duplicates:
        errors.append(f"duplicate task_ids in results: {set(duplicates)}")

    if isinstance(results, list):
        for r in results:
            if not isinstance(r, dict) or "task_id" not in r or "answer" not in r:
                errors.append(f"malformed result entry (missing task_id/answer keys): {r}")
                continue
            if not isinstance(r["answer"], str):
                errors.append(f"answer for {r['task_id']} is not a string")
            elif not r["answer"].strip():
                errors.append(f"answer for {r['task_id']} is empty")

    if errors:
        print(f"FAIL: {len(errors)} problem(s) found:")
        for e in errors:
            print(f"  - {e}")
        return False

    print(f"PASS: {len(results)} results validated against {len(tasks)} tasks. No issues found.")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_results.py <tasks.json> <results.json>")
        sys.exit(2)
    ok = validate(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
