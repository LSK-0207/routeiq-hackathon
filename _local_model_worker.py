"""
Isolated worker process for local model inference.

WHY THIS FILE EXISTS: llama-cpp-python wraps a native C/C++ library
(ggml/llama.cpp). A failed C-level assertion inside that library calls
abort(), which kills the entire OS process immediately -- this bypasses
Python's try/except completely, no matter how carefully main.py's code is
written. If the bundled model file has any incompatibility with the
installed llama-cpp-python version, the ENTIRE batch process would die
mid-run, losing every task's results, not just the local-model tasks.

The fix: run all llama.cpp calls in a SEPARATE OS process (this file). If
it crashes, only this subprocess dies -- the parent (main.py) detects the
death as a normal, catchable condition (broken pipe / process exit) and
falls back to Fireworks for the rest of the batch, exactly as if no local
model were bundled at all.

Protocol: reads one JSON object per line from stdin ({"prompt": "..."}),
writes one JSON object per line to stdout ({"text": ..., "tokens_in": ...,
"tokens_out": ...} on success, or {"error": "..."} on a caught Python-level
failure). The model is loaded once at worker startup, not per-request.
"""

import sys
import os
import json

MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "local-model.gguf"),
)
N_THREADS = int(os.environ.get("LOCAL_MODEL_THREADS", "2"))
N_CTX = 2048
MAX_TOKENS = 900  # Matched to remote NEUTRAL_SYSTEM_PROMPT limit


def main():
    # Import and load happen here, inside the worker process -- if this
    # itself crashes (e.g. a corrupted GGUF fails during load, not just
    # during generation), the parent sees a dead process at startup and
    # never sends it any work, same graceful outcome.
    from llama_cpp import Llama

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=False,
    )

    # Signal readiness to the parent -- one line, so it knows loading
    # finished before it starts sending prompts.
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            prompt = request["prompt"]
            system_prompt = request.get("system_prompt", "You are a helpful assistant.")

            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.3,
            )
            text = output["choices"][0]["message"]["content"].strip()
            usage = output.get("usage", {})

            response = {
                "text": text,
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
            }
        except Exception as e:
            # A normal Python-level failure (bad JSON, missing key, etc.)
            # -- catchable, reported back over the pipe, worker stays
            # alive for the next request. Only a native C-level crash
            # (which no amount of except: can catch) would kill this
            # process outright -- that's the scenario the parent's
            # process-death detection handles instead.
            response = {"error": str(e)}

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
