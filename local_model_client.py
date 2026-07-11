"""
Local model client -- subprocess-isolated version.

See _local_model_worker.py's docstring for why this isolation exists: a
native crash inside llama-cpp-python's C library would otherwise kill the
entire batch process, losing every task's results, not just the
local-model ones. Running inference in a separate OS process means a
crash there is just a dead subprocess -- a normal, catchable condition --
rather than a total submission failure.

CIRCUIT BREAKER: if the worker ever dies (crash, or fails to load the
model at startup), local routing is permanently disabled for the rest of
this run. There's no value in respawning and retrying a model/library
combination that already proved broken -- every remaining local-eligible
task just falls through to Fireworks instead, same as if no model were
bundled at all.
"""

import asyncio
import json
import os
import select
import subprocess
import sys
import threading

from router import wants_detailed_response

_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "_local_model_worker.py")

# How long to wait for the worker to finish loading the model and print
# its "ready" signal. Model loading (reading a ~2GB file, initializing
# ggml structures) can take a while on a 2 vCPU box -- generous but still
# bounded, since the container must be ready within 60 seconds total.
STARTUP_TIMEOUT_SECONDS = 45.0

# Per-request generation timeout -- must stay safely under the 30-second
# per-task ceiling from the harness rules.
REQUEST_TIMEOUT_SECONDS = 20.0

_worker_process = None
_circuit_broken = False
_startup_attempted = False
_worker_lock = threading.Lock()

CONCISE_SYSTEM_PROMPT = (
    "You are a direct, concise AI. Output ONLY the answer. "
    "Do not include your internal reasoning. Do not explain what the user wants. "
    "Do not use filler text."
)

NEUTRAL_SYSTEM_PROMPT = (
    "You are an accurate, helpful assistant. "
    "Follow the task's instructions exactly, including any requested "
    "length, word count, or format. Do not repeat the question back to "
    "the user."
)


def _start_worker() -> bool:
    """
    Spawns the worker subprocess and waits for its readiness signal.
    Returns True if the worker is up and the model loaded successfully,
    False otherwise (missing model file, load failure, crash during
    load, or timeout) -- never raises for these expected failure modes.
    """
    global _worker_process, _circuit_broken, _startup_attempted

    if _startup_attempted:
        return _worker_process is not None and not _circuit_broken
    _startup_attempted = True

    model_path = os.environ.get(
        "LOCAL_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "models", "local-model.gguf"),
    )
    if not os.path.exists(model_path):
        print(
            f"[local_model] no model file at {model_path} -- "
            f"local routing disabled, all tasks will use Fireworks",
            file=sys.stderr,
        )
        _circuit_broken = True
        return False

    try:
        proc = subprocess.Popen(
            [sys.executable, _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
    except Exception as e:
        print(f"[local_model] failed to spawn worker process: {e}", file=sys.stderr)
        _circuit_broken = True
        return False

    # Wait for the "ready" line, but don't block forever -- if the model
    # load hangs or the worker crashes during load, we need to give up
    # and fall back rather than stall the whole batch.
    ready = False
    readable, _, _ = select.select([proc.stdout], [], [], STARTUP_TIMEOUT_SECONDS)
    if readable:
        line = proc.stdout.readline().strip()
        if line:
            try:
                msg = json.loads(line)
                ready = bool(msg.get("ready"))
            except json.JSONDecodeError:
                ready = False

    if not ready:
        # Either it crashed during load, or took too long. Capture
        # whatever stderr the worker produced (e.g. the GGML_ASSERT
        # crash message) for visibility in the batch's own logs, then
        # clean up and permanently disable local routing.
        stderr_output = ""
        try:
            proc.terminate()
            _, stderr_output = proc.communicate(timeout=5)
        except Exception:
            pass
        print(
            f"[local_model] worker failed to become ready "
            f"(crashed during load or timed out). stderr: {stderr_output[:500]}",
            file=sys.stderr,
        )
        _circuit_broken = True
        return False

    _worker_process = proc
    print(f"[local_model] worker ready, model loaded from {model_path}", file=sys.stderr)
    return True


def is_available() -> bool:
    return _start_worker()


def _call_worker_sync(prompt: str) -> dict | None:
    """
    Sends one prompt to the worker and reads back one response line.
    Runs inside a thread executor (see call_local_model) since this is
    blocking I/O on the subprocess pipes.

    Returns None if the worker has died (crash) or the call otherwise
    fails -- this permanently trips the circuit breaker so subsequent
    local-eligible tasks skip straight to Fireworks without re-attempting
    a worker that's already proven broken.
    """
    global _circuit_broken, _worker_process, _worker_lock

    with _worker_lock:
        if _worker_process is None or _worker_process.poll() is not None:
            print("[local_model] worker process is no longer running -- disabling local routing", file=sys.stderr)
            _circuit_broken = True
            return None

        try:
            sys_prompt = NEUTRAL_SYSTEM_PROMPT if wants_detailed_response(prompt) else CONCISE_SYSTEM_PROMPT
            
            _worker_process.stdin.write(json.dumps({
                "prompt": prompt,
                "system_prompt": sys_prompt
            }) + "\n")
            _worker_process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            print(f"[local_model] write to worker failed ({e}) -- worker likely crashed, disabling local routing", file=sys.stderr)
            _circuit_broken = True
            return None

        readable, _, _ = select.select([_worker_process.stdout], [], [], REQUEST_TIMEOUT_SECONDS)
        if not readable:
            print("[local_model] worker did not respond in time -- disabling local routing", file=sys.stderr)
            _circuit_broken = True
            try:
                _worker_process.terminate()
            except Exception:
                pass
            return None

        line = _worker_process.stdout.readline().strip()
        if not line:
            print("[local_model] worker closed its output (crashed) -- disabling local routing", file=sys.stderr)
            _circuit_broken = True
            return None

        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            print(f"[local_model] worker sent unparseable output: {line[:200]!r}", file=sys.stderr)
            return None

        if "error" in response:
            print(f"[local_model] worker reported an error: {response['error']}", file=sys.stderr)
            return None

        response["model_used"] = "local-bundled-model"
        return response


async def call_local_model(prompt: str) -> dict | None:
    """
    Generates an answer using the bundled local model, isolated in its
    own OS process. Returns None if no model is available, the circuit
    breaker has tripped, or this specific call failed -- in every case,
    the caller (main.py) should fall back to Fireworks.
    """
    if _circuit_broken or not is_available():
        return None

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_worker_sync, prompt)


def shutdown():
    """
    Cleanly terminates the worker process. Call this once at the end of
    main.py's batch run for a tidy exit -- not strictly required, since
    the container exits right after writing results.json anyway, but
    avoids leaving an orphaned subprocess if main.py is ever reused in a
    longer-lived context.
    """
    global _worker_process
    if _worker_process is not None and _worker_process.poll() is None:
        try:
            _worker_process.terminate()
            _worker_process.wait(timeout=5)
        except Exception:
            try:
                _worker_process.kill()
            except Exception:
                pass