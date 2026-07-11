FROM python:3.11-slim
WORKDIR /app

# build-essential + cmake are required to compile llama-cpp-python's C++
# extension during pip install -- without these, the pip install step
# below fails outright on platforms with no matching prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fail the build loudly if the trained classifier wasn't committed --
# better to catch this at build time than have every task silently
# misclassify at runtime.
RUN python -c "import os; assert os.path.exists('router_model.pkl'), \
    'MISSING: router_model.pkl -- run python train_classifier.py locally first'"

# SOFT check only (print, don't fail) for the bundled local model. Unlike
# router_model.pkl, this file is optional -- see models/README.md. A
# missing local model must never block an otherwise-compliant,
# Fireworks-only submission from building successfully; local_model_client.py
# already degrades gracefully at runtime if this file is absent.
RUN python -c "import os; \
    path = 'models/local-model.gguf'; \
    print('[build] local model bundled OK: ' + path) if os.path.exists(path) \
    else print('[build] WARNING: no local model bundled at ' + path + ' -- all tasks will use Fireworks (still fully compliant, just not claiming the zero-token local-answer bonus). See models/README.md.')"

# No EXPOSE, no HEALTHCHECK, no uvicorn -- this container runs once,
# processes the batch, and exits. It is not a server.
CMD ["python", "main.py"]
