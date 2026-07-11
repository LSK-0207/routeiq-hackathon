# RouteIQ

RouteIQ is a highly optimized, dynamic routing agent designed to maximize token efficiency and accuracy across diverse NLP and reasoning tasks.

## Overview
As AI tasks become increasingly varied in complexity, routing every request to a single flagship model creates massive inefficiencies. RouteIQ introduces a context-aware routing system designed to evaluate incoming prompts and distribute workloads across a tiered network of local and remote models. 

By analyzing the specific requirements of each request before execution, the system ensures that lightweight tasks are handled with maximum cost-efficiency, while complex logic and reasoning challenges are directed to specialized, high-performance models.

## Features
- **Dynamic Token Capping:** Uses dynamic constraint analysis to allocate the optimal required tokens for output generation, drastically reducing API spend.
- **Circuit Breaker Architecture:** Gracefully catches LLM refusals, output length violations, and local processing timeouts, falling back to reliable models without breaking the processing batch.
- **Zero-Token Execution:** Employs an asynchronous offline pipeline for basic classification tasks, completely bypassing remote API costs when appropriate.

## Instructions

### 1. Requirements
- Docker Desktop (ensure the backend is running)
- An active API Key for the designated cloud fallback proxy.

### 2. Building the Image
This repository is completely self-contained. To build the grading image, run the following command from the root of the project:

```bash
docker build -t routeiq-agent:latest .
```

### 3. Running the Agent
You can test the agent using the standard JSON test harness format. Ensure your host machine has a `test_harness` folder containing a `tasks.json` file.

```bash
docker run --rm \
   -e PYTHONUNBUFFERED=1 \
   -v "${PWD}/test_harness:/input" \
   -v "${PWD}/test_harness/output:/output" \
   -e TASKS_INPUT_PATH=/input/tasks.json \
   -e RESULTS_OUTPUT_PATH=/output/results.json \
   -e FIREWORKS_API_KEY="YOUR_API_KEY" \
   -e FIREWORKS_BASE_URL="YOUR_API_BASE_URL" \
   -e ALLOWED_MODELS="accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code" \
   routeiq-agent:latest
```

The output answers will be generated rapidly and cleanly saved to `test_harness/output/results.json`.
