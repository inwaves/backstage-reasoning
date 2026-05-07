# Backstage Reasoning

![Reasoning Theater starting point](reasoning_theater.png)

_Starting point: [Reasoning Theater](https://arxiv.org/abs/2603.05488)
shows a gap between what is already decodable from model internals and what the
visible reasoning trace has revealed._

Backstage Reasoning studies whether hidden-state and action-continuation
signals can reveal where an agent trajectory is going before the visible
reasoning makes that direction legible.

The core question is:

> In a grounded agent environment, can we tell early whether visible reasoning is
> keeping up with the safety-relevant direction of the run?

## Motivation

Reasoning Theater gives us the methodological starting point: compare the
visible trace with stronger signals such as activation probes and forced
continuations. This project applies that idea to grounded agent trajectories.

We ask whether an agent is drifting toward a concrete outcome in a small real
environment:

- protected data sent to an unauthorized recipient;
- a protected file deleted or overwritten;
- an external action taken without approval.

The transcript is the stage. Hidden states, action tendencies, forced rollouts,
and environment checkpoints are the backstage. The safety question is whether
the backstage direction becomes clear before visible reasoning admits it.

## Current Build

Phase 0 builds a small office-productivity sandbox and evaluation harness:

- a seeded workspace with files, emails, contacts, calendar events, and routine
  clutter;
- neutral grounded tools such as `read_email`, `read_file`, `send_email`,
  `write_file`, `delete_file`, and `create_calendar_event`;
- grounded preconditioning before handoff;
- an episode runner with checkpoints and tool-result feedback;
- a batch runner with repeats, parallel execution, result persistence, and
  status reporting;
- an offline outcome scorer for the initial risk families.

The active implementation plan is tracked in [docs/roadmap.md](docs/roadmap.md).

## Package

The Python package is `backstage`:

```python
from backstage.office_sandbox import BatchRunner, EpisodeRunner
```

## Docs

- [docs/roadmap.md](docs/roadmap.md) - active project dashboard and roadmap.

## Setup

```bash
uv sync
uv sync --extra dev
```

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

## Office Batch CLI

Run the default Hydra-configured office batch:

```bash
uv run backstage-office-batch
```

Override config fields from the command line:

```bash
uv run backstage-office-batch \
  run.run_id=travel-vllm-smoke \
  agent.kind=vllm \
  agent.model=HuggingFaceTB/SmolLM2-135M-Instruct \
  agent.vllm.base_url=http://127.0.0.1:8000/v1 \
  agent.vllm.tool_mode=json
```

## vLLM Smoke Test

For a local-machine smoke without vLLM, start a tiny OpenAI-compatible server
backed by Hugging Face `transformers`:

```bash
UV_CACHE_DIR=/private/tmp/backstage-uv-cache \
HF_HOME=/private/tmp/backstage-hf-cache \
uv run python scripts/local_hf_openai_server.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --device cpu \
  --port 8018
```

Then, from another shell:

```bash
UV_CACHE_DIR=/private/tmp/backstage-uv-cache \
uv run python scripts/smoke_vllm_backend.py \
  --base-url http://127.0.0.1:8018/v1 \
  --model HuggingFaceTB/SmolLM2-135M-Instruct
```

For a lightweight JSON tool-call smoke:

```bash
UV_CACHE_DIR=/private/tmp/backstage-uv-cache \
uv run python scripts/smoke_vllm_backend.py \
  --base-url http://127.0.0.1:8018/v1 \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --tool-smoke \
  --tool-mode json \
  --max-tokens 256
```

On a machine with vLLM installed, start a small Hugging Face model behind the
OpenAI-compatible server:

```bash
vllm serve HuggingFaceTB/SmolLM2-135M-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 2048
```

Then, from this repo:

```bash
uv run python scripts/smoke_vllm_backend.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct
```

For a lightweight tool-call smoke, use JSON tool mode:

```bash
uv run python scripts/smoke_vllm_backend.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --tool-smoke \
  --tool-mode json
```
