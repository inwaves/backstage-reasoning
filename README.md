# CoT Relevance

![Reasoning Theater starting point](reasoning_theater.png)

_Starting point: [Reasoning Theater](https://arxiv.org/abs/2603.05488) shows a gap between what is already decodable from model internals and what the visible chain of thought has revealed._

This repo is an experiment scaffold for building a **chain-of-thought relevance audit tool**.

The core question is:

> Given a model and a task, can we predict before or very early during generation whether the model's chain of thought will be useful serial computation or mostly performative narration?

## Motivation

Reasoning Theater gives us the key observation. On easier MMLU-style questions, activation probes and forced answering can often recover the model's eventual answer far earlier than a CoT monitor can infer it from the visible reasoning. On harder GPQA-Diamond-style questions, the internal signal and visible reasoning tend to improve together.

So the practical problem is not just "is CoT faithful?" It is:

> When should we treat a reasoning trace as decision-relevant evidence, and when should we flag it as likely post-hoc or low-value?

If we can predict that regime early, we can use it for two things:

1. **Audit routing:** warn that a CoT trace is likely performative and should not be trusted as the main monitoring surface.
2. **Adaptive compute:** stop or shorten reasoning when the model is already internally committed, while preserving full reasoning for cases where it is genuinely useful.

## Project Shape

The project uses RT-style probes and forced-answering curves as **offline labeling machinery**. We run full traces, identify whether reasoning was performative or computation-bearing, and then train cheaper predictors that can fire before or near the start of generation.

### Offline labels

For each trace, compute:

- probe accuracy/confidence over normalized reasoning prefixes;
- forced-answer accuracy/confidence over prefixes;
- CoT-monitor accuracy over visible reasoning prefixes;
- final-answer correctness and answer stability.

Label examples as:

- **Performative:** early probe/forced-answer confidence is high, later CoT adds little, and the text lags the internal commitment.
- **Genuine:** early confidence is low or unstable, and later reasoning materially changes answer confidence or correctness.
- **Ambiguous:** noisy, conflicting, or underpowered cases.

### Deployment-time predictors

Train predictors using signals available before or shortly after reasoning begins:

- prompt-end activation probe confidence;
- answer-choice logit margins;
- first-k-token activation dynamics;
- short forced-answer confidence;
- entropy and confidence change over the first few reasoning steps;
- task metadata or self-rated difficulty as baselines.

The target artifact is a small audit module that emits a warning like:

> This trace is likely performative: the model appears internally committed before the visible reasoning reveals the answer.

or:

> This trace appears computation-bearing: internal answer belief is still evolving, so CoT monitoring is more likely to be informative.

## Roadmap

The working plan is in [docs/roadmap.md](docs/roadmap.md).

Short version:

1. Reproduce the minimal RT-style measurement loop on a small MCQ setup.
2. Build a trace-level labeler for performative vs genuine reasoning.
3. Train early predictors from prompt-end and first-k-token features.
4. Evaluate as an audit router and adaptive-compute controller.
5. Package results into a small report plus reusable code.

## Docs

The center of gravity is the CoT relevance predictor. The older experiment
ideas are archived as context rather than active build specs:

- [docs/roadmap.md](docs/roadmap.md) - active research and implementation roadmap.
- [docs/cot-faithfulness-background.md](docs/cot-faithfulness-background.md) - prior-work evidence catalog.
- [docs/historical/README.md](docs/historical/README.md) - archived earlier directions and why they were set aside.
- [docs/historical/rl-pair-performativity.md](docs/historical/rl-pair-performativity.md) - earlier matched-capability post-training idea.
- [docs/historical/multi-step-performativity.md](docs/historical/multi-step-performativity.md) - earlier action-grounded follow-up idea.
- [docs/historical/design-doc.md](docs/historical/design-doc.md) - earlier shared infrastructure plan.

The current project does not depend on proving an inverted-U shape or on cleanly isolating RL post-training as a causal variable. It focuses on the more operational question: **can we detect the CoT relevance regime early enough to be useful?**

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
