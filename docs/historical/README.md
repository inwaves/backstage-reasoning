# Historical Work

This folder holds the earlier research directions that led into the current
CoT relevance project. They are useful context, but they are not the active
implementation plan.

## What Changed

The original framing asked whether chain-of-thought performativity was driven
by post-training, task difficulty, or richer multi-step task structure. After
reading the surrounding literature, especially Reasoning Theater, the project
pivoted to a more operational question:

> Given a model and task, can we predict before or very early during generation
> whether the chain of thought will be computation-bearing or mostly
> performative?

The active roadmap is now [../roadmap.md](../roadmap.md).

## Contents

- [rl-pair-performativity.md](rl-pair-performativity.md) - an earlier
  matched-capability design for testing whether post-training increases
  performative CoT. This remains a plausible follow-up, but it is not the
  shortest path to a useful audit tool.
- [multi-step-performativity.md](multi-step-performativity.md) - an earlier
  action-grounded design for measuring factor verbalization in optimization
  tasks. This is still interesting as a later extension once the basic
  relevance predictor exists.
- [design-doc.md](design-doc.md) - the shared infrastructure design for those
  older experiments. Treat it as a historical sketch, not as the current build
  specification.

## Why Archive These Instead of Delete Them?

The old directions are not wrong; they are just less direct. They try to answer
why performativity emerges or whether it generalizes to richer domains. The
current project first asks whether we can detect the regime early enough to be
useful. If that works, the archived designs become natural second-stage
experiments.
