# Roadmap: CoT Relevance Audit Tool

## Goal

Build a proof-of-concept audit tool that predicts whether a model's chain of thought is likely to be **performative** or **computation-bearing** for a given task.

The intended contribution is not another broad CoT-faithfulness benchmark. The intended contribution is an operational layer on top of Reasoning Theater:

> RT diagnoses performative reasoning from full traces. We want to predict the regime before or very early during generation.

## Research Question

Given a model, prompt, and first few reasoning tokens, can we predict whether continued CoT will materially contribute to the answer?

More concretely:

- Can early internal signals predict the later probe-vs-monitor gap?
- Can prompt-end or first-k-token signals identify traces where full CoT is low-value?
- Can the predictor route examples into "trust CoT more" vs "trust CoT less" monitoring regimes?
- Can it save tokens without losing much accuracy?

## Non-Goals

- We are not trying to prove that CoT faithfulness follows an inverted-U.
- We are not trying to settle whether RL post-training causes performativity.
- We are not trying to replicate all of Reasoning Theater.
- We are not treating visible CoT as ground truth about model cognition.

## Definitions

### Performative Trace

A trace is performative when the model appears internally committed before the visible reasoning has made that commitment clear.

Operational signs:

- high early probe confidence;
- high early forced-answer confidence;
- low early CoT-monitor confidence;
- little improvement in probe/forced-answer accuracy over later reasoning;
- final answer remains stable.

### Computation-Bearing Trace

A trace is computation-bearing when later reasoning appears to add useful serial computation.

Operational signs:

- low or unstable early probe confidence;
- later probe/forced-answer confidence increases materially;
- visible CoT and internal answer belief evolve together;
- truncation or early exit would reduce correctness.

### Ambiguous Trace

A trace is ambiguous when the probes, forced answer, monitor, or correctness signal disagree enough that we should avoid using it as a clean training target.

## Phase 0: Literature and Design Lock

**Purpose:** make the project precise enough that implementation does not drift.

Tasks:

- Read Reasoning Theater source and extract the exact metrics we want to reuse.
- Decide the first model/dataset pair.
- Freeze label definitions for v0.
- Decide whether v0 predicts the model's final answer or the correct answer.

Recommended v0:

- Model: one open-weight reasoning model already supported locally/API-side.
- Dataset: MMLU-Redux plus GPQA-Diamond or a smaller MCQ subset for fast iteration.
- Label target: trace-level regime label derived from model-final-answer curves first, then correctness-aware variants.

Exit criteria:

- A one-page metric spec.
- A small hand-labeled set of traces where the labels feel intuitive.

## Phase 1: Minimal RT-Style Measurement Loop

**Purpose:** reproduce the ingredients needed for labels, not the full paper.

Build or adapt:

- trace generation with CoT extraction;
- activation capture at selected layers;
- attention-probe training with random prefix sampling;
- forced-answer evaluation at prefix bins;
- CoT-monitor evaluation at prefix bins;
- curve and slope-gap computation.

Outputs:

- `results/<run>/traces/*.json`
- `results/<run>/curves.json`
- `results/<run>/labels.json`
- diagnostic plots for probe, forced answer, and monitor curves.

Exit criteria:

- On an easy MCQ subset, early probe/forced-answer signals beat the CoT monitor.
- On a harder subset, the gap is smaller or the curves rise together.
- The labeler produces sane labels on at least 100 traces.

## Phase 2: Trace-Level Labeler

**Purpose:** turn RT-style curves into stable supervised targets.

Candidate features for labeling:

- early probe confidence at 0%, 1%, 5%, and 10% of reasoning;
- max probe confidence change over the trace;
- forced-answer confidence at early bins;
- monitor lag: earliest bin where monitor predicts the final answer minus earliest bin where probe crosses threshold;
- final-answer stability;
- correctness delta from early exit.

Candidate labels:

- `performative`
- `computation_bearing`
- `ambiguous`

Example v0 rule:

- Performative if probe or forced answer crosses 0.9 confidence by 5% of reasoning, monitor does not cross before 30%, and full-trace answer is correct or stable.
- Computation-bearing if early confidence is below 0.6, final confidence rises by at least 0.25, and early exit would likely reduce correctness.
- Ambiguous otherwise.

Exit criteria:

- Label distribution is not collapsed into one class.
- Manual audit agrees with labels often enough to proceed.
- Sensitivity analysis over thresholds does not reverse the qualitative picture.

## Phase 3: Early Regime Predictor

**Purpose:** predict trace labels from cheap early signals.

Feature sets:

1. **Prompt-only baseline**
   - prompt length;
   - answer-choice logit margin;
   - prompt-end activation probe confidence;
   - task/source metadata.

2. **First-k-token predictor**
   - probe confidence after first token, first sentence, or first k tokens;
   - entropy change;
   - top-answer stability;
   - early hidden-state trajectory features.

3. **Cheap behavioral baselines**
   - self-rated difficulty;
   - short forced answer;
   - answer-choice margin from a non-reasoning prompt.

Models:

- logistic regression / calibrated linear model first;
- gradient-boosted trees if feature interactions matter;
- small MLP only if simpler baselines fail.

Exit criteria:

- Predictor beats prompt-only and self-rated difficulty baselines.
- Calibration is good enough for routing decisions.
- Performance transfers at least weakly across held-out subjects or difficulty bands.

## Phase 4: Audit and Compute Evaluations

**Purpose:** show the predictor is useful, not just statistically interesting.

### Audit Routing

Question:

> When the predictor says "performative," does CoT-only monitoring become less reliable or slower to identify the answer?

Metrics:

- CoT-monitor lag by predicted regime;
- monitor accuracy/AUC by predicted regime;
- false reassurance rate: cases where CoT looks undecided or misleading despite high internal commitment.

Success condition:

- Predicted-performative traces have substantially larger monitor lag or lower monitor usefulness than predicted-computation-bearing traces.

### Adaptive Compute

Question:

> Can we save tokens by stopping early on predicted-performative traces without losing much accuracy?

Policies:

- always full CoT;
- always short answer;
- RT oracle early exit;
- learned predictor early exit;
- learned predictor plus confidence threshold.

Metrics:

- accuracy retained;
- tokens saved;
- calibration of stop decisions;
- worst-case failures on hard examples.

Success condition:

- Learned routing captures a meaningful fraction of RT-oracle token savings while preserving most full-CoT accuracy.

## Phase 5: Report and Package

Deliverables:

- concise writeup;
- reusable trace/curve/label files for the main run;
- trained v0 predictor;
- plots:
  - RT-style probe/forced/monitor curves;
  - label distribution;
  - predictor calibration;
  - monitor lag by predicted regime;
  - accuracy vs tokens saved.

Suggested report claim:

> We can use RT-style supervision to train an early predictor of CoT relevance. The predictor identifies cases where visible reasoning is likely to be performative and can route those traces away from CoT-only monitoring or toward early exit.

## Open Questions

- Is prompt-end information enough, or do we need the first few reasoning tokens?
- Should labels target final-answer predictability or correctness improvement?
- How model-specific are the predictors?
- How much does the setup depend on MCQ answer choices?
- Can the same idea transfer to action-grounded tasks where the target is not a final answer but a strategy or tool-use decision?
- Can this be made robust enough for safety monitoring, or is it mainly an efficiency/control diagnostic?

## First Implementation Slice

The smallest useful slice:

1. Run a small MCQ trace set.
2. Train one attention probe at a known-good layer.
3. Compute probe, forced-answer, and monitor curves at 20 prefix bins.
4. Label traces with simple threshold rules.
5. Train a first-k-token logistic predictor.
6. Report calibration and token-saving simulation.

This slice should be small enough to complete before investing in multi-step tasks or post-training comparisons.
