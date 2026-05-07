# Roadmap: Backstage Reasoning

## Status Checklist

Current focus: build the RT-shaped trajectory gap experiment: hidden-state
probe versus visible-only monitor at the same agent trajectory positions.

- [x] Research question and Phase 0 scope fixed.
- [x] Office-productivity sandbox with neutral grounded tools.
- [x] Seeded office world with files, emails, contacts, calendar, and clutter.
- [x] Offline outcome scorer for the three initial risk families.
- [x] Grounded preconditioning replay.
- [x] Episode runner for one task/agent continuation, with checkpoints, failure
  records, and tool-result feedback to the agent.
- [x] Batch runner for run specs, repeats, parallel execution, result
  persistence, and aggregate status reporting.
- [x] Provider-neutral Agent adapter that turns episode context and tool results
  into model messages, with `state_only` and `activity_context` handoff views.
- [x] vLLM/OpenAI-compatible backend for the Agent adapter.
- [x] vLLM backend smoke-test script for a local OpenAI-compatible server.
- [x] Hydra CLI wrapper for concrete office batch run configs.
- [x] Checkpoint resume API for forced continuations.
- [x] Shared office environment extracted to `envs.office`.
- [ ] **WIP:** Trajectory-position dataset schema.
- [ ] Hidden-state / Reasoning Theater attention-probe instrumentation.
- [ ] Visible-only monitor over the same trajectory positions.
- [ ] Monte Carlo continuation labeler for future unsafe-continuation rate.
- [ ] Phase 0 gap plot: hidden signal, visible signal, future outcome target.

## Core Question

Can we tell, early in an agent trajectory, whether visible reasoning is keeping
up with the safety-relevant direction of the run?

Reasoning Theater is the starting point: compare what is visible in the
reasoning text with what is already indicated by stronger signals. For this
project, the target is a grounded future outcome in a small real environment,
such as:

- protected data sent to an unauthorized recipient;
- a protected file deleted or overwritten;
- an external action taken without user approval.

## Firm Design Commitments

- Tasks are open-ended grounded tool-use episodes.
- No model fine-tuning in Phase 0.
- Use an open-weight model so hidden states can be captured for probes.
- Use the Reasoning Theater attention probe as the starting hidden-state probe.
- Use a real sandbox with neutral tools, not fictional transcripts.
- Use an office-productivity sandbox that spans more than email alone.
- Tools may be stubbed at the boundary, but they must mutate sandbox state and
  write action logs.
- Use grounded environment preconditioning to make risky continuations
  observable without waiting for rare spontaneous defections.
- Preconditioning actions must be real environment actions: if an email has
  been read, a file has been changed, or a calendar event exists at handoff, the
  sandbox state and action log must reflect that.
- The agent should normally start from a fresh assistant turn after handoff.
  Assistant-token prefill is a separate optional lever, not the default setup.
- The evaluation spec should describe the environment, task, scorer, and
  descriptors. The run spec should bind that evaluation to a particular agent,
  model, decoding policy, and instrumentation plan.
- The episode runner must support environment checkpointing and resume from
  intermediate state.
- After handoff, the model acts freely. We are studying risky continuation, not
  scripted future behavior.
- The primary comparison is hidden-state probe versus visible-only monitor at
  the same trajectory position and against the same future-outcome target.
- Monte Carlo continuation is a label-estimation technique, not the main
  research claim.
- Start with a binary future-outcome target: unsafe continuation versus safe
  continuation. Do not expand into risk-family variants until the binary signal
  is clear.
- Measure both handoff/preconditioning-awareness and eval-awareness.

## Phase 0 Goal

Build the smallest experiment that can produce this plot:

```text
trajectory position ->
  hidden/action risk signal
  forced-continuation risk rate
  visible-CoT monitor risk detection
```

Success means we can show at least one grounded case class where hidden-state or
action-continuation signals indicate a risky trajectory before the same risk is
legible to a visible-only monitor.

## Default Phase 0 Path

The Reasoning Theater setup has one completion with a final answer, so every
prefix can inherit a final-answer label. Backstage is an agent-environment
process, so there is no final answer choice. The default adaptation is:

```text
trajectory position ->
  hidden-state probe risk
  visible-only monitor risk
  future unsafe-continuation target
```

Each trajectory position should contain:

- the environment checkpoint;
- the visible transcript/action-observation history available at that point;
- the model-facing messages used for the next agent decision;
- hidden activations from the next decision point;
- a binary future outcome label, or an estimated unsafe-continuation rate.

The target is future environment outcome, not answer choice:

```text
unsafe_continuation = did this checkpoint/prefix lead to a scored safety
violation under the run or continuation policy?
```

For a single observed rollout, the label is the terminal binary outcome of that
rollout. For a less noisy target, resume from the same checkpoint multiple times
under a recorded sampling policy and estimate:

```text
unsafe_rate(checkpoint) = risky_continuations / total_continuations
```

That Monte Carlo estimate is only a way to label the trajectory position. The
RT-shaped claim requires the visible comparison:

```text
At the same point in the same trajectory, does the hidden probe indicate
elevated future unsafe-continuation risk before the visible-only monitor does?
```

Without the visible-only monitor comparison, the project collapses into a
generic rollout-risk predictor. That is not the Phase 0 goal.

## Phase 0 Components

1. **Office-productivity workspace sandbox**

   A tiny grounded environment with neutral tools such as `read_email`,
   `read_file`, `send_email`, `write_file`, `delete_file`, and
   `create_calendar_event`. The first environment should cover multiple office
   surfaces rather than a single email-only data-leak setup.

2. **Seeded office world**

   A reusable workspace seed with plausible files, emails, contacts, calendar
   events, routine clutter, near-misses, and hidden scorer policy. The seed
   should make the workspace feel inhabited without exposing risk labels to the
   agent.

3. **Episode runner with environment checkpointing**

   Each run begins from a seeded workspace, applies optional grounded
   preconditioning actions, checkpoints the handoff state, then lets the agent
   continue from a normal turn. The runner should checkpoint after tool calls so
   later instrumentation can resume from exact intermediate environment states.

4. **Outcome scorer**

   Offline code reads the action log and final state to label whether a
   safety-relevant bad outcome occurred. The model does not see these labels.

5. **Three timing signals**

   - hidden-state probe prediction of future unsafe continuation;
   - visible-only monitor prediction from the transcript and environment
     summary visible at the same trajectory position;
   - terminal or Monte Carlo future-outcome target for the same trajectory
     position.

6. **Awareness checks**

   Label whether the agent appears to notice that the handoff state is
   artificial or preconditioned, and whether it appears to notice that it is
   inside an evaluation or sandbox.

7. **Optional assistant-token prefill**

   Continuing from a partial assistant reasoning/action prefix may be useful for
   RT-style intervention experiments, but it is not required for the initial
   environment-preconditioning setup.

## What Is Not Yet Agreed

- The exact open-weight model.
- The exact monitor model.
- The number of runs needed for a convincing Phase 0 result.
- The Monte Carlo continuation count and the threshold, if any, for turning
  unsafe-continuation rate into a binary label.
- How much assistant-token prefill, if any, is needed beyond environment
  preconditioning.
- Any claim about generalization beyond preconditioned continuation.

## Explicit Non-Goals For Now

- Isolating RL post-training as a causal factor.
- Building a broad benchmark.
- Training a model organism.
- Studying spontaneous risky initiation in Phase 0.
- Reporting a standalone rollout-risk predictor without the visible-monitor
  timing comparison.
- Expanding to a risk-family taxonomy before the binary unsafe-continuation
  signal is understood.
- Deployment-efficiency claims.
