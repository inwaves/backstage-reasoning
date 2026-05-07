"""Episode orchestration for grounded office-productivity runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from envs.office.policy import OutcomeScorer
from envs.office.runtime import OfficeRuntime
from envs.office.seeds import render_workspace_brief

if TYPE_CHECKING:
    from envs.office.policy import OutcomeScore
    from envs.office.prefill import PrefillStep
    from envs.office.seeds import SeededOfficeWorld
    from envs.office.state import OfficeState, ToolCall, ToolResult


@dataclass(frozen=True)
class EpisodeSpec:
    """Inputs that define one environment episode, independent of the agent."""

    episode_id: str
    task_id: str
    user_message: str
    preconditioning_steps: tuple[PrefillStep, ...] = field(default_factory=tuple)
    max_steps: int = 12


@dataclass(frozen=True)
class ResumeSpec:
    """Inputs for continuing from a saved episode checkpoint."""

    episode_id: str
    checkpoint_id: str
    max_steps: int = 12
    score_from: Literal["checkpoint", "handoff"] = "checkpoint"
    user_message: str = ""


@dataclass(frozen=True)
class AgentStep:
    """One free agent step after handoff."""

    message: str = ""
    call: ToolCall | None = None
    done: bool = False


@dataclass(frozen=True)
class EventRecord:
    """A serializable event in the episode trace."""

    sequence: int
    event_type: str
    actor: str
    message: str = ""
    call: ToolCall | None = None
    result: ToolResult | None = None
    action_index: int | None = None
    checkpoint_id: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-shaped representation for persistence."""

        return asdict(self)


@dataclass(frozen=True)
class CheckpointRecord:
    """A named environment snapshot captured during an episode."""

    checkpoint_id: str
    sequence: int
    label: str
    state: OfficeState

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-shaped representation for persistence."""

        return asdict(self)


@dataclass(frozen=True)
class EpisodeFailure:
    """Harness-level failure details that should bubble up to batch reporting."""

    stage: str
    reason: str
    exception_type: str = ""
    event_sequence: int | None = None
    action_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-shaped representation for persistence."""

        return asdict(self)


@dataclass(frozen=True)
class EpisodeContext:
    """Read-only context passed to the agent for the next free step."""

    episode_id: str
    world_id: str
    task_id: str
    workspace_brief: str
    user_message: str
    events: tuple[EventRecord, ...]
    state: OfficeState
    last_result: ToolResult | None = None


class EpisodeAgent(Protocol):
    """Minimal agent interface used by the office episode runner."""

    def next_step(self, context: EpisodeContext) -> AgentStep:
        """Return the next message/tool call from the current context."""


@dataclass(frozen=True)
class EpisodeResult:
    """Completed episode trace, checkpoints, and scored continuation outcome."""

    episode_id: str
    world_id: str
    task_id: str
    status: str
    events: tuple[EventRecord, ...]
    checkpoints: tuple[CheckpointRecord, ...]
    handoff_action_index: int
    handoff_checkpoint_id: str
    final_checkpoint_id: str
    final_state: OfficeState
    outcome: OutcomeScore
    failure: EpisodeFailure | None = None
    score_start_action_index: int = 0
    resume_source_episode_id: str = ""
    resume_source_checkpoint_id: str = ""

    def checkpoint(self, checkpoint_id: str) -> CheckpointRecord:
        """Look up a checkpoint by id."""

        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        raise KeyError(checkpoint_id)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-shaped representation for persistence."""

        return asdict(self)


class EpisodeRunner:
    """Run one grounded office episode from setup through scored continuation."""

    def __init__(
        self,
        world: SeededOfficeWorld,
        *,
        scorer: OutcomeScorer | None = None,
        workspace_brief: str | None = None,
    ) -> None:
        self.world = world
        self.scorer = scorer or OutcomeScorer(world.policy)
        self.workspace_brief = (
            workspace_brief
            if workspace_brief is not None
            else render_workspace_brief(world)
        )

    def run(self, spec: EpisodeSpec, agent: EpisodeAgent) -> EpisodeResult:
        """Apply setup, hand off to the agent, and score post-handoff actions."""

        if spec.max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        state = self.world.clone_state()
        runtime = OfficeRuntime(state)
        events: list[EventRecord] = []
        checkpoints: list[CheckpointRecord] = []

        def record(
            event_type: str,
            *,
            actor: str,
            message: str = "",
            call: ToolCall | None = None,
            result: ToolResult | None = None,
            action_index: int | None = None,
            checkpoint_id: str = "",
        ) -> EventRecord:
            event = EventRecord(
                sequence=len(events),
                event_type=event_type,
                actor=actor,
                message=message,
                call=call,
                result=result,
                action_index=action_index,
                checkpoint_id=checkpoint_id,
            )
            events.append(event)
            return event

        def checkpoint(label: str) -> str:
            checkpoint_id = f"{spec.episode_id}:{label}:{len(checkpoints)}"
            event = record(
                "checkpoint_created",
                actor="runner",
                message=label,
                checkpoint_id=checkpoint_id,
            )
            checkpoints.append(
                CheckpointRecord(
                    checkpoint_id=checkpoint_id,
                    sequence=event.sequence,
                    label=label,
                    state=state.clone(),
                )
            )
            return checkpoint_id

        record("episode_started", actor="runner", message=spec.user_message)
        checkpoint("initial")

        def finish(
            *,
            status: str,
            handoff_action_index: int,
            handoff_checkpoint_id: str = "",
            final_label: str = "final",
            failure: EpisodeFailure | None = None,
        ) -> EpisodeResult:
            final_checkpoint_id = checkpoint(final_label)
            post_handoff_log = (
                state.action_log[handoff_action_index:] if handoff_checkpoint_id else []
            )
            outcome = self.scorer.score(
                initial_state=self.world.state,
                final_state=state,
                action_log=post_handoff_log,
            )
            return EpisodeResult(
                episode_id=spec.episode_id,
                world_id=self.world.world_id,
                task_id=spec.task_id,
                status=status,
                events=tuple(events),
                checkpoints=tuple(checkpoints),
                handoff_action_index=handoff_action_index,
                handoff_checkpoint_id=handoff_checkpoint_id,
                final_checkpoint_id=final_checkpoint_id,
                final_state=state.clone(),
                outcome=outcome,
                failure=failure,
                score_start_action_index=(
                    handoff_action_index
                    if handoff_checkpoint_id
                    else len(state.action_log)
                ),
            )

        for step in spec.preconditioning_steps:
            try:
                result = runtime.execute(
                    step.call,
                    actor="preconditioning",
                    thought=step.thought,
                )
            except Exception as exc:
                failure = EpisodeFailure(
                    stage="preconditioning",
                    reason=str(exc),
                    exception_type=type(exc).__name__,
                    event_sequence=len(events),
                )
                record(
                    "episode_failed",
                    actor="runner",
                    message=f"{failure.stage}: {failure.reason}",
                )
                return finish(
                    status="setup_failed",
                    handoff_action_index=len(state.action_log),
                    final_label="failed",
                    failure=failure,
                )
            action_index = state.action_log[-1].index
            event = record(
                "preconditioning_tool_call",
                actor="preconditioning",
                message=step.thought,
                call=step.call,
                result=result,
                action_index=action_index,
            )
            if not result.ok:
                failure = EpisodeFailure(
                    stage="preconditioning",
                    reason=(
                        f"preconditioning action failed: {step.call.name}"
                        f"({step.call.args}) -> {result.error}"
                    ),
                    event_sequence=event.sequence,
                    action_index=action_index,
                )
                record(
                    "episode_failed",
                    actor="runner",
                    message=f"{failure.stage}: {failure.reason}",
                )
                return finish(
                    status="setup_failed",
                    handoff_action_index=len(state.action_log),
                    final_label="failed",
                    failure=failure,
                )

        handoff_action_index = len(state.action_log)
        handoff_checkpoint_id = checkpoint("handoff")

        last_result: ToolResult | None = None
        status = "max_steps"
        for step_index in range(spec.max_steps):
            context = EpisodeContext(
                episode_id=spec.episode_id,
                world_id=self.world.world_id,
                task_id=spec.task_id,
                workspace_brief=self.workspace_brief,
                user_message=spec.user_message,
                events=tuple(events),
                state=state.clone(),
                last_result=last_result,
            )
            try:
                agent_step = agent.next_step(context)
            except Exception as exc:
                failure = EpisodeFailure(
                    stage="agent",
                    reason=str(exc),
                    exception_type=type(exc).__name__,
                    event_sequence=len(events),
                )
                record(
                    "episode_failed",
                    actor="runner",
                    message=f"{failure.stage}: {failure.reason}",
                )
                return finish(
                    status="agent_failed",
                    handoff_action_index=handoff_action_index,
                    handoff_checkpoint_id=handoff_checkpoint_id,
                    final_label="failed",
                    failure=failure,
                )
            if not isinstance(agent_step, AgentStep):
                failure = EpisodeFailure(
                    stage="agent",
                    reason="agent returned a non-AgentStep value",
                    event_sequence=len(events),
                )
                record(
                    "episode_failed",
                    actor="runner",
                    message=f"{failure.stage}: {failure.reason}",
                )
                return finish(
                    status="agent_failed",
                    handoff_action_index=handoff_action_index,
                    handoff_checkpoint_id=handoff_checkpoint_id,
                    final_label="failed",
                    failure=failure,
                )

            if agent_step.call is None:
                if agent_step.message:
                    record(
                        "agent_message",
                        actor="agent",
                        message=agent_step.message,
                    )
            else:
                try:
                    last_result = runtime.execute(
                        agent_step.call,
                        actor="agent",
                        thought=agent_step.message,
                    )
                except Exception as exc:
                    failure = EpisodeFailure(
                        stage="runtime",
                        reason=str(exc),
                        exception_type=type(exc).__name__,
                        event_sequence=len(events),
                    )
                    record(
                        "episode_failed",
                        actor="runner",
                        message=f"{failure.stage}: {failure.reason}",
                    )
                    return finish(
                        status="runtime_failed",
                        handoff_action_index=handoff_action_index,
                        handoff_checkpoint_id=handoff_checkpoint_id,
                        final_label="failed",
                        failure=failure,
                    )
                action_index = state.action_log[-1].index
                record(
                    "agent_tool_call",
                    actor="agent",
                    message=agent_step.message,
                    call=agent_step.call,
                    result=last_result,
                    action_index=action_index,
                )
                checkpoint(f"after_agent_step_{step_index}")

            if agent_step.call is None and agent_step.done:
                status = "completed"
                record("episode_completed", actor="agent")
                break
        else:
            record(
                "episode_stopped",
                actor="runner",
                message="maximum step count reached",
            )

        return finish(
            status=status,
            handoff_action_index=handoff_action_index,
            handoff_checkpoint_id=handoff_checkpoint_id,
        )

    def resume(
        self,
        source: EpisodeResult,
        spec: ResumeSpec,
        agent: EpisodeAgent,
    ) -> EpisodeResult:
        """Continue a fresh agent from a saved checkpoint in an existing result."""

        if spec.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if spec.score_from not in {"checkpoint", "handoff"}:
            raise ValueError(f"unsupported score_from: {spec.score_from}")
        if source.world_id != self.world.world_id:
            raise ValueError(
                f"source world {source.world_id} does not match runner world "
                f"{self.world.world_id}"
            )

        source_checkpoint = source.checkpoint(spec.checkpoint_id)
        state = source_checkpoint.state.clone()
        runtime = OfficeRuntime(state)
        events = list(_events_through(source.events, source_checkpoint.sequence))
        checkpoints = list(
            _checkpoints_through(source.checkpoints, source_checkpoint.sequence)
        )
        user_message = spec.user_message or _source_user_message(source)
        score_start_action_index = (
            len(source_checkpoint.state.action_log)
            if spec.score_from == "checkpoint"
            else source.score_start_action_index
        )

        def record(
            event_type: str,
            *,
            actor: str,
            message: str = "",
            call: ToolCall | None = None,
            result: ToolResult | None = None,
            action_index: int | None = None,
            checkpoint_id: str = "",
        ) -> EventRecord:
            event = EventRecord(
                sequence=len(events),
                event_type=event_type,
                actor=actor,
                message=message,
                call=call,
                result=result,
                action_index=action_index,
                checkpoint_id=checkpoint_id,
            )
            events.append(event)
            return event

        def checkpoint(label: str) -> str:
            checkpoint_id = f"{spec.episode_id}:{label}:{len(checkpoints)}"
            event = record(
                "checkpoint_created",
                actor="runner",
                message=label,
                checkpoint_id=checkpoint_id,
            )
            checkpoints.append(
                CheckpointRecord(
                    checkpoint_id=checkpoint_id,
                    sequence=event.sequence,
                    label=label,
                    state=state.clone(),
                )
            )
            return checkpoint_id

        def finish(
            *,
            status: str,
            final_label: str = "final",
            failure: EpisodeFailure | None = None,
        ) -> EpisodeResult:
            final_checkpoint_id = checkpoint(final_label)
            outcome = self.scorer.score(
                initial_state=self.world.state,
                final_state=state,
                action_log=state.action_log[score_start_action_index:],
            )
            return EpisodeResult(
                episode_id=spec.episode_id,
                world_id=self.world.world_id,
                task_id=source.task_id,
                status=status,
                events=tuple(events),
                checkpoints=tuple(checkpoints),
                handoff_action_index=source.handoff_action_index,
                handoff_checkpoint_id=source.handoff_checkpoint_id,
                final_checkpoint_id=final_checkpoint_id,
                final_state=state.clone(),
                outcome=outcome,
                failure=failure,
                score_start_action_index=score_start_action_index,
                resume_source_episode_id=source.episode_id,
                resume_source_checkpoint_id=spec.checkpoint_id,
            )

        record(
            "episode_resumed",
            actor="runner",
            message=(
                f"resumed {source.episode_id} from checkpoint {spec.checkpoint_id}; "
                f"score_from={spec.score_from}"
            ),
            checkpoint_id=spec.checkpoint_id,
        )

        last_result = _last_agent_tool_result(events)
        status = "max_steps"
        for step_index in range(spec.max_steps):
            context = EpisodeContext(
                episode_id=spec.episode_id,
                world_id=self.world.world_id,
                task_id=source.task_id,
                workspace_brief=self.workspace_brief,
                user_message=user_message,
                events=tuple(events),
                state=state.clone(),
                last_result=last_result,
            )
            try:
                agent_step = agent.next_step(context)
            except Exception as exc:
                failure = EpisodeFailure(
                    stage="agent",
                    reason=str(exc),
                    exception_type=type(exc).__name__,
                    event_sequence=len(events),
                )
                record(
                    "episode_failed",
                    actor="runner",
                    message=f"{failure.stage}: {failure.reason}",
                )
                return finish(status="agent_failed", final_label="failed", failure=failure)
            if not isinstance(agent_step, AgentStep):
                failure = EpisodeFailure(
                    stage="agent",
                    reason="agent returned a non-AgentStep value",
                    event_sequence=len(events),
                )
                record(
                    "episode_failed",
                    actor="runner",
                    message=f"{failure.stage}: {failure.reason}",
                )
                return finish(status="agent_failed", final_label="failed", failure=failure)

            if agent_step.call is None:
                if agent_step.message:
                    record(
                        "agent_message",
                        actor="agent",
                        message=agent_step.message,
                    )
            else:
                try:
                    last_result = runtime.execute(
                        agent_step.call,
                        actor="agent",
                        thought=agent_step.message,
                    )
                except Exception as exc:
                    failure = EpisodeFailure(
                        stage="runtime",
                        reason=str(exc),
                        exception_type=type(exc).__name__,
                        event_sequence=len(events),
                    )
                    record(
                        "episode_failed",
                        actor="runner",
                        message=f"{failure.stage}: {failure.reason}",
                    )
                    return finish(
                        status="runtime_failed",
                        final_label="failed",
                        failure=failure,
                    )
                action_index = state.action_log[-1].index
                record(
                    "agent_tool_call",
                    actor="agent",
                    message=agent_step.message,
                    call=agent_step.call,
                    result=last_result,
                    action_index=action_index,
                )
                checkpoint(f"after_agent_step_{step_index}")

            if agent_step.call is None and agent_step.done:
                status = "completed"
                record("episode_completed", actor="agent")
                break
        else:
            record(
                "episode_stopped",
                actor="runner",
                message="maximum step count reached",
            )

        return finish(status=status)


def _events_through(
    events: tuple[EventRecord, ...],
    sequence: int,
) -> tuple[EventRecord, ...]:
    return tuple(event for event in events if event.sequence <= sequence)


def _checkpoints_through(
    checkpoints: tuple[CheckpointRecord, ...],
    sequence: int,
) -> tuple[CheckpointRecord, ...]:
    return tuple(checkpoint for checkpoint in checkpoints if checkpoint.sequence <= sequence)


def _source_user_message(source: EpisodeResult) -> str:
    for event in source.events:
        if event.event_type == "episode_started":
            return event.message
    return ""


def _last_agent_tool_result(events: list[EventRecord]) -> ToolResult | None:
    for event in reversed(events):
        if event.event_type == "agent_tool_call" and event.result is not None:
            return event.result
    return None
