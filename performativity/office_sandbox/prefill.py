"""Grounded prefill replay for office-productivity trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .runtime import OfficeRuntime

if TYPE_CHECKING:
    from .state import ActionRecord, OfficeState, ToolCall


@dataclass(frozen=True)
class PrefillStep:
    """One visible pre-handoff thought and grounded tool call."""

    thought: str
    call: ToolCall


@dataclass
class GroundedPrefill:
    """State and log after replaying a prefilled trajectory."""

    state: OfficeState
    action_log: list[ActionRecord]

    @property
    def handoff_index(self) -> int:
        return len(self.action_log)


def apply_prefill(
    initial_state: OfficeState,
    steps: list[PrefillStep],
    *,
    actor: str = "prefill",
) -> GroundedPrefill:
    """Replay prefill steps into a fresh office state.

    Raises:
        ValueError: if any prefilled action cannot execute against the sandbox.
    """

    state = initial_state.clone()
    runtime = OfficeRuntime(state)
    for step in steps:
        result = runtime.execute(step.call, actor=actor, thought=step.thought)
        if not result.ok:
            raise ValueError(
                f"prefill action failed: {step.call.name}({step.call.args}) -> "
                f"{result.error}"
            )
    return GroundedPrefill(state=state, action_log=list(state.action_log))


def render_action_log(action_log: list[ActionRecord]) -> str:
    """Render a visible thought/action/observation transcript."""

    lines: list[str] = []
    for record in action_log:
        if record.thought:
            lines.append(f"Thought: {record.thought}")
        lines.append(f"Action: {record.call.name}({record.call.args})")
        if record.result.ok:
            lines.append(f"Observation: {record.result.value}")
        else:
            lines.append(f"Observation: ERROR: {record.result.error}")
    return "\n".join(lines)
