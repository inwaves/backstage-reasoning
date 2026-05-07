from __future__ import annotations

from dataclasses import dataclass

import pytest

from backstage.office_sandbox import (
    Agent,
    AgentConfig,
    ChatMessage,
    EpisodeRunner,
    EpisodeSpec,
    ModelResponse,
    PrefillStep,
    ToolCall,
    ToolSchema,
    make_office_productivity_world,
)


@dataclass
class BackendCall:
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolSchema, ...]
    config: AgentConfig


class RecordingBackend:
    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.calls: list[BackendCall] = []

    def complete(
        self,
        *,
        messages: tuple[ChatMessage, ...],
        tools: tuple[ToolSchema, ...],
        config: AgentConfig,
    ) -> ModelResponse:
        self.calls.append(BackendCall(messages=messages, tools=tools, config=config))
        if not self.responses:
            return ModelResponse(message="Done.")
        return self.responses.pop(0)


def _message_text(messages: tuple[ChatMessage, ...]) -> str:
    return "\n\n".join(message.content for message in messages)


def test_agent_state_only_handoff_omits_preconditioning_activity() -> None:
    world = make_office_productivity_world()
    task = world.tasks[0]
    backend = RecordingBackend([ModelResponse(message="Done.")])
    agent = Agent(
        backend=backend,
        config=AgentConfig(model="test-model", handoff_view="state_only"),
    )
    spec = EpisodeSpec(
        episode_id="state-only",
        task_id=task.task_id,
        user_message=task.user_message,
        preconditioning_steps=(
            PrefillStep(
                thought="I should inspect the fast-track message.",
                call=ToolCall(name="read_email", args={"email_id": "email-004"}),
            ),
        ),
        max_steps=1,
    )

    result = EpisodeRunner(world).run(spec, agent)
    text = _message_text(backend.calls[0].messages)

    assert result.status == "completed"
    assert "Recent workspace activity" not in text
    assert "email-004" not in text


def test_agent_activity_context_handoff_includes_grounded_activity() -> None:
    world = make_office_productivity_world()
    task = world.tasks[0]
    backend = RecordingBackend([ModelResponse(message="Done.")])
    agent = Agent(
        backend=backend,
        config=AgentConfig(model="test-model", handoff_view="activity_context"),
    )
    spec = EpisodeSpec(
        episode_id="activity-context",
        task_id=task.task_id,
        user_message=task.user_message,
        preconditioning_steps=(
            PrefillStep(
                thought="I should inspect the fast-track message.",
                call=ToolCall(name="read_email", args={"email_id": "email-004"}),
            ),
        ),
        max_steps=1,
    )

    result = EpisodeRunner(world).run(spec, agent)
    text = _message_text(backend.calls[0].messages)

    assert result.status == "completed"
    assert "Recent workspace activity" in text
    assert "read_email" in text
    assert "email-004" in text
    assert "I should inspect the fast-track message" not in text


def test_agent_feeds_failed_tool_result_back_to_backend_messages() -> None:
    world = make_office_productivity_world()
    task = world.tasks[0]
    backend = RecordingBackend(
        [
            ModelResponse(
                message="I will inspect the file.",
                tool_call=ToolCall(
                    name="read_file",
                    args={"path": "missing.txt"},
                ),
            ),
            ModelResponse(message="That path failed; done."),
        ]
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(model="test-model", handoff_view="state_only"),
    )
    spec = EpisodeSpec(
        episode_id="tool-error-feedback",
        task_id=task.task_id,
        user_message=task.user_message,
        max_steps=3,
    )

    result = EpisodeRunner(world).run(spec, agent)
    second_call_text = _message_text(backend.calls[1].messages)

    assert result.status == "completed"
    assert len(backend.calls) == 2
    assert "ERROR" in second_call_text
    assert "file not found: missing.txt" in second_call_text
    assert any(tool.name == "send_email" for tool in backend.calls[0].tools)


def test_agent_rejects_unsupported_handoff_view() -> None:
    with pytest.raises(ValueError, match="unsupported handoff view"):
        Agent(
            backend=RecordingBackend([]),
            config=AgentConfig(
                model="test-model",
                handoff_view="assistant_history",  # type: ignore[arg-type]
            ),
        )
