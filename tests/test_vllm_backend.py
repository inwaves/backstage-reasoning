from __future__ import annotations

from dataclasses import dataclass

import pytest

from backstage.office_sandbox import (
    AgentConfig,
    ChatMessage,
    ToolSchema,
    VLLMBackend,
    VLLMBackendConfig,
    VLLMBackendError,
)


@dataclass
class FakeCompletions:
    response: object
    calls: list[dict[str, object]]

    def create(self, **request: object) -> object:
        self.calls.append(request)
        return self.response


class FakeClient:
    def __init__(self, response: object):
        completions = FakeCompletions(response=response, calls=[])
        self.chat = type("FakeChat", (), {"completions": completions})()
        self.completions = completions


def _read_file_tool() -> ToolSchema:
    return ToolSchema(
        name="read_file",
        description="Read one file.",
        parameters={
            "type": "object",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
        },
    )


def test_vllm_backend_sends_native_tool_request_and_parses_tool_call() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": "I will inspect the file.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"travel/passport.png"}',
                            }
                        }
                    ],
                }
            }
        ]
    }
    client = FakeClient(response)
    backend = VLLMBackend(
        backend_config=VLLMBackendConfig(
            base_url="http://gpu-box:8000/v1",
            tool_mode="native",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        ),
        client=client,
    )

    result = backend.complete(
        messages=(
            ChatMessage(role="system", content="system"),
            ChatMessage(role="tool", name="read_file", content="OK: previous"),
        ),
        tools=(_read_file_tool(),),
        config=AgentConfig(
            model="Qwen/Qwen3-32B",
            temperature=0.2,
            max_tokens=64,
            seed=11,
        ),
    )

    request = client.completions.calls[0]

    assert request["model"] == "Qwen/Qwen3-32B"
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 64
    assert request["seed"] == 11
    assert request["tool_choice"] == "auto"
    assert request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read one file.",
                "parameters": _read_file_tool().parameters,
            },
        }
    ]
    assert request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    assert request["messages"][-1] == {
        "role": "user",
        "content": "Tool result from read_file:\nOK: previous",
    }
    assert result.message == "I will inspect the file."
    assert result.tool_call is not None
    assert result.tool_call.name == "read_file"
    assert result.tool_call.args == {"path": "travel/passport.png"}


def test_vllm_backend_json_tool_mode_uses_prompt_protocol() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"message":"Looking it up.",'
                        '"tool_call":{"name":"read_file",'
                        '"args":{"path":"notes.txt"}}}'
                    )
                }
            }
        ]
    }
    client = FakeClient(response)
    backend = VLLMBackend(
        backend_config=VLLMBackendConfig(tool_mode="json"),
        client=client,
    )

    result = backend.complete(
        messages=(ChatMessage(role="user", content="Please inspect notes."),),
        tools=(_read_file_tool(),),
        config=AgentConfig(model="test-model"),
    )
    request = client.completions.calls[0]

    assert "tools" not in request
    assert "Tool call protocol" in request["messages"][0]["content"]
    assert "read_file" in request["messages"][0]["content"]
    assert request["messages"][1] == {
        "role": "user",
        "content": "Please inspect notes.",
    }
    assert result.message == "Looking it up."
    assert result.tool_call is not None
    assert result.tool_call.name == "read_file"
    assert result.tool_call.args == {"path": "notes.txt"}


def test_vllm_backend_plain_text_response_finishes_without_tool_call() -> None:
    client = FakeClient({"choices": [{"message": {"content": "Done."}}]})
    backend = VLLMBackend(client=client)

    result = backend.complete(
        messages=(ChatMessage(role="user", content="Hello"),),
        tools=(_read_file_tool(),),
        config=AgentConfig(model="test-model"),
    )

    assert result.message == "Done."
    assert result.tool_call is None


def test_vllm_backend_json_mode_does_not_add_tool_protocol_without_tools() -> None:
    client = FakeClient({"choices": [{"message": {"content": "Hello."}}]})
    backend = VLLMBackend(
        backend_config=VLLMBackendConfig(tool_mode="json"),
        client=client,
    )

    result = backend.complete(
        messages=(ChatMessage(role="user", content="Say hello."),),
        tools=(),
        config=AgentConfig(model="test-model"),
    )
    request = client.completions.calls[0]

    assert result.message == "Hello."
    assert request["messages"] == [{"role": "user", "content": "Say hello."}]


def test_vllm_backend_returns_explicit_unknown_json_tool_to_runtime() -> None:
    client = FakeClient(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"message":"Trying this.",'
                            '"tool_call":{"name":"open_file",'
                            '"args":{"path":"notes.txt"}}}'
                        )
                    }
                }
            ]
        }
    )
    backend = VLLMBackend(
        backend_config=VLLMBackendConfig(tool_mode="json"),
        client=client,
    )

    result = backend.complete(
        messages=(ChatMessage(role="user", content="Please inspect notes."),),
        tools=(_read_file_tool(),),
        config=AgentConfig(model="test-model"),
    )

    assert result.tool_call is not None
    assert result.tool_call.name == "open_file"
    assert result.tool_call.args == {"path": "notes.txt"}


def test_vllm_backend_parses_fenced_json_action_shape() -> None:
    client = FakeClient(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            '{"message":"Reading.",'
                            '"action":"read_file",'
                            '"arguments":{"path":"notes.txt"}}'
                            "\n```"
                        )
                    }
                }
            ]
        }
    )
    backend = VLLMBackend(
        backend_config=VLLMBackendConfig(tool_mode="json"),
        client=client,
    )

    result = backend.complete(
        messages=(ChatMessage(role="user", content="Please inspect notes."),),
        tools=(_read_file_tool(),),
        config=AgentConfig(model="test-model"),
    )

    assert result.message == "Reading."
    assert result.tool_call is not None
    assert result.tool_call.name == "read_file"
    assert result.tool_call.args == {"path": "notes.txt"}


def test_vllm_backend_rejects_bad_config_and_responses() -> None:
    with pytest.raises(ValueError, match="unsupported vLLM tool mode"):
        VLLMBackend(
            backend_config=VLLMBackendConfig(tool_mode="xml"),  # type: ignore[arg-type]
            client=FakeClient({}),
        )

    backend = VLLMBackend(client=FakeClient({"choices": []}))

    with pytest.raises(VLLMBackendError, match="choices"):
        backend.complete(
            messages=(ChatMessage(role="user", content="Hello"),),
            tools=(_read_file_tool(),),
            config=AgentConfig(model="test-model"),
        )


def test_vllm_backend_rejects_non_mapping_extra_body_metadata() -> None:
    backend = VLLMBackend(client=FakeClient({"choices": [{"message": {"content": ""}}]}))

    with pytest.raises(VLLMBackendError, match="extra_body"):
        backend.complete(
            messages=(ChatMessage(role="user", content="Hello"),),
            tools=(),
            config=AgentConfig(
                model="test-model",
                metadata={"extra_body": "not-a-mapping"},
            ),
        )


def test_vllm_backend_bad_native_arguments_are_harness_failure() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": "{not json",
                            }
                        }
                    ]
                }
            }
        ]
    }
    backend = VLLMBackend(client=FakeClient(response))

    with pytest.raises(VLLMBackendError, match="tool arguments"):
        backend.complete(
            messages=(ChatMessage(role="user", content="Hello"),),
            tools=(_read_file_tool(),),
            config=AgentConfig(model="test-model"),
        )
