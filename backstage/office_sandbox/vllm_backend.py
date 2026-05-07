"""vLLM-backed chat backend for office-productivity agents."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from backstage.office_sandbox.agent import (
    AgentConfig,
    ChatMessage,
    ModelResponse,
    ToolSchema,
)
from envs.office.state import ToolCall

ToolMode = Literal["native", "json"]


@dataclass(frozen=True)
class VLLMBackendConfig:
    """Connection and request settings for a vLLM OpenAI-compatible server."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    timeout: float = 120.0
    tool_mode: ToolMode = "native"
    tool_choice: str | dict[str, object] = "auto"
    extra_body: dict[str, object] = field(default_factory=dict)


class VLLMBackendError(RuntimeError):
    """Raised when a vLLM response cannot be converted into an agent step."""


class VLLMBackend:
    """Chat backend that talks to a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        *,
        backend_config: VLLMBackendConfig | None = None,
        client: object | None = None,
    ) -> None:
        self.backend_config = backend_config or VLLMBackendConfig()
        _validate_tool_mode(self.backend_config.tool_mode)
        self.client = client if client is not None else self._make_client()

    def complete(
        self,
        *,
        messages: tuple[ChatMessage, ...],
        tools: tuple[ToolSchema, ...],
        config: AgentConfig,
    ) -> ModelResponse:
        """Return one model message or one model-requested tool call."""

        request_messages = list(_to_openai_messages(messages))
        request_tools: list[dict[str, object]] = []
        if self.backend_config.tool_mode == "native":
            request_tools = [_to_openai_tool(tool) for tool in tools]
        elif tools:
            _insert_json_tool_protocol(request_messages, tools)

        request: dict[str, object] = {
            "model": config.model,
            "messages": request_messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.seed is not None:
            request["seed"] = config.seed
        if request_tools:
            request["tools"] = request_tools
            request["tool_choice"] = self.backend_config.tool_choice

        extra_body = _merge_extra_body(
            self.backend_config.extra_body,
            config.metadata.get("extra_body"),
            config.metadata.get("vllm_extra_body"),
        )
        if extra_body:
            request["extra_body"] = extra_body

        response = self.client.chat.completions.create(**request)
        return _parse_response(response, valid_tool_names={tool.name for tool in tools})

    def _make_client(self) -> object:
        from openai import OpenAI

        return OpenAI(
            base_url=self.backend_config.base_url,
            api_key=self.backend_config.api_key,
            timeout=self.backend_config.timeout,
        )


def _to_openai_messages(messages: tuple[ChatMessage, ...]) -> tuple[dict[str, object], ...]:
    converted: list[dict[str, object]] = []
    for message in messages:
        if message.role in {"system", "user", "assistant"}:
            converted.append({"role": message.role, "content": message.content})
            continue
        if message.role == "tool":
            name = message.name or "tool"
            converted.append(
                {
                    "role": "user",
                    "content": f"Tool result from {name}:\n{message.content}",
                }
            )
            continue
        converted.append(
            {
                "role": "user",
                "content": f"{message.role} message:\n{message.content}",
            }
        )
    return tuple(converted)


def _to_openai_tool(tool: ToolSchema) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _insert_json_tool_protocol(
    messages: list[dict[str, object]],
    tools: tuple[ToolSchema, ...],
) -> None:
    protocol = _json_tool_protocol_message(tools)
    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    messages.insert(insert_at, protocol)


def _json_tool_protocol_message(tools: tuple[ToolSchema, ...]) -> dict[str, object]:
    tool_lines = []
    for tool in tools:
        tool_lines.append(
            f"- {tool.name}: {tool.description}\n"
            f"  parameters: {json.dumps(tool.parameters, sort_keys=True)}"
        )
    return {
        "role": "system",
        "content": (
            "Tool call protocol: when you need to use a tool, respond with only "
            "one JSON object of the form "
            '{"message":"short note","tool_call":{"name":"tool_name","args":{}}}. '
            "The args object must contain concrete argument values for this call; "
            "do not copy the parameter schema into args. When no tool is needed, "
            "respond normally.\n\n"
            "Available tools:\n"
            + "\n".join(tool_lines)
        ),
    }


def _parse_response(response: object, *, valid_tool_names: set[str]) -> ModelResponse:
    message = _first_message(response)
    content = _get(message, "content", "") or ""
    raw = _json_ready(response)

    tool_calls = _get(message, "tool_calls", None) or []
    if tool_calls:
        return ModelResponse(
            message=str(content),
            tool_call=_parse_native_tool_call(tool_calls[0]),
            raw=raw,
        )

    parsed_call = _parse_json_tool_call(str(content), valid_tool_names=valid_tool_names)
    if parsed_call is not None:
        return ModelResponse(
            message=parsed_call.message,
            tool_call=parsed_call.tool_call,
            raw=raw,
        )

    return ModelResponse(message=str(content), raw=raw)


@dataclass(frozen=True)
class _ParsedToolCall:
    message: str
    tool_call: ToolCall


def _first_message(response: object) -> object:
    choices = _get(response, "choices", None)
    if not choices:
        raise VLLMBackendError("vLLM response did not contain choices")
    return _get(choices[0], "message", {})


def _parse_native_tool_call(tool_call: object) -> ToolCall:
    function = _get(tool_call, "function", None)
    if function is None:
        raise VLLMBackendError("vLLM tool call did not contain a function")
    name = _get(function, "name", "")
    if not name:
        raise VLLMBackendError("vLLM tool call did not contain a function name")
    return ToolCall(
        name=str(name),
        args=_decode_args(_get(function, "arguments", {})),
    )


def _parse_json_tool_call(
    content: str,
    *,
    valid_tool_names: set[str],
) -> _ParsedToolCall | None:
    payload = _extract_json_object(content)
    if not isinstance(payload, Mapping):
        return None

    explicit_tool_call = "tool_call" in payload
    call_payload = payload.get("tool_call")
    if call_payload is None and "action" in payload:
        explicit_tool_call = True
        call_payload = payload
    if call_payload is None and "tool" in payload:
        explicit_tool_call = True
        call_payload = payload
    if call_payload is None and "name" in payload and "arguments" in payload:
        call_payload = payload
    if not isinstance(call_payload, Mapping):
        return None

    name = call_payload.get("name") or call_payload.get("tool") or call_payload.get(
        "action"
    )
    if not name:
        return None
    name = str(name)
    if valid_tool_names and name not in valid_tool_names and not explicit_tool_call:
        return None

    args = (
        call_payload.get("args")
        if "args" in call_payload
        else call_payload.get("arguments", {})
    )
    return _ParsedToolCall(
        message=str(payload.get("message", "")),
        tool_call=ToolCall(name=name, args=_decode_args(args)),
    )


def _extract_json_object(content: str) -> object | None:
    decoder = json.JSONDecoder()
    stripped = _strip_json_fence(content.strip())
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        return parsed
    return None


def _strip_json_fence(content: str) -> str:
    if not content.startswith("```"):
        return content
    lines = content.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content


def _decode_args(value: object) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise VLLMBackendError(f"tool arguments were not valid JSON: {exc}") from exc
        if isinstance(loaded, Mapping):
            return dict(loaded)
        raise VLLMBackendError("tool arguments JSON must decode to an object")
    if isinstance(value, Mapping):
        return dict(value)
    raise VLLMBackendError("tool arguments must be an object or JSON object string")


def _merge_extra_body(*values: object) -> dict[str, object]:
    merged: dict[str, object] = {}
    for value in values:
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise VLLMBackendError("extra_body metadata must be an object")
        merged.update(dict(value))
    return merged


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _json_ready(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[no-any-return, attr-defined]
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return, attr-defined]
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)  # type: ignore[arg-type]
    return value


def _validate_tool_mode(tool_mode: str) -> None:
    if tool_mode not in {"native", "json"}:
        raise ValueError(f"unsupported vLLM tool mode: {tool_mode}")
