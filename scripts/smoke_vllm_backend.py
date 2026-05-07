"""Smoke-test the office Agent backend against an OpenAI-compatible server.

This script expects a vLLM or local smoke server to already be running, e.g.

    vllm serve HuggingFaceTB/SmolLM2-135M-Instruct --port 8000

The experiment process only needs the OpenAI-compatible client.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from backstage.office_sandbox import (
    AgentConfig,
    ChatMessage,
    ToolCall,
    ToolSchema,
    VLLMBackend,
    VLLMBackendConfig,
)


def main() -> int:
    args = _parse_args()
    backend = VLLMBackend(
        backend_config=VLLMBackendConfig(
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            tool_mode=args.tool_mode,
            extra_body=_json_arg(args.extra_body),
        )
    )

    if args.tool_smoke:
        tools = (_read_file_tool(),)
        prompt = (
            "Return exactly this JSON object and no other text:\n"
            '{"message":"reading file",'
            '"tool_call":{"name":"read_file",'
            '"args":{"path":"ops/week-ahead.md"}}}'
        )
    else:
        tools = ()
        prompt = args.prompt

    try:
        response = backend.complete(
            messages=(
                ChatMessage(
                    role="system",
                    content="You are a concise assistant for a backend smoke test.",
                ),
                ChatMessage(role="user", content=prompt),
            ),
            tools=tools,
            config=AgentConfig(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed,
            ),
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "model": args.model,
                    "base_url": args.base_url,
                    "tool_mode": args.tool_mode,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    payload = {
        "ok": True,
        "model": args.model,
        "base_url": args.base_url,
        "tool_mode": args.tool_mode,
        "message": response.message,
        "tool_call": asdict(response.tool_call) if response.tool_call else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.tool_smoke and response.tool_call is None:
        print("Expected a tool call, but the model returned plain text.", file=sys.stderr)
        return 2
    if args.tool_smoke and response.tool_call != _expected_tool_call():
        print("Tool call did not match the expected smoke call.", file=sys.stderr)
        return 3
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible vLLM server URL.",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Model id served by vLLM.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--tool-mode",
        choices=("native", "json"),
        default="json",
        help="Use JSON mode for small instruct models; use native when vLLM/model tool calling is configured.",
    )
    parser.add_argument(
        "--extra-body",
        default="{}",
        help="JSON object passed through to the vLLM request as extra_body.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: pong",
        help="Plain text smoke prompt used when --tool-smoke is not set.",
    )
    parser.add_argument(
        "--tool-smoke",
        action="store_true",
        help="Ask the model to return one read_file tool call and fail if it does not.",
    )
    return parser.parse_args()


def _read_file_tool() -> ToolSchema:
    return ToolSchema(
        name="read_file",
        description="Read one workspace file by path.",
        parameters={
            "type": "object",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        },
    )


def _expected_tool_call() -> ToolCall:
    return ToolCall(
        name="read_file",
        args={"path": "ops/week-ahead.md"},
    )


def _json_arg(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--extra-body must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--extra-body must decode to a JSON object")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
