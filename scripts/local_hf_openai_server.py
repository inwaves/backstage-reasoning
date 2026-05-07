"""Tiny local OpenAI-compatible chat server backed by Hugging Face models.

This is a local smoke-test helper for the backend adapter. It is not intended
to replace vLLM on the GPU box; it only exercises the same /v1/chat/completions
surface with a real Hugging Face model on this machine.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    args = _parse_args()
    runner = LocalHFRunner(
        model_id=args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    handler = _make_handler(runner)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        json.dumps(
            {
                "ok": True,
                "server": f"http://{args.host}:{args.port}/v1",
                "model": args.model,
                "device": runner.device,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()
    return 0


class LocalHFRunner:
    """Small synchronous text generator used by the local smoke server."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        torch_dtype: str,
    ) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = _torch_dtype(torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.device = _resolve_device(device)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = self._render_prompt(payload.get("messages", []))
        temperature = float(payload.get("temperature", 0.0) or 0.0)
        max_new_tokens = int(payload.get("max_tokens", 64) or 64)
        seed = payload.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        with torch.inference_mode():
            output = self.model.generate(**inputs, **generate_kwargs)

        input_length = int(inputs["input_ids"].shape[-1])
        generated_ids = output[0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return _chat_response(
            model_id=str(payload.get("model") or self.model_id),
            content=text,
        )

    def _render_prompt(self, messages: list[dict[str, Any]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered = ""
            if rendered:
                return str(rendered)
        lines = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)


def _make_handler(runner: LocalHFRunner) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": runner.model_id,
                                "object": "model",
                            }
                        ],
                    }
                )
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json({"error": "not found"}, status=404)
                return
            try:
                payload = self._read_json()
                response = runner.complete(payload)
            except Exception as exc:
                self._send_json(
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    status=500,
                )
                return
            self._send_json(response)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _chat_response(*, model_id: str, content: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-local-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _torch_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "auto":
        return None
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
