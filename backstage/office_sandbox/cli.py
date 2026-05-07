"""Hydra CLI for office-productivity batch runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from backstage.office_sandbox.agent import Agent, AgentConfig
from backstage.office_sandbox.batch_runner import (
    BatchEpisode,
    BatchRunner,
    BatchRunResult,
    BatchRunSpec,
)
from backstage.office_sandbox.episode_runner import (
    AgentStep,
    EpisodeContext,
    EpisodeSpec,
)
from backstage.office_sandbox.prefill import PrefillStep
from backstage.office_sandbox.seeds import (
    OfficeTaskSeed,
    SeededOfficeWorld,
    make_office_productivity_world,
)
from backstage.office_sandbox.state import ToolCall
from backstage.office_sandbox.vllm_backend import VLLMBackend, VLLMBackendConfig


def main(argv: list[str] | None = None) -> int:
    """Run an office-productivity batch from Hydra overrides."""

    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in args):
        print(_help_text())
        return 0
    config_dir = str(Path(__file__).with_name("conf").resolve())
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = hydra.compose(config_name="run_office_batch", overrides=args)
    result = run_from_config(cfg)
    print(result.status_line())
    return 0


def run_from_config(cfg: DictConfig | dict[str, Any]) -> BatchRunResult:
    """Build and run a batch from a Hydra/OmegaConf-compatible config."""

    config = _to_plain_dict(cfg)
    world = _make_world(_mapping(config, "world", default={"name": "office_productivity"}))
    run_config = _mapping(config, "run")
    agent_config = _mapping(config, "agent")
    episodes = tuple(
        _episode_spec(item, world=world)
        for item in _list(config, "episodes", required=True)
    )

    spec = BatchRunSpec(
        run_id=_string(run_config, "run_id", required=True),
        episodes=episodes,
        repeats=_integer(run_config, "repeats", default=1),
        max_workers=_integer(run_config, "max_workers", default=1),
        output_dir=_optional_path(run_config.get("output_dir")),
        metadata=_mapping(run_config, "metadata", default={}),
    )
    result = BatchRunner(world).run(spec, _agent_factory(agent_config))
    return result


class ScriptedAgent:
    """Small deterministic agent used for smoke runs and CLI tests."""

    def __init__(self, steps: tuple[AgentStep, ...]) -> None:
        self.steps = list(steps)

    def next_step(self, context: EpisodeContext) -> AgentStep:
        if not self.steps:
            return AgentStep(message="Done.", done=True)
        return self.steps.pop(0)


def _agent_factory(config: dict[str, Any]):
    kind = _string(config, "kind", default="scripted")
    if kind == "scripted":
        steps = _scripted_steps(_mapping(config, "scripted", default={}))

        def make_scripted(batch_episode: BatchEpisode) -> ScriptedAgent:
            return ScriptedAgent(steps)

        return make_scripted

    if kind == "vllm":

        def make_vllm(batch_episode: BatchEpisode) -> Agent:
            return Agent(
                backend=VLLMBackend(
                    backend_config=VLLMBackendConfig(
                        **_vllm_backend_kwargs(_mapping(config, "vllm", default={}))
                    )
                ),
                config=AgentConfig(
                    model=_string(config, "model", required=True),
                    temperature=_float(config, "temperature", default=0.0),
                    max_tokens=_integer(config, "max_tokens", default=1024),
                    seed=_optional_int(config.get("seed")),
                    handoff_view=_string(config, "handoff_view", default="activity_context"),
                    metadata=_mapping(config, "metadata", default={}),
                ),
            )

        return make_vllm

    raise ValueError(f"unsupported agent kind: {kind}")


def _make_world(config: dict[str, Any]) -> SeededOfficeWorld:
    name = _string(config, "name", default="office_productivity")
    if name == "office_productivity":
        return make_office_productivity_world()
    raise ValueError(f"unsupported world: {name}")


def _episode_spec(config: dict[str, Any], *, world: SeededOfficeWorld) -> EpisodeSpec:
    task_id = _string(config, "task_id", required=True)
    task = _find_task(world, task_id)
    return EpisodeSpec(
        episode_id=_string(config, "episode_id", default=task_id),
        task_id=task_id,
        user_message=_string(config, "user_message", default=task.user_message),
        preconditioning_steps=tuple(
            _prefill_step(item)
            for item in _list(config, "preconditioning_steps", default=[])
        ),
        max_steps=_integer(config, "max_steps", default=12),
    )


def _find_task(world: SeededOfficeWorld, task_id: str) -> OfficeTaskSeed:
    for task in world.tasks:
        if task.task_id == task_id:
            return task
    raise ValueError(f"unknown task_id for {world.world_id}: {task_id}")


def _prefill_step(config: dict[str, Any]) -> PrefillStep:
    return PrefillStep(
        thought=_string(config, "thought", default=""),
        call=_tool_call(_mapping(config, "call")),
    )


def _scripted_steps(config: dict[str, Any]) -> tuple[AgentStep, ...]:
    step_configs = _list(
        config,
        "steps",
        default=[{"message": "Done.", "done": True}],
    )
    return tuple(_agent_step(item) for item in step_configs)


def _agent_step(config: dict[str, Any]) -> AgentStep:
    call_config = config.get("call")
    call = _tool_call(call_config) if call_config is not None else None
    return AgentStep(
        message=_string(config, "message", default=""),
        call=call,
        done=bool(config.get("done", call is None)),
    )


def _tool_call(config: object) -> ToolCall:
    if not isinstance(config, dict):
        raise TypeError("tool call config must be a mapping")
    return ToolCall(
        name=_string(config, "name", required=True),
        args=_mapping(config, "args", default={}),
    )


def _vllm_backend_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "base_url": _string(config, "base_url", default="http://127.0.0.1:8000/v1"),
        "api_key": _string(config, "api_key", default="EMPTY"),
        "timeout": _float(config, "timeout", default=120.0),
        "tool_mode": _string(config, "tool_mode", default="native"),
        "tool_choice": config.get("tool_choice", "auto"),
        "extra_body": _mapping(config, "extra_body", default={}),
    }
    return kwargs


def _to_plain_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        value = OmegaConf.to_container(cfg, resolve=True)
    else:
        value = cfg
    if not isinstance(value, dict):
        raise TypeError("config root must be a mapping")
    return value


def _mapping(
    config: dict[str, Any],
    key: str,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = config.get(key, default)
    if value is None:
        raise KeyError(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def _list(
    config: dict[str, Any],
    key: str,
    *,
    default: list[dict[str, Any]] | None = None,
    required: bool = False,
) -> list[dict[str, Any]]:
    if key not in config:
        if required:
            raise KeyError(key)
        return list(default or [])
    value = config[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise TypeError(f"{key} entries must be mappings")
    return value


def _string(
    config: dict[str, Any],
    key: str,
    *,
    default: str = "",
    required: bool = False,
) -> str:
    if key not in config:
        if required:
            raise KeyError(key)
        return default
    value = config[key]
    if value is None:
        if required:
            raise KeyError(key)
        return default
    return str(value)


def _integer(config: dict[str, Any], key: str, *, default: int) -> int:
    value = config.get(key, default)
    if value is None:
        return default
    return int(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _float(config: dict[str, Any], key: str, *, default: float) -> float:
    value = config.get(key, default)
    if value is None:
        return default
    return float(value)


def _optional_path(value: object) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


def _help_text() -> str:
    return (
        "Usage: backstage-office-batch [Hydra overrides]\n\n"
        "Examples:\n"
        "  backstage-office-batch\n"
        "  backstage-office-batch run.run_id=my-run run.output_dir=results/my-run\n"
        "  backstage-office-batch agent.kind=vllm "
        "agent.model=HuggingFaceTB/SmolLM2-135M-Instruct "
        "agent.vllm.base_url=http://127.0.0.1:8000/v1\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
