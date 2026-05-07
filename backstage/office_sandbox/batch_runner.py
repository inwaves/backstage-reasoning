"""Batch execution harness for office-productivity episodes."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from backstage.office_sandbox.episode_runner import EpisodeRunner, EpisodeSpec

if TYPE_CHECKING:
    from backstage.office_sandbox.episode_runner import EpisodeAgent, EpisodeResult
    from backstage.office_sandbox.seeds import SeededOfficeWorld


AgentFactory = Callable[["BatchEpisode"], "EpisodeAgent"]


@dataclass(frozen=True)
class BatchRunSpec:
    """Experiment-level run specification."""

    run_id: str
    episodes: tuple[EpisodeSpec, ...]
    repeats: int = 1
    max_workers: int = 1
    output_dir: Path | str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchEpisode:
    """One expanded episode repeat inside a batch run."""

    run_id: str
    item_id: str
    sequence: int
    repeat_index: int
    base_episode_id: str
    episode_spec: EpisodeSpec


@dataclass(frozen=True)
class BatchFailure:
    """Batch-level failure details for one expanded episode."""

    stage: str
    reason: str
    exception_type: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-shaped representation for persistence."""

        return asdict(self)


@dataclass(frozen=True)
class BatchEpisodeResult:
    """The batch-level record for one expanded episode."""

    run_id: str
    item_id: str
    sequence: int
    repeat_index: int
    base_episode_id: str
    status: str
    episode_result: EpisodeResult | None = None
    failure: BatchFailure | None = None
    output_path: str = ""

    @property
    def is_risky(self) -> bool:
        return self.episode_result is not None and self.episode_result.outcome.is_risky

    @property
    def risk_families(self) -> frozenset[str]:
        if self.episode_result is None:
            return frozenset()
        return self.episode_result.outcome.risk_families

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-shaped representation for persistence."""

        return _json_ready(self)


@dataclass(frozen=True)
class BatchRunResult:
    """Results and aggregate status for a full batch run."""

    run_id: str
    results: tuple[BatchEpisodeResult, ...]
    output_dir: str = ""

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def status_counts(self) -> dict[str, int]:
        return dict(Counter(result.status for result in self.results))

    @property
    def risky_count(self) -> int:
        return sum(1 for result in self.results if result.is_risky)

    @property
    def risk_family_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for result in self.results:
            counts.update(result.risk_families)
        return dict(counts)

    @property
    def failure_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for result in self.results:
            if result.failure is not None:
                counts[result.failure.stage] += 1
            if (
                result.episode_result is not None
                and result.episode_result.failure is not None
            ):
                counts[result.episode_result.failure.stage] += 1
        return dict(counts)

    def summary_dict(self) -> dict[str, object]:
        """Return the compact status summary used by CLI/reporting layers."""

        return {
            "run_id": self.run_id,
            "total": self.total,
            "status_counts": self.status_counts,
            "risky_count": self.risky_count,
            "risk_family_counts": self.risk_family_counts,
            "failure_counts": self.failure_counts,
            "output_dir": self.output_dir,
        }

    def status_line(self) -> str:
        """Return a one-line human-readable run summary."""

        statuses = ", ".join(
            f"{status}={count}" for status, count in sorted(self.status_counts.items())
        )
        failures = sum(
            count
            for status, count in self.status_counts.items()
            if status not in {"completed", "max_steps"}
        )
        return (
            f"{self.run_id}: total={self.total}, risky={self.risky_count}, "
            f"failures={failures}, {statuses}"
        )


class BatchRunner:
    """Run many episode repeats and persist their traces independently."""

    def __init__(self, world: SeededOfficeWorld):
        self.world = world

    def run(self, spec: BatchRunSpec, agent_factory: AgentFactory) -> BatchRunResult:
        """Expand repeats, run each episode, and write optional result artifacts."""

        _validate_spec(spec)
        output_dir = Path(spec.output_dir) if spec.output_dir is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                output_dir / "run_spec.json",
                {
                    "run_id": spec.run_id,
                    "repeats": spec.repeats,
                    "max_workers": spec.max_workers,
                    "metadata": spec.metadata,
                    "episodes": [_json_ready(episode) for episode in spec.episodes],
                },
            )

        batch_episodes = _expand_batch(spec)
        if spec.max_workers == 1:
            results = [
                self._run_one(batch_episode, agent_factory, output_dir=output_dir)
                for batch_episode in batch_episodes
            ]
        else:
            results = self._run_parallel(
                batch_episodes,
                agent_factory,
                max_workers=spec.max_workers,
                output_dir=output_dir,
            )

        result = BatchRunResult(
            run_id=spec.run_id,
            results=tuple(sorted(results, key=lambda item: item.sequence)),
            output_dir=str(output_dir) if output_dir is not None else "",
        )
        if output_dir is not None:
            _write_json(output_dir / "summary.json", result.summary_dict())
            _write_json(
                output_dir / "manifest.json",
                {
                    **result.summary_dict(),
                    "items": [
                        {
                            "item_id": item.item_id,
                            "sequence": item.sequence,
                            "repeat_index": item.repeat_index,
                            "base_episode_id": item.base_episode_id,
                            "status": item.status,
                            "output_path": item.output_path,
                        }
                        for item in result.results
                    ],
                },
            )
        return result

    def _run_parallel(
        self,
        batch_episodes: tuple[BatchEpisode, ...],
        agent_factory: AgentFactory,
        *,
        max_workers: int,
        output_dir: Path | None,
    ) -> list[BatchEpisodeResult]:
        results: list[BatchEpisodeResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_one,
                    batch_episode,
                    agent_factory,
                    output_dir=output_dir,
                ): batch_episode
                for batch_episode in batch_episodes
            }
            for future in as_completed(futures):
                batch_episode = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        BatchEpisodeResult(
                            run_id=batch_episode.run_id,
                            item_id=batch_episode.item_id,
                            sequence=batch_episode.sequence,
                            repeat_index=batch_episode.repeat_index,
                            base_episode_id=batch_episode.base_episode_id,
                            status="worker_failed",
                            failure=BatchFailure(
                                stage="worker",
                                reason=str(exc),
                                exception_type=type(exc).__name__,
                            ),
                        )
                    )
        return results

    def _run_one(
        self,
        batch_episode: BatchEpisode,
        agent_factory: AgentFactory,
        *,
        output_dir: Path | None,
    ) -> BatchEpisodeResult:
        episode_result: EpisodeResult | None = None
        output_path = ""
        try:
            agent = agent_factory(batch_episode)
        except Exception as exc:
            result = BatchEpisodeResult(
                run_id=batch_episode.run_id,
                item_id=batch_episode.item_id,
                sequence=batch_episode.sequence,
                repeat_index=batch_episode.repeat_index,
                base_episode_id=batch_episode.base_episode_id,
                status="agent_factory_failed",
                failure=BatchFailure(
                    stage="agent_factory",
                    reason=str(exc),
                    exception_type=type(exc).__name__,
                ),
            )
            return _persist_or_mark(result, output_dir)

        try:
            episode_result = EpisodeRunner(self.world).run(
                batch_episode.episode_spec,
                agent,
            )
            result = BatchEpisodeResult(
                run_id=batch_episode.run_id,
                item_id=batch_episode.item_id,
                sequence=batch_episode.sequence,
                repeat_index=batch_episode.repeat_index,
                base_episode_id=batch_episode.base_episode_id,
                status=episode_result.status,
                episode_result=episode_result,
            )
            return _persist_or_mark(result, output_dir)
        except Exception as exc:
            if output_dir is not None:
                output_path = str(
                    _episode_output_path(output_dir, batch_episode.item_id)
                )
            result = BatchEpisodeResult(
                run_id=batch_episode.run_id,
                item_id=batch_episode.item_id,
                sequence=batch_episode.sequence,
                repeat_index=batch_episode.repeat_index,
                base_episode_id=batch_episode.base_episode_id,
                status="runner_failed",
                episode_result=episode_result,
                failure=BatchFailure(
                    stage="runner",
                    reason=str(exc),
                    exception_type=type(exc).__name__,
                ),
                output_path=output_path,
            )
            return _persist_or_mark(result, output_dir)


def _expand_batch(spec: BatchRunSpec) -> tuple[BatchEpisode, ...]:
    batch_episodes: list[BatchEpisode] = []
    sequence = 0
    for repeat_index in range(spec.repeats):
        for episode in spec.episodes:
            item_id = _item_id(sequence, episode.episode_id, repeat_index)
            batch_episodes.append(
                BatchEpisode(
                    run_id=spec.run_id,
                    item_id=item_id,
                    sequence=sequence,
                    repeat_index=repeat_index,
                    base_episode_id=episode.episode_id,
                    episode_spec=replace(episode, episode_id=item_id),
                )
            )
            sequence += 1
    return tuple(batch_episodes)


def _persist_batch_item(
    result: BatchEpisodeResult,
    output_dir: Path | None,
) -> BatchEpisodeResult:
    if output_dir is None:
        return result
    path = _episode_output_path(output_dir, result.item_id)
    persisted = replace(result, output_path=str(path))
    payload = persisted.to_dict()
    payload["risk_families"] = sorted(result.risk_families)
    payload["is_risky"] = result.is_risky
    _write_json(path, payload)
    return persisted


def _persist_or_mark(
    result: BatchEpisodeResult,
    output_dir: Path | None,
) -> BatchEpisodeResult:
    try:
        return _persist_batch_item(result, output_dir)
    except Exception as exc:
        return replace(
            result,
            status="persistence_failed",
            failure=BatchFailure(
                stage="persistence",
                reason=str(exc),
                exception_type=type(exc).__name__,
            ),
        )


def _episode_output_path(output_dir: Path, item_id: str) -> Path:
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    return episode_dir / f"{_slug(item_id)}.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _validate_spec(spec: BatchRunSpec) -> None:
    if not spec.run_id:
        raise ValueError("run_id must be non-empty")
    if not spec.episodes:
        raise ValueError("episodes must be non-empty")
    if spec.repeats < 1:
        raise ValueError("repeats must be at least 1")
    if spec.max_workers < 1:
        raise ValueError("max_workers must be at least 1")


def _item_id(sequence: int, episode_id: str, repeat_index: int) -> str:
    return f"{sequence:04d}_{episode_id}_r{repeat_index:03d}"


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )


def _json_ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, frozenset | set):
        return sorted(_json_ready(item) for item in value)
    return value
