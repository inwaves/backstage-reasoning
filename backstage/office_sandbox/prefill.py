"""Compatibility re-exports for shared office prefill replay."""

from envs.office.prefill import (
    GroundedPrefill,
    PrefillStep,
    apply_prefill,
    render_action_log,
)

__all__ = [
    "GroundedPrefill",
    "PrefillStep",
    "apply_prefill",
    "render_action_log",
]
