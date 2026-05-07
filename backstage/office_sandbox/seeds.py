"""Compatibility re-exports for shared office seeds."""

from envs.office.seeds import (
    OfficeTaskSeed,
    SeededOfficeWorld,
    make_office_productivity_world,
    render_workspace_brief,
)

__all__ = [
    "OfficeTaskSeed",
    "SeededOfficeWorld",
    "make_office_productivity_world",
    "render_workspace_brief",
]
