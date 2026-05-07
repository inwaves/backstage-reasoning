"""Grounded office-productivity sandbox for Phase 0."""

from .policy import (
    DATA_EXFILTRATION,
    DESTRUCTIVE_MODIFICATION,
    UNAUTHORIZED_EXTERNAL_ACTION,
    OutcomeScore,
    OutcomeScorer,
    SafetyPolicy,
    Violation,
)
from .prefill import GroundedPrefill, PrefillStep, apply_prefill
from .runtime import OfficeRuntime
from .seeds import (
    OfficeTaskSeed,
    SeededOfficeWorld,
    make_office_productivity_world,
    render_workspace_brief,
)
from .state import (
    ActionRecord,
    CalendarEvent,
    Contact,
    Email,
    FileObject,
    OfficeState,
    SentEmail,
    ToolCall,
    ToolResult,
)

__all__ = [
    "DATA_EXFILTRATION",
    "DESTRUCTIVE_MODIFICATION",
    "UNAUTHORIZED_EXTERNAL_ACTION",
    "ActionRecord",
    "CalendarEvent",
    "Contact",
    "Email",
    "FileObject",
    "GroundedPrefill",
    "OfficeRuntime",
    "OfficeState",
    "OfficeTaskSeed",
    "OutcomeScore",
    "OutcomeScorer",
    "PrefillStep",
    "SafetyPolicy",
    "SeededOfficeWorld",
    "SentEmail",
    "ToolCall",
    "ToolResult",
    "Violation",
    "apply_prefill",
    "make_office_productivity_world",
    "render_workspace_brief",
]
