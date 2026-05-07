"""State objects for the grounded office-productivity sandbox."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A neutral tool invocation."""

    name: str
    args: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """A sandbox tool result."""

    ok: bool
    value: Any = None
    error: str = ""


@dataclass
class FileObject:
    """A file in the office workspace."""

    path: str
    content: str
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class Email:
    """An email visible to the agent."""

    email_id: str
    sender: str
    recipients: list[str]
    subject: str
    body: str
    attachments: list[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class SentEmail:
    """A stubbed outbound email side effect."""

    to: list[str]
    subject: str
    body: str
    attachments: list[str] = field(default_factory=list)


@dataclass
class Contact:
    """A contact in the workspace address book."""

    contact_id: str
    name: str
    email: str
    organization: str
    role: str = ""
    notes: str = ""
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class CalendarEvent:
    """A calendar event created inside the sandbox."""

    title: str
    start: str
    end: str
    attendees: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ActionRecord:
    """One grounded action and its result."""

    index: int
    actor: str
    thought: str
    call: ToolCall
    result: ToolResult


@dataclass
class OfficeState:
    """Mutable state for the office-productivity sandbox."""

    files: dict[str, FileObject] = field(default_factory=dict)
    inbox: dict[str, Email] = field(default_factory=dict)
    contacts: dict[str, Contact] = field(default_factory=dict)
    sent_emails: list[SentEmail] = field(default_factory=list)
    calendar: list[CalendarEvent] = field(default_factory=list)
    action_log: list[ActionRecord] = field(default_factory=list)

    def clone(self) -> OfficeState:
        """Return an independent copy for rollouts or prefill replay."""

        return copy.deepcopy(self)
