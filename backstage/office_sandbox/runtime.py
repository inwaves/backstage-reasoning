"""Neutral grounded tools for the office-productivity sandbox."""

from __future__ import annotations

from backstage.office_sandbox.state import (
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


class OfficeRuntime:
    """Executes neutral tool calls against an :class:`OfficeState`."""

    def __init__(self, state: OfficeState):
        self.state = state

    def execute(
        self,
        call: ToolCall,
        *,
        actor: str = "agent",
        thought: str = "",
        record: bool = True,
    ) -> ToolResult:
        """Execute one tool call and append it to the action log."""

        handler = getattr(self, f"_tool_{call.name}", None)
        if handler is None:
            result = ToolResult(ok=False, error=f"unknown tool: {call.name}")
        else:
            try:
                result = handler(**call.args)
            except TypeError as exc:
                result = ToolResult(ok=False, error=f"bad arguments: {exc}")

        if record:
            self.state.action_log.append(
                ActionRecord(
                    index=len(self.state.action_log),
                    actor=actor,
                    thought=thought,
                    call=call,
                    result=result,
                )
            )
        return result

    def _tool_list_files(self, prefix: str = "") -> ToolResult:
        paths = sorted(
            path for path in self.state.files if not prefix or path.startswith(prefix)
        )
        return ToolResult(ok=True, value=paths)

    def _tool_search_files(
        self,
        query: str = "",
        prefix: str = "",
        limit: int = 20,
    ) -> ToolResult:
        matches = []
        for file_obj in self.state.files.values():
            if prefix and not file_obj.path.startswith(prefix):
                continue
            haystack = " ".join(
                [
                    file_obj.path,
                    file_obj.content,
                    str(file_obj.metadata.get("title", "")),
                    str(file_obj.metadata.get("summary", "")),
                ]
            )
            if query and not _contains_query(haystack, query):
                continue
            matches.append(
                {
                    "path": file_obj.path,
                    "title": file_obj.metadata.get("title", file_obj.path),
                    "summary": file_obj.metadata.get("summary", ""),
                    "tags": file_obj.metadata.get("tags", []),
                }
            )
        return ToolResult(ok=True, value=matches[: max(limit, 0)])

    def _tool_read_file(self, path: str) -> ToolResult:
        file_obj = self.state.files.get(path)
        if file_obj is None:
            return ToolResult(ok=False, error=f"file not found: {path}")
        return ToolResult(
            ok=True,
            value={
                "path": file_obj.path,
                "content": file_obj.content,
                "metadata": file_obj.metadata,
            },
        )

    def _tool_read_email(self, email_id: str) -> ToolResult:
        email = self.state.inbox.get(email_id)
        if email is None:
            return ToolResult(ok=False, error=f"email not found: {email_id}")
        return ToolResult(
            ok=True,
            value={
                "email_id": email.email_id,
                "sender": email.sender,
                "recipients": email.recipients,
                "subject": email.subject,
                "body": email.body,
                "attachments": email.attachments,
                "metadata": email.metadata,
            },
        )

    def _tool_search_email(
        self,
        query: str = "",
        sender: str = "",
        limit: int = 10,
    ) -> ToolResult:
        matches = []
        for email in self.state.inbox.values():
            if sender and sender.lower() not in email.sender.lower():
                continue
            haystack = " ".join(
                [
                    email.sender,
                    " ".join(email.recipients),
                    email.subject,
                    email.body,
                    " ".join(email.attachments),
                ]
            )
            if query and not _contains_query(haystack, query):
                continue
            matches.append(_email_summary(email))
        matches.sort(key=lambda item: str(item.get("timestamp", "")), reverse=True)
        return ToolResult(ok=True, value=matches[: max(limit, 0)])

    def _tool_list_contacts(self, query: str = "", limit: int = 20) -> ToolResult:
        matches = []
        for contact in self.state.contacts.values():
            haystack = " ".join(
                [
                    contact.name,
                    contact.email,
                    contact.organization,
                    contact.role,
                    contact.notes,
                ]
            )
            if query and not _contains_query(haystack, query):
                continue
            matches.append(_contact_summary(contact))
        matches.sort(key=lambda item: str(item["name"]))
        return ToolResult(ok=True, value=matches[: max(limit, 0)])

    def _tool_read_contact(self, contact_id: str) -> ToolResult:
        contact = self.state.contacts.get(contact_id)
        if contact is None:
            contact = next(
                (
                    item
                    for item in self.state.contacts.values()
                    if item.email.lower() == contact_id.lower()
                ),
                None,
            )
        if contact is None:
            return ToolResult(ok=False, error=f"contact not found: {contact_id}")
        return ToolResult(
            ok=True,
            value={
                "contact_id": contact.contact_id,
                "name": contact.name,
                "email": contact.email,
                "organization": contact.organization,
                "role": contact.role,
                "notes": contact.notes,
                "metadata": contact.metadata,
            },
        )

    def _tool_send_email(
        self,
        to: str | list[str],
        subject: str,
        body: str,
        attachments: list[str] | None = None,
    ) -> ToolResult:
        recipients = [to] if isinstance(to, str) else list(to)
        attachment_paths = list(attachments or [])
        missing = [path for path in attachment_paths if path not in self.state.files]
        if missing:
            return ToolResult(
                ok=False,
                error=f"attachment(s) not found: {', '.join(sorted(missing))}",
            )

        sent = SentEmail(
            to=recipients,
            subject=subject,
            body=body,
            attachments=attachment_paths,
        )
        self.state.sent_emails.append(sent)
        return ToolResult(
            ok=True,
            value={
                "to": recipients,
                "subject": subject,
                "attachments": attachment_paths,
            },
        )

    def _tool_write_file(self, path: str, content: str) -> ToolResult:
        existed = path in self.state.files
        self.state.files[path] = FileObject(path=path, content=content)
        return ToolResult(ok=True, value={"path": path, "existed": existed})

    def _tool_delete_file(self, path: str) -> ToolResult:
        if path not in self.state.files:
            return ToolResult(ok=False, error=f"file not found: {path}")
        del self.state.files[path]
        return ToolResult(ok=True, value={"path": path})

    def _tool_list_calendar_events(
        self,
        start: str = "",
        end: str = "",
        query: str = "",
        limit: int = 20,
    ) -> ToolResult:
        matches = []
        for event in self.state.calendar:
            if start and event.end < start:
                continue
            if end and event.start > end:
                continue
            haystack = " ".join([event.title, event.notes, " ".join(event.attendees)])
            if query and not _contains_query(haystack, query):
                continue
            matches.append(
                {
                    "title": event.title,
                    "start": event.start,
                    "end": event.end,
                    "attendees": event.attendees,
                    "notes": event.notes,
                }
            )
        matches.sort(key=lambda item: str(item["start"]))
        return ToolResult(ok=True, value=matches[: max(limit, 0)])

    def _tool_create_calendar_event(
        self,
        title: str,
        start: str,
        end: str,
        attendees: list[str] | None = None,
        notes: str = "",
    ) -> ToolResult:
        event = CalendarEvent(
            title=title,
            start=start,
            end=end,
            attendees=list(attendees or []),
            notes=notes,
        )
        self.state.calendar.append(event)
        return ToolResult(
            ok=True,
            value={
                "title": title,
                "start": start,
                "end": end,
                "attendees": event.attendees,
            },
        )


def _contains_query(text: str, query: str) -> bool:
    return query.lower() in text.lower()


def _email_summary(email: Email) -> dict[str, object]:
    return {
        "email_id": email.email_id,
        "sender": email.sender,
        "subject": email.subject,
        "timestamp": email.metadata.get("timestamp", ""),
        "attachments": email.attachments,
        "snippet": email.body[:160],
    }


def _contact_summary(contact: Contact) -> dict[str, object]:
    return {
        "contact_id": contact.contact_id,
        "name": contact.name,
        "email": contact.email,
        "organization": contact.organization,
        "role": contact.role,
    }
