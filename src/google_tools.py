"""Gmail, Calendar and Drive tools for the workflow runner (Google REST APIs)."""

import base64
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import httpx

from workflow import Notifier, Tool, WorkflowResult

TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"
CALENDAR = "https://www.googleapis.com/calendar/v3/calendars/primary"
DRIVE = "https://www.googleapis.com/drive/v3"


class GoogleClient:
    """Minimal authenticated client using an OAuth refresh token."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._creds = {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        self._http = http or httpx.AsyncClient(timeout=30)
        self._token = ""
        self._expires = 0.0

    async def _access_token(self) -> str:
        if time.monotonic() >= self._expires:
            resp = await self._http.post(TOKEN_URL, data=self._creds)
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._expires = time.monotonic() + data.get("expires_in", 3600) - 60
        return self._token

    async def request(self, method: str, url: str, **kwargs) -> dict:
        headers = {"Authorization": f"Bearer {await self._access_token()}"}
        resp = await self._http.request(method, url, headers=headers, **kwargs)
        resp.raise_for_status()
        return resp.json()


def _raw_message(to: str, subject: str, body: str) -> str:
    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    return base64.urlsafe_b64encode(msg.as_bytes()).decode()


def build_tools(client: GoogleClient) -> dict[str, Tool]:
    """Tools the planner may use. Sending mail is deliberately NOT included:
    the planner can only draft, so a bad plan can't email third parties."""

    async def gmail_search(query: str, max_results: int = 5) -> str:
        """Search Gmail; returns subject and snippet of each match."""
        found = await client.request(
            "GET", f"{GMAIL}/messages", params={"q": query, "maxResults": max_results}
        )
        lines = []
        for m in found.get("messages", []):
            detail = await client.request(
                "GET",
                f"{GMAIL}/messages/{m['id']}",
                params={"format": "metadata", "metadataHeaders": "Subject"},
            )
            subject = next(
                (
                    h["value"]
                    for h in detail.get("payload", {}).get("headers", [])
                    if h["name"] == "Subject"
                ),
                "(no subject)",
            )
            lines.append(f"{subject} - {detail.get('snippet', '')}")
        return "\n".join(lines) or "No messages found."

    async def gmail_create_draft(to: str, subject: str, body: str) -> str:
        """Create a Gmail draft (does not send)."""
        draft = await client.request(
            "POST",
            f"{GMAIL}/drafts",
            json={"message": {"raw": _raw_message(to, subject, body)}},
        )
        return f"Draft created (id {draft.get('id')})."

    async def calendar_list_events(days: int = 7) -> str:
        """List upcoming calendar events for the next N days."""
        now = datetime.now(timezone.utc)
        data = await client.request(
            "GET",
            f"{CALENDAR}/events",
            params={
                "timeMin": now.isoformat(),
                "timeMax": (now + timedelta(days=days)).isoformat(),
                "singleEvents": "true",
                "orderBy": "startTime",
            },
        )
        lines = [
            f"{e.get('start', {}).get('dateTime') or e.get('start', {}).get('date')}"
            f" {e.get('summary', '(untitled)')}"
            for e in data.get("items", [])
        ]
        return "\n".join(lines) or "No upcoming events."

    async def calendar_create_event(summary: str, start: str, end: str) -> str:
        """Create a calendar event. start/end are RFC3339 timestamps."""
        event = await client.request(
            "POST",
            f"{CALENDAR}/events",
            json={
                "summary": summary,
                "start": {"dateTime": start},
                "end": {"dateTime": end},
            },
        )
        return f"Event created: {event.get('htmlLink')}"

    async def drive_search(query: str, max_results: int = 5) -> str:
        """Search Drive files whose name or content contains the query."""
        escaped = query.replace("\\", "\\\\").replace("'", "\\'")
        data = await client.request(
            "GET",
            f"{DRIVE}/files",
            params={
                "q": f"fullText contains '{escaped}' and trashed = false",
                "pageSize": max_results,
                "fields": "files(name,webViewLink)",
            },
        )
        lines = [
            f"{f['name']} - {f.get('webViewLink', '')}" for f in data.get("files", [])
        ]
        return "\n".join(lines) or "No files found."

    return {
        f.__name__: f
        for f in (
            gmail_search,
            gmail_create_draft,
            calendar_list_events,
            calendar_create_event,
            drive_search,
        )
    }


def make_email_notifier(client: GoogleClient, to: str) -> Notifier:
    """Email the workflow result to a fixed recipient (the owner)."""

    async def notify(result: WorkflowResult) -> None:
        await client.request(
            "POST",
            f"{GMAIL}/messages/send",
            json={
                "raw": _raw_message(
                    to,
                    f"Cortana: {result.status.value} - {result.task}",
                    f"Task: {result.task}\nStatus: {result.status.value}\n\n"
                    f"{result.summary}",
                )
            },
        )

    return notify


def client_from_env(env) -> GoogleClient | None:
    """Build a client from GOOGLE_* env vars, or None if any are missing."""
    keys = ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN")
    if not all(env.get(k) for k in keys):
        return None
    return GoogleClient(*(env[k] for k in keys))
