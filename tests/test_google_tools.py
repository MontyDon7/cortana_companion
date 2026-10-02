import base64
import json
from email import message_from_bytes

import httpx

from google_tools import GoogleClient, build_tools, make_email_notifier
from workflow import Step, WorkflowResult, WorkflowStatus


def _client(handler) -> GoogleClient:
    def route(request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        return handler(request)

    return GoogleClient(
        client_id="id",
        client_secret="secret",
        refresh_token="refresh",
        http=httpx.AsyncClient(transport=httpx.MockTransport(route)),
    )


async def test_gmail_search_returns_subjects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer tok"
        if request.url.path.endswith("/messages"):
            assert request.url.params["q"] == "from:bob"
            return httpx.Response(200, json={"messages": [{"id": "m1"}]})
        return httpx.Response(
            200,
            json={
                "snippet": "hello there",
                "payload": {"headers": [{"name": "Subject", "value": "Hi"}]},
            },
        )

    tools = build_tools(_client(handler))
    out = await tools["gmail_search"](query="from:bob")
    assert "Hi" in out and "hello there" in out


async def test_gmail_search_no_results() -> None:
    tools = build_tools(_client(lambda r: httpx.Response(200, json={})))
    assert "No messages" in await tools["gmail_search"](query="x")


async def test_gmail_draft_creates_draft_not_send() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "d1"})

    tools = build_tools(_client(handler))
    out = await tools["gmail_create_draft"](to="a@b.com", subject="S", body="B")

    assert seen["path"].endswith("/drafts")
    assert "d1" in out
    assert "gmail_send" not in tools  # the planner must not send mail on its own


async def test_calendar_list_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "summary": "Standup",
                        "start": {"dateTime": "2026-10-03T09:00:00Z"},
                    }
                ]
            },
        )

    tools = build_tools(_client(handler))
    assert "Standup" in await tools["calendar_list_events"](days=3)


async def test_calendar_create_event() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"htmlLink": "http://cal/e1"})

    tools = build_tools(_client(handler))
    out = await tools["calendar_create_event"](
        summary="Lunch", start="2026-10-05T12:00:00Z", end="2026-10-05T13:00:00Z"
    )
    assert seen["body"]["summary"] == "Lunch"
    assert "http://cal/e1" in out


async def test_drive_search() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "budget" in request.url.params["q"]
        return httpx.Response(
            200, json={"files": [{"name": "Budget.xlsx", "webViewLink": "http://d/1"}]}
        )

    tools = build_tools(_client(handler))
    assert "Budget.xlsx" in await tools["drive_search"](query="budget")


async def test_drive_search_escapes_quotes() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["q"] = request.url.params["q"]
        return httpx.Response(200, json={"files": []})

    tools = build_tools(_client(handler))
    await tools["drive_search"](query="it's")
    assert "it\\'s" in seen["q"]


async def test_http_error_surfaces_as_exception() -> None:
    tools = build_tools(_client(lambda r: httpx.Response(403, json={"error": "no"})))
    try:
        await tools["drive_search"](query="x")
    except httpx.HTTPStatusError:
        return
    raise AssertionError("expected HTTPStatusError")


async def test_email_notifier_sends_summary_to_owner_only() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        raw = json.loads(request.content)["raw"]
        seen["msg"] = message_from_bytes(base64.urlsafe_b64decode(raw))
        return httpx.Response(200, json={"id": "sent1"})

    notify = make_email_notifier(_client(handler), to="me@example.com")
    await notify(
        WorkflowResult(
            "file report",
            WorkflowStatus.COMPLETED,
            [Step("drive_search", {})],
            "drive_search: Budget.xlsx",
        )
    )

    assert seen["path"].endswith("/messages/send")
    assert seen["msg"]["To"] == "me@example.com"
    assert "file report" in seen["msg"]["Subject"]
    assert "Budget.xlsx" in seen["msg"].get_payload(decode=True).decode()


def test_client_from_env_requires_all_credentials() -> None:
    from google_tools import client_from_env

    assert client_from_env({}) is None
    assert client_from_env({"GOOGLE_CLIENT_ID": "a"}) is None
    full = {
        "GOOGLE_CLIENT_ID": "a",
        "GOOGLE_CLIENT_SECRET": "b",
        "GOOGLE_REFRESH_TOKEN": "c",
    }
    assert client_from_env(full) is not None
