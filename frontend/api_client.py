"""HTTP client for the backend.

The Streamlit app contains no agent logic: it talks to the API over HTTP and
renders what comes back. That keeps the two deployable separately, and it means
the UI cannot accidentally disagree with the graph about what happened.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
ADMIN_KEY = os.environ.get("ADMIN_API_KEY", "change-me")
TIMEOUT = 120.0


class ApiError(RuntimeError):
    """The backend was unreachable, or answered with an error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _headers(admin: bool = False) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if admin:
        headers["X-Admin-Key"] = ADMIN_KEY
    return headers


def _request(method: str, path: str, *, json: dict[str, Any] | None = None, admin: bool = False) -> Any:
    url = f"{API_BASE_URL.rstrip('/')}{path}"
    try:
        response = httpx.request(method, url, json=json, headers=_headers(admin), timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        raise ApiError(
            f"Could not reach the backend at {API_BASE_URL}. Is it running? ({exc})"
        ) from exc

    if response.status_code >= 400:
        detail = response.text[:300]
        raise ApiError(f"{method} {path} failed ({response.status_code}): {detail}", response.status_code)
    if not response.content:
        return None
    return response.json()


# --- chat -----------------------------------------------------------------


def chat(customer_id: str, message: str, thread_id: str | None = None) -> dict[str, Any]:
    """Send a message. Returns the reply, or an interrupt waiting on a decision."""
    return _request(
        "POST", "/chat", json={"customer_id": customer_id, "message": message, "thread_id": thread_id}
    )


def confirm(thread_id: str, confirmed: bool, note: str | None = None) -> dict[str, Any]:
    """Answer a cancellation waiting on the customer."""
    return _request(
        "POST", "/chat/confirm", json={"thread_id": thread_id, "confirmed": confirmed, "note": note}
    )


def thread_history(thread_id: str) -> dict[str, Any]:
    return _request("GET", f"/threads/{thread_id}")


# --- staff ----------------------------------------------------------------


def decide_approval(thread_id: str, decision: str, note: str | None = None) -> dict[str, Any]:
    return _request(
        "POST", f"/approvals/{thread_id}", json={"decision": decision, "note": note}, admin=True
    )


def pending_approvals() -> list[dict[str, Any]]:
    return _request("GET", "/approvals/pending", admin=True)


def tickets(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    query = f"?limit={limit}" + (f"&status={status}" if status else "")
    return _request("GET", f"/tickets{query}", admin=True)


def set_ticket_status(ticket_id: str, status: str) -> dict[str, Any]:
    return _request("PATCH", f"/tickets/{ticket_id}", json={"status": status}, admin=True)


def metrics() -> dict[str, Any]:
    return _request("GET", "/metrics", admin=True)


def traces(thread_id: str) -> dict[str, Any]:
    return _request("GET", f"/traces/{thread_id}", admin=True)


def health() -> dict[str, Any]:
    return _request("GET", "/health")


def customers() -> list[dict[str, str]]:
    """Demo customers, from the seed data, for the sidebar picker."""
    from app.db.session import query_all

    rows = query_all("SELECT id, name, email FROM customers ORDER BY name")
    return [dict(row) for row in rows]
