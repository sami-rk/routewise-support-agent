"""Shared FastAPI dependencies: settings, the graph, and staff authorisation."""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import Header, HTTPException, Request, status

from ..agent.graph import build_graph, thread_config
from ..config import Settings, get_settings
from ..observability.tracing import new_run_id


def settings_dep() -> Settings:
    return get_settings()


def graph_dep(request: Request):
    """The compiled graph, built once in the lifespan."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The agent is still starting up.",
        )
    return graph


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    """Guard the staff endpoints.

    Uses `secrets.compare_digest` so the comparison does not leak the key length
    or prefix through timing.
    """
    settings = get_settings()
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-Admin-Key header is required.",
        )


def config_for(thread_id: str) -> dict[str, Any]:
    """The LangGraph config for a thread, with a run id for the traces."""
    return {"configurable": {"thread_id": thread_id}, "run_id": new_run_id()}


__all__ = [
    "build_graph",
    "config_for",
    "graph_dep",
    "require_admin",
    "settings_dep",
    "thread_config",
]
