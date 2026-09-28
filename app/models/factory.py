"""Router construction: pick a backend from settings, or build one directly.

Loading the model is separate from building the object, so a server can create
the router at import time and pay the checkpoint cost once in its lifespan.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings, get_settings
from .router import FakeRouter, LayaRouter, RouterDecision, RouterModel

__all__ = [
    "FakeRouter",
    "LayaRouter",
    "RouterDecision",
    "RouterModel",
    "build_router",
    "get_router",
    "set_router",
]


def build_router(settings: Settings | None = None, *, load: bool = False) -> RouterModel:
    """Create the router named by `ROUTER_BACKEND`.

    Args:
        settings: settings to read, defaulting to the process singleton.
        load: load the model now instead of at first use. The FastAPI lifespan
            passes True; the CLI and the tests leave it lazy.

    Returns:
        A `LayaRouter` or a `FakeRouter`.
    """
    settings = settings or get_settings()
    if settings.router_backend == "fake":
        return FakeRouter()

    router = LayaRouter(
        checkpoint=settings.router_checkpoint,
        device=settings.router_device,
    )
    if load:
        router.load()  # type: ignore[union-attr]
    return router


# The process-wide router, set once by the FastAPI lifespan or the CLI.
_router: RouterModel | None = None


def get_router() -> RouterModel:
    """The shared router, built on first use."""
    global _router
    if _router is None:
        _router = build_router()
    return _router


def set_router(router: RouterModel | None) -> None:
    """Replace the shared router. Used by tests to inject a fake."""
    global _router
    _router = router


def router_health() -> dict[str, Any]:
    """Router status for the health endpoint."""
    router = get_router()
    return {"name": getattr(router, "name", "unknown"), **router.health()}
