"""The FastAPI application.

The lifespan loads everything once — the router, the retriever, the compiled
graph and its checkpointer — so a request does not pay for a 1.7 GB model load.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite import SqliteSaver

from .agent.graph import build_graph
from .api import chat, memory, staff
from .config import get_settings
from .db.session import init_db
from .rag.retriever import is_index_built

logger = logging.getLogger(__name__)

DESCRIPTION = """A customer support agent for CloudSync Pro.

Routing is done locally with Laya; every LLM call goes through OpenRouter on a
free-tier model. Refunds and cancellations pause for a decision before anything
happens."""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the models, the index and the graph once, then serve."""
    settings = get_settings()
    logging.basicConfig(level=logging.INFO)

    init_db()

    from .models.factory import build_router

    router = build_router(load=settings.router_backend == "laya")
    if settings.router_backend == "laya":
        logger.info("router ready: %s", router.health())
    app.state.router = router

    if is_index_built(settings):
        from .rag.retriever import get_retriever

        app.state.retriever = get_retriever()
        logger.info("knowledge base: %s", app.state.retriever.health())
    else:
        # Not fatal: retrieval returns nothing and the agent says it does not
        # know, rather than the server refusing to start.
        app.state.retriever = None
        logger.warning(
            "no FAISS index at %s; run python -m scripts.build_kb", settings.kb_index_dir
        )

    # The saver is a context manager, so the connection has to outlive the app.
    saver_cm = SqliteSaver.from_conn_string(str(settings.checkpoint_db_path))
    saver = saver_cm.__enter__()
    app.state.saver_cm = saver_cm
    app.state.graph = build_graph(checkpointer=saver, settings=settings)

    if not settings.has_llm_key:
        logger.warning(
            "OPENROUTER_API_KEY is not set: LLM nodes will answer with the busy message."
        )

    try:
        yield
    finally:
        saver_cm.__exit__(None, None, None)


def create_app() -> FastAPI:
    """Build the application, so tests can make one with their own settings."""
    app = FastAPI(
        title="CloudSync Pro support agent",
        description=DESCRIPTION,
        version="2.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat.router)
    app.include_router(memory.router)
    app.include_router(staff.router)

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, Any]:
        """Whether the service is usable, and what is loaded.

        Reports the three things that decide whether a request will work: the
        router, the knowledge base index, and whether an LLM key is present.
        """
        settings = get_settings()
        router = getattr(app.state, "router", None)
        retriever = getattr(app.state, "retriever", None)
        database_ok = True
        try:
            from .db.session import query_one

            query_one("SELECT 1")
        except Exception as exc:  # noqa: BLE001
            logger.error("database check failed: %s", exc)
            database_ok = False

        return {
            "status": "ok" if (router is not None and database_ok) else "degraded",
            "router": router.health() if router else {"loaded": False},
            "knowledge_base": retriever.health() if retriever else {"loaded": False},
            "llm_key_present": settings.has_llm_key,
            "llm_models": settings.llm_models,
            "database": database_ok,
            "graph_loaded": getattr(app.state, "graph", None) is not None,
        }

    return app


app = create_app()
