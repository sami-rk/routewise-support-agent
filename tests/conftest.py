"""Shared fixtures for the graph tests.

The graph is exercised with the keyword router and a scripted LLM, so no model
is downloaded and no request leaves the machine.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.agent.graph import build_graph
from app.agent.state import new_state
from app.config import Settings
from app.db.session import close_connection, init_db, set_db_path
from app.models.factory import set_router
from app.models.router import FakeRouter
from app.rag.chunking import chunk_markdown
from app.rag.indexer import build_index
from app.rag.ingest import HashEmbedder
from app.rag.retriever import set_retriever, Retriever
from scripts.seed_db import seed


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointed at a temporary database and checkpoint file."""
    return Settings(
        _env_file=None,
        db_path=tmp_path / "support.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        kb_dir=tmp_path / "knowledge_base",
        router_backend="fake",
        refund_window_days=14,
        refund_auto_limit=20.0,
    )


@pytest.fixture
def seeded(settings, monkeypatch) -> Settings:
    """A seeded support database, wired into the settings singleton."""
    from app.config import reset_settings_cache

    monkeypatch.setenv("DB_PATH", str(settings.db_path))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(settings.checkpoint_db_path))
    # The keyword router, so the suite never downloads a 1.7 GB checkpoint.
    monkeypatch.setenv("ROUTER_BACKEND", "fake")
    reset_settings_cache()
    set_db_path(settings.db_path)
    init_db()
    seed()
    yield settings
    set_router(None)
    set_retriever(None)
    close_connection()
    reset_settings_cache()


@pytest.fixture
def knowledge_base(settings) -> Retriever:
    """A small in-memory index with the passages the graph needs."""
    docs = {
        "pricing_plans.md": (
            "# Pricing and plans\n\n## The plans\n\n"
            "| Plan | Price | Storage | Devices |\n|---|---|---|---|\n"
            "| Basic | $9/mo | 100 GB | 2 |\n| Pro | $19/mo | 1 TB | 5 |\n"
            "| Business | $49/mo | 5 TB | Unlimited |\n"
        ),
        "cancellation_refunds.md": (
            "# Cancellation and refunds\n\n## Cancelling your subscription\n\n"
            "You can cancel at any time, from Account > Subscription > Cancel subscription.\n\n"
            "## Refunds\n\nA charge is refundable within 14 days of the charge date. "
            "Refunds above $20 need a support specialist to approve. "
            "A refund returns the charge to your original payment method within 5 to 10 business days.\n"
        ),
        "sync_troubleshooting.md": (
            "# Sync troubleshooting\n\n## 1. Check the status marker\n\n"
            "Look at the sync folder. A tick means synced, a circle means work in progress.\n\n"
            "## 2. Check you are online\n\nOffline mode keeps editing available.\n"
        ),
        "account_password.md": (
            "# Account and password\n\n## Resetting your password\n\n"
            "The reset link expires after 1 hour. Request a new one from Forgot password.\n"
        ),
    }
    kb_dir = settings.kb_dir
    kb_dir.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        (kb_dir / name).write_text(text, encoding="utf-8")

    chunks = []
    for name, text in docs.items():
        chunks.extend(chunk_markdown(text, name, max_chars=400, overlap=0))
    # 512 buckets rather than 64: with a dozen passages a small hash collides
    # often enough that unrelated passages outrank the right one.
    embedder = HashEmbedder(512)
    index_path = kb_dir.parent / "kb_index" / "index.faiss"
    build_index(chunks, embedder, index_path)

    from app.rag.indexer import load_index

    index, passages, metadata = load_index(index_path)
    # A floor above zero, so a question with no lexical overlap with any passage
    # comes back empty and the no-answer path is exercised for real.
    retriever = Retriever(index, passages, embedder, top_k=4, min_score=0.05)
    set_retriever(retriever)
    return retriever


@pytest.fixture
def graph(seeded, knowledge_base):
    """A compiled graph with the keyword router, no checkpointer."""
    return build_graph(checkpointer=None, settings=seeded)


@pytest.fixture
def state():
    """A starting state for one turn."""
    return new_state(thread_id="t1", customer_id="cus_ada", user_input="hello")


def user_turn(message: str, thread_id: str = "t1", customer_id: str = "cus_ada") -> dict:
    """The input a chat request turns into."""
    from langchain_core.messages import HumanMessage

    state = new_state(thread_id=thread_id, customer_id=customer_id, user_input=message)
    state["messages"] = [HumanMessage(content=message)]
    return state
