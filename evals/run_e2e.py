"""Run the scripted end-to-end conversations in `e2e_cases.yaml`.

    python -m evals.run_e2e                # stubbed LLM, fully offline
    python -m evals.run_e2e --real         # live OpenRouter free models
    python -m evals.run_e2e --only refund   # cases whose name matches

The assertions are about what the graph *did* — a refund row exists, a thread
paused with the right mode, a ticket was created — not about the exact words a
model produced. With `--real` the prose is the live model's; everything else is
identical, so a case that passes offline and fails live is telling you something
about the model rather than about the wiring.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from langgraph.types import Command

from app.agent.graph import build_graph, thread_config
from app.config import get_settings, reset_settings_cache
from tests.stub_llm import ScriptedLLM

CASES_PATH = Path(__file__).with_name("e2e_cases.yaml")


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    """Read the case file."""
    cases = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    for case in cases:
        if "name" not in case or "messages" not in case:
            raise ValueError(f"Every case needs a name and messages: {case!r}")
    return cases


def scripted_replies(case: dict[str, Any]) -> list[Any]:
    """Turn the YAML reply shorthand into stub replies.

    A string is prose; a mapping with `tool` is a tool call.
    """
    replies: list[Any] = []
    for reply in case.get("replies", []) or []:
        if isinstance(reply, str):
            replies.append(reply)
        elif isinstance(reply, dict) and "tool" in reply:
            replies.append({"content": "", "tool_calls": [{"name": reply["tool"], "args": reply.get("args", {})} | {"id": f"call_{len(replies)}"}]})
        else:
            raise ValueError(f"Unusable reply in case {case['name']!r}: {reply!r}")
    return replies


def run_case(case: dict[str, Any], *, real: bool = False) -> dict[str, Any]:
    """Run one case and check its assertions.

    Each case starts from a freshly seeded database: cases that refund an
    invoice would otherwise make every later case for the same customer fail with
    "already refunded".

    Returns:
        A result dict with `name`, `state`, and one entry per failed assertion.
    """
    from tests.conftest import user_turn

    settings = get_settings()
    thread_id = f"e2e_{abs(hash(case['name'])) % 100000}"
    customer = case.get("customer", "cus_ada")
    message = case["messages"][0]

    replies = scripted_replies(case)
    if case.get("resume") is not None:
        replies += scripted_replies({"replies": case.get("resume_replies", [])})

    from langgraph.checkpoint.sqlite import SqliteSaver

    reset_database()

    state: dict[str, Any] = {}
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as saver:
        graph = build_graph(checkpointer=saver, settings=settings)
        # A default reply keeps a case from depending on exactly how many times a
        # model loops. The assertions are about actions, not about call counts.
        context = ScriptedLLM(replies, default=NEUTRAL_REPLY) if not real else None
        if context:
            context.__enter__()
        try:
            state = graph.invoke(user_turn(message, thread_id=thread_id, customer_id=customer), thread_config(thread_id))
            if case.get("resume") is not None:
                state = graph.invoke(
                    Command(
                        resume={
                            "decision": "approved" if case["resume"] in ("approved", True) else "rejected",
                            "approved_by": case.get("resume_approved_by", "staff:eval"),
                        }
                    ),
                    thread_config(thread_id),
                )
        finally:
            if context:
                context.__exit__(None, None, None)

    return {"name": case["name"], "state": state, "failures": check(case, state)}


NEUTRAL_REPLY = "Thanks for your patience, I am checking that for you now."

# The lexical embedder scores text that shares words above 0 and unrelated text
# near 0, so the floor is low. The shipped default of 0.35 is tuned for
# bge-small-en-v1.5 and would reject everything here.
HASH_MIN_SCORE = 0.02


def build_eval_retriever(workspace: Path, *, force_hash: bool = False) -> Any:
    """The retriever the eval runs against.

    Prefers the real index, so a developer with `scripts.build_kb` run gets the
    same retrieval the product uses. When the embedding model is not installed —
    which is the case in CI, deliberately, to keep it to a few minutes — it falls
    back to building an index from the same documents with the deterministic
    hash embedder. The eval is about graph behaviour: refunds, approvals,
    guardrails and tickets. Which passages come back is covered by the RAG
    tests, and retrieval quality by `scripts/build_kb` and a human reading the
    citations.

    Args:
        workspace: a temporary directory for the fallback index.
        force_hash: skip the real index even when it is usable.

    Returns:
        A `Retriever`, also installed as the process-wide one.
    """
    from app.rag.retriever import Retriever, set_retriever

    # Loading the index succeeds even when the embedding model is missing,
    # because SentenceTransformerEmbedder is lazy and only imports the model on
    # the first query. So check for the module, not for a successful load.
    import importlib.util

    have_model = importlib.util.find_spec("sentence_transformers") is not None
    have_index = (Path.cwd() / "data" / "kb_index" / "index.faiss").exists()

    if not force_hash and have_model and have_index:
        try:
            from app.rag.retriever import get_retriever

            return get_retriever()
        except Exception as exc:  # noqa: BLE001 - any failure means "use the fallback"
            print(f"real index unusable ({type(exc).__name__}); using the lexical index")
    else:
        missing = []
        if not have_model:
            missing.append("sentence-transformers is not installed")
        if not have_index:
            missing.append("the index has not been built (python -m scripts.build_kb)")
        print("using the lexical index: " + "; ".join(missing))

    from app.rag.chunking import chunk_directory
    from app.rag.indexer import build_index, load_index
    from app.rag.ingest import HashEmbedder

    kb_dir = Path.cwd() / "data" / "knowledge_base"
    chunks = chunk_directory(kb_dir)
    embedder = HashEmbedder(512)
    index_path = workspace / "kb_index" / "index.faiss"
    build_index(chunks, embedder, index_path)
    index, passages, metadata = load_index(index_path)
    retriever = Retriever(
        index, passages, embedder, top_k=4, min_score=HASH_MIN_SCORE, metadata=metadata
    )
    set_retriever(retriever)
    return retriever


def reset_database() -> None:
    """Empty the database and re-seed, so a case starts from a known state."""
    from app.db.session import execute, get_connection, init_db
    from scripts.seed_db import seed

    connection = get_connection()
    for table in (
        "refunds", "tickets", "invoices", "subscriptions", "customers",
        "customer_memory", "traces", "router_decisions", "llm_calls", "pending_actions",
    ):
        execute(f"DELETE FROM {table}")
    init_db()
    seed()
    connection.commit()


def check(case: dict[str, Any], state: dict[str, Any]) -> list[str]:
    """Compare a finished turn against the case's expectations."""
    from app.db.session import query_all, query_one

    expect = case.get("expect", {}) or {}
    failures: list[str] = []
    response = state.get("response") or ""

    def fail(message: str) -> None:
        failures.append(message)

    def interrupts() -> list[dict[str, Any]]:
        found = state.get("__interrupt__") or []
        return [getattr(i, "value", i) for i in found]

    def refunded(invoice_id: str) -> bool:
        return bool(query_all("SELECT id FROM refunds WHERE invoice_id = ?", (invoice_id,)))

    if "intent" in expect and state.get("intent") != expect["intent"]:
        fail(f"intent was {state.get('intent')!r}, expected {expect['intent']!r}")
    if expect.get("escalated") is not None and bool(state.get("escalate")) != expect["escalated"]:
        fail(f"escalated was {bool(state.get('escalate'))}, expected {expect['escalated']}")
    if expect.get("unsafe") is not None and bool(state.get("unsafe")) != expect["unsafe"]:
        fail(f"unsafe was {bool(state.get('unsafe'))}, expected {expect['unsafe']}")

    if "contains" in expect and expect["contains"] not in response:
        fail(f"reply did not contain {expect['contains']!r}: {response[:120]!r}")
    if "contains_not" in expect and expect["contains_not"] in response:
        fail(f"reply should not have contained {expect['contains_not']!r}")

    citations = " ".join(p.get("citation", "") for p in (state.get("retrieved") or []))
    if "has_citation" in expect and expect["has_citation"] not in citations:
        fail(f"no citation from {expect['has_citation']}; got {citations[:100]!r}")
    if expect.get("no_retrieved") and state.get("retrieved"):
        fail(f"expected no passages, got {len(state['retrieved'])}")

    if expect.get("no_interrupt") and interrupts():
        fail("the turn should not have interrupted")
    if "interrupt_mode" in expect:
        found = interrupts()
        if not found:
            fail(f"expected an interrupt, got none ({state.get('response', '')[:80]!r})")
        elif found[0].get("mode") != expect["interrupt_mode"]:
            fail(f"interrupt mode was {found[0].get('mode')!r}, expected {expect['interrupt_mode']!r}")
    if "interrupt_amount" in expect:
        found = interrupts()
        if not found or found[0].get("amount") != expect["interrupt_amount"]:
            fail(f"interrupt amount was {found[0].get('amount') if found else None!r}, expected {expect['interrupt_amount']!r}")

    if expect.get("has_ticket") and not state.get("ticket_id"):
        fail("expected a ticket to be created")
    if expect.get("no_ticket") and state.get("ticket_id"):
        fail(f"a ticket was created: {state.get('ticket_id')}")

    action = state.get("action_result") or {}
    if "action_ok" in expect and action.get("ok") != expect["action_ok"]:
        fail(f"action ok was {action.get('ok')!r} ({action.get('reason', '')!r}), expected {expect['action_ok']!r}")
    if "action_type" in expect and action.get("type") != expect["action_type"]:
        fail(f"action type was {action.get('type')!r}, expected {expect['action_type']!r}")

    if "refunded" in expect:
        invoice = expect["refunded"]
        row = query_one("SELECT approved_by FROM refunds ORDER BY created_at DESC, id DESC LIMIT 1")
        if invoice == "none":
            if row is not None:
                fail(f"nothing should have been refunded, but {dict(row)} was")
        else:
            if not refunded(invoice):
                fail(f"invoice {invoice} was not refunded")
            elif "refund_approved_by" in expect and row and row["approved_by"] != expect["refund_approved_by"]:
                fail(f"approved_by was {row['approved_by']!r}, expected {expect['refund_approved_by']!r}")

    if "subscription_status" in expect:
        row = query_one("SELECT status FROM subscriptions WHERE customer_id = ?", (case.get("customer"),))
        if not row or row["status"] != expect["subscription_status"]:
            fail(f"subscription was {row['status'] if row else None!r}, expected {expect['subscription_status']!r}")
    if "subscription_still" in expect:
        row = query_one("SELECT status FROM subscriptions WHERE customer_id = ?", (case.get("customer"),))
        if not row or row["status"] != expect["subscription_still"]:
            fail(f"subscription was {row['status'] if row else None!r}, expected {expect['subscription_still']!r}")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the scripted end-to-end conversations.")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--real", action="store_true", help="Use live OpenRouter models.")
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="Always use the lexical index, even if the real one is available.",
    )
    parser.add_argument("--only", default=None, help="Only cases whose name contains this.")
    args = parser.parse_args(argv)

    from app.config import validate_settings

    validate_settings()
    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if args.only.lower() in c["name"].lower()]

    # An isolated database and index, so an eval never disturbs real data.
    workspace = Path(tempfile.mkdtemp(prefix="e2e-"))
    import os

    os.environ.update(
        DB_PATH=str(workspace / "support.db"),
        CHECKPOINT_DB_PATH=str(workspace / "checkpoints.db"),
        KB_DIR=str(Path.cwd() / "data" / "knowledge_base"),
        ROUTER_BACKEND="fake",
    )
    reset_settings_cache()
    from app.db.session import close_connection, init_db, set_db_path
    from app.models.factory import set_router
    from app.models.router import FakeRouter
    from app.rag.retriever import get_retriever, set_retriever
    from scripts.seed_db import seed

    set_router(FakeRouter())
    set_db_path(workspace / "support.db")
    init_db()
    seed()
    retriever = build_eval_retriever(workspace, force_hash=args.hash_only)

    print(f"{len(cases)} cases from {args.cases}")
    print(f"backend: {'live OpenRouter' if args.real else 'stubbed model'}")
    print(f"retrieval: {retriever.health()['embedding_model']}\n")

    passed = 0
    for case in cases:
        try:
            result = run_case(case, real=args.real)
            failures = result["failures"]
        except Exception as exc:  # noqa: BLE001
            failures = [f"{type(exc).__name__}: {exc}"]

        if failures:
            print(f"FAIL  {case['name']}")
            for failure in failures:
                print(f"        {failure}")
        else:
            passed += 1
            print(f"pass  {case['name']}")

    close_connection()
    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
