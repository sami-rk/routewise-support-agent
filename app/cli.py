"""A terminal chat loop over the same graph the API uses.

This is the development front door, mirroring the original project's
`python agent.py`. It shows the routing decision under each reply, so a
misroute is visible while you are writing the fix rather than after.

    python -m app.cli                       # Laya if it is available
    python -m app.cli --router fake         # keywords, no model download
    python -m app.cli --customer cus_ada --kb-dir ./docs
"""

from __future__ import annotations

import argparse
import uuid
from typing import Any

from .agent.graph import build_graph, thread_config
from .agent.state import new_state
from .config import get_settings
from .db.session import close_connection, init_db
from .observability.tracing import new_run_id

BANNER = "CloudSync Pro support agent. Type 'exit' to leave, 'trace' for the last turn."


def ask(prompt: str = "you> ") -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "exit"


def show(result: dict[str, Any]) -> None:
    """Print the reply plus the chips the Streamlit UI shows."""
    print(f"bot> {result.get('response') or ''}")

    chips = [f"intent={result.get('intent')}"]
    if result.get("intent_conf") is not None:
        chips.append(f"conf={float(result['intent_conf']):.2f}")
    if result.get("tools_used"):
        chips.append("tools=" + ",".join(result["tools_used"]))
    sources = [p.get("citation") for p in (result.get("retrieved") or []) if p.get("citation")]
    if sources:
        chips.append("sources=" + ", ".join(sources))
    print("     " + "  ".join(chips))

    if result.get("escalate"):
        print(f"     [ESCALATED] ticket {result.get('ticket_id')}")
    if result.get("llm_note") == "rate_limited":
        print("     [models busy] the reply above is the fallback")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Chat with the support agent in the terminal.")
    parser.add_argument("--customer", default="cus_ada", help="Customer id to act as.")
    parser.add_argument("--router", choices=("laya", "fake"), default=None)
    parser.add_argument("--thread", default=None, help="Resume an existing thread id.")
    parser.add_argument("--kb-dir", default=None, help="Alternate knowledge base directory.")
    args = parser.parse_args()

    if args.router:
        settings.router_backend = args.router
    if args.kb_dir:
        from pathlib import Path

        settings.kb_dir = Path(args.kb_dir).resolve()

    init_db()

    from .models.factory import build_router

    router = build_router(settings, load=settings.router_backend == "laya")
    print(f"router: {router.health()}")

    from .models.factory import set_router

    set_router(router)

    from .rag.retriever import is_index_built

    if is_index_built(settings):
        from .rag.retriever import get_retriever

        print(f"knowledge base: {get_retriever().health()}")
    else:
        print(f"no index at {settings.kb_index_dir}; run python -m scripts.build_kb")

    from langgraph.checkpoint.sqlite import SqliteSaver

    thread_id = args.thread or f"cli_{uuid.uuid4().hex[:8]}"
    print(f"thread: {thread_id}\n{BANNER}\n")

    with SqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as saver:
        graph = build_graph(checkpointer=saver, settings=settings)
        last: dict[str, Any] = {}

        while True:
            message = ask()
            if not message:
                continue
            if message in ("exit", "quit", ":q"):
                break
            if message == "trace":
                for entry in (last.get("trace") or []):
                    print(f"     {entry['node']:<16} {entry['duration_ms']:>8.1f} ms")
                continue

            from langchain_core.messages import HumanMessage
            from langgraph.types import Command

            state = new_state(thread_id, args.customer, message)
            state["messages"] = [HumanMessage(content=message)]

            config = {"configurable": {"thread_id": thread_id}, "run_id": new_run_id()}
            result = graph.invoke(state, config)

            interrupt = _interrupt_of(result)
            if interrupt is not None:
                _handle_interrupt(graph, thread_id, interrupt, config)
                result = graph.invoke(Command(resume="approved"), config)

            show(result)
            last = result
            print()

    close_connection()


def _interrupt_of(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", interrupts[0])
    return value if isinstance(value, dict) else None


def _handle_interrupt(graph, thread_id: str, payload: dict[str, Any], config: dict) -> None:
    """Ask the operator for the decision the graph is waiting on."""
    from .api.responses import STATUS_FOR_MODE

    print(f"bot> {STATUS_FOR_MODE.get(payload.get('mode'), 'awaiting_approval')}: {payload}")
    answer = ask("approve? [y/N]> ")
    if answer.strip().lower() in ("y", "yes", "approve", "approved"):
        graph.invoke(
            __import__("langgraph.types", fromlist=["Command"]).Command(resume="approved"), config
        )
    else:
        graph.invoke(
            __import__("langgraph.types", fromlist=["Command"]).Command(resume="rejected"), config
        )


if __name__ == "__main__":
    main()
