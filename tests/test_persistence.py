"""A thread survives the process that was serving it.

This is the first acceptance criterion: restarting the server and sending another
message with the same `thread_id` continues the conversation. The saver is closed
and reopened between the two turns here, which is the same thing a restart does
to the connection.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agent.graph import build_graph, thread_config
from app.db.session import close_connection, query_all
from tests.conftest import user_turn
from tests.stub_llm import ScriptedLLM

THREAD = "restart-1"


def test_the_conversation_continues_after_a_restart(seeded, knowledge_base) -> None:
    db = seeded

    # First "process": one turn, then the connection is closed.
    with SqliteSaver.from_conn_string(str(db.checkpoint_db_path)) as saver:
        graph = build_graph(checkpointer=saver, settings=db)
        with ScriptedLLM(["The Pro plan is $19 per month."]):
            first = graph.invoke(
                user_turn("How much is the Pro plan?", thread_id=THREAD), thread_config(THREAD)
            )
    assert first["response"]
    close_connection()

    # Second "process": a new saver over the same file.
    with SqliteSaver.from_conn_string(str(db.checkpoint_db_path)) as saver:
        graph = build_graph(checkpointer=saver, settings=db)
        history = graph.get_state(thread_config(THREAD))
        assert history.values["user_input"] == "How much is the Pro plan?"

        with ScriptedLLM(["It is $19 per month for 1 TB."]):
            second = graph.invoke(
                user_turn("And how much storage?", thread_id=THREAD), thread_config(THREAD)
            )

    assert second["response"]
    # Both turns are in the conversation, not just the second.
    contents = [str(m.content) for m in second["messages"]]
    assert "How much is the Pro plan?" in contents
    assert "And how much storage?" in contents


def test_an_interrupted_thread_survives_a_restart(seeded, knowledge_base) -> None:
    """A refund paused for approval can be approved by a different process."""
    from langgraph.types import Command

    db = seeded
    proposal = {
        "content": "",
        "tool_calls": [
            {
                "name": "propose_refund",
                "args": {"invoice_id": "inv_barbara_current", "amount": 49.0, "reason": "asked"},
            }
        ],
    }

    with SqliteSaver.from_conn_string(str(db.checkpoint_db_path)) as saver:
        graph = build_graph(checkpointer=saver, settings=db)
        with ScriptedLLM([proposal, "with our team"]):
            paused = graph.invoke(
                user_turn("refund the 49 dollars", customer_id="cus_barbara", thread_id="restart-2"),
                thread_config("restart-2"),
            )
    assert paused.get("__interrupt__")
    assert query_all("SELECT * FROM refunds WHERE invoice_id = 'inv_barbara_current'") == []
    close_connection()

    with SqliteSaver.from_conn_string(str(db.checkpoint_db_path)) as saver:
        graph = build_graph(checkpointer=saver, settings=db)
        with ScriptedLLM(["Your refund is on its way."]):
            done = graph.invoke(Command(resume="approved"), thread_config("restart-2"))

    assert done["action_result"]["ok"] is True
    rows = query_all("SELECT * FROM refunds WHERE invoice_id = 'inv_barbara_current'")
    assert len(rows) == 1


def test_traces_survive_a_restart(seeded, knowledge_base) -> None:
    db = seeded
    with SqliteSaver.from_conn_string(str(db.checkpoint_db_path)) as saver:
        graph = build_graph(checkpointer=saver, settings=db)
        with ScriptedLLM(["The Pro plan is $19."]):
            graph.invoke(user_turn("How much is the Pro plan?", thread_id="restart-3"), thread_config("restart-3"))
    close_connection()

    from app.observability.tracing import get_thread_trace

    nodes = [row["node"] for row in get_thread_trace("restart-3")]
    assert nodes[0] == "load_context"
    assert "laya_router" in nodes
    assert nodes[-1] == "update_memory"
