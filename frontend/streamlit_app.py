"""CloudSync Pro support console.

Four pages, all of them thin over the API:

* **Chat** — talk to the agent as a customer, with the detected intent, the tools
  that ran, the knowledge base sources and an `[ESCALATED]` badge under each
  reply.
* **Approvals** — refunds waiting for a staff decision.
* **Tickets** — the queue, with status and priority.
* **Metrics** — KPI cards, the intent distribution and a per-node timeline.

Run it with the API already up:

    uvicorn app.main:app --port 8000
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from api_client import (
    ApiError,
    chat,
    confirm,
    decide_approval,
    health,
    metrics,
    pending_approvals,
    set_ticket_status,
    tickets,
    traces,
)

st.set_page_config(page_title="CloudSync Pro support console", page_icon="☁️", layout="wide")

DEMO_CUSTOMERS = {
    "cus_ada": "Ada Lovelace (Pro, 3/5 devices, charge 5 days ago)",
    "cus_grace": "Grace Hopper (Basic, charge 3 days ago)",
    "cus_alan": "Alan Turing (Business, charge 20 days ago, outside the window)",
    "cus_barbara": "Barbara Liskov (Business, $49 charge, needs staff approval)",
    "cus_margaret": "Margaret Hamilton (Basic, charge 1 day ago)",
    "cus_edsger": "Edsger Dijkstra (Pro, charge 30 days ago)",
}


def sidebar() -> tuple[str, str]:
    """The sidebar. Returns the chosen page and the customer to act as."""
    with st.sidebar:
        st.header("Console")
        page = st.radio("Page", ["Chat", "Approvals", "Tickets", "Metrics"])
        st.divider()
        try:
            status = health()
            st.caption(f"backend: {status.get('status')}")
            router = status.get("router", {})
            st.caption(f"router: {router.get('backend', '?')} ({router.get('model', '?')})")
            kb = status.get("knowledge_base", {})
            st.caption(f"knowledge base: {kb.get('passages', 0)} passages")
            if not status.get("llm_key_present"):
                st.warning("No OPENROUTER_API_KEY: replies will be the busy message.")
        except ApiError as exc:
            st.error(str(exc))

        st.divider()
        st.caption("Acting as")
        options = list(DEMO_CUSTOMERS)
        customer_id = st.selectbox(
            "Customer", options, format_func=lambda c: DEMO_CUSTOMERS[c], key="customer_pick"
        )
        st.caption("These are the seeded demo customers.")
    return page, customer_id


def chips(reply: dict[str, Any]) -> None:
    """The small labels under a reply: intent, tools, sources."""
    column = st.columns([2, 2, 3])
    with column[0]:
        confidence = reply.get("intent_conf")
        label = f"intent `{reply.get('intent') or 'n/a'}`"
        if confidence is not None:
            label += f" {float(confidence):.0%}"
        st.caption(label)
    with column[1]:
        tools = reply.get("tools_used") or []
        st.caption("tools: " + (", ".join(tools) if tools else "none"))
    with column[2]:
        sources = reply.get("citations") or []
        st.caption("sources: " + (", ".join(s.split("›")[0].strip(" [") for s in sources) if sources else "none"))


def render_reply(reply: dict[str, Any]) -> None:
    """One agent turn: the text, the chips, and any badge."""
    st.markdown(reply.get("response") or "")
    if reply.get("escalated"):
        st.error(f"[ESCALATED] handed to a person · ticket `{reply.get('ticket_id')}`")
    chips(reply)


def page_chat() -> None:
    customer_id = st.session_state["customer_id"]
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.subheader("Chat")
    st.caption(DEMO_CUSTOMERS.get(customer_id, customer_id))

    if st.session_state.messages and st.button("New conversation"):
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    for turn in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            render_reply(turn)

    message = st.chat_input("Ask about your account")
    if not message:
        render_pending()
        return

    st.session_state.messages.append({"user": message, "reply": {"response": "", "citations": []}})
    with st.chat_message("user"):
        st.markdown(message)
    with st.chat_message("assistant"):
        try:
            result = chat(customer_id, message, st.session_state.thread_id)
        except ApiError as exc:
            st.error(str(exc))
            return

        st.session_state.thread_id = result.get("thread_id")

        if result.get("status") == "awaiting_confirmation":
            payload = result.get("interrupt") or {}
            st.session_state.pending = {"question": payload.get("question", "Shall I go ahead?")}
            st.info("I need your confirmation before I cancel anything.")
        elif result.get("status") == "awaiting_approval":
            payload = result.get("interrupt") or {}
            # A staff decision is made in the Approvals page, not here.
            st.info(
                f"Waiting for staff approval of a ${payload.get('amount', 0):.2f} refund. "
                "See the Approvals page."
            )
        else:
            render_reply(result)
        st.session_state.messages[-1]["reply"] = result

    # Rendered after the handler, not before: a Streamlit run that sets the
    # pending state would otherwise leave the customer looking at nothing until
    # their next interaction.
    render_pending()


def render_pending() -> None:
    """The yes/no prompt for a cancellation waiting on the customer."""
    pending = st.session_state.get("pending")
    if not pending:
        return
    st.warning(pending["question"])
    left, right = st.columns(2)
    if left.button("Yes, go ahead", type="primary"):
        result = confirm(st.session_state.thread_id, True)
        st.session_state.messages[-1]["reply"] = result
        st.session_state.pending = None
        st.rerun()
    if right.button("No, leave it"):
        result = confirm(st.session_state.thread_id, False)
        st.session_state.messages[-1]["reply"] = result
        st.session_state.pending = None
        st.rerun()


def page_approvals() -> None:
    st.subheader("Approvals")
    st.caption("Refunds above the automatic limit, waiting for a person.")
    try:
        rows = pending_approvals()
    except ApiError as exc:
        st.error(str(exc))
        return

    if not rows:
        st.info("Nothing is waiting for approval.")
        return

    for row in rows:
        with st.container(border=True):
            try:
                payload = json.loads(row["payload"])
            except (ValueError, TypeError):
                payload = {}
            st.write(f"**{payload.get('amount', '?')} refund** on `{payload.get('invoice_id', '?')}`")
            st.caption(
                f"thread `{row['thread_id']}` · customer `{row['customer_id']}` · mode `{row['mode']}`"
            )
            if payload.get("eligibility"):
                st.caption(payload["eligibility"].get("reason", ""))
            left, middle, right = st.columns([1, 1, 3])
            if left.button("Approve", key=f"approve-{row['thread_id']}"):
                try:
                    decide_approval(row["thread_id"], "approved", "approved in the console")
                    st.success("Approved.")
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))
            if middle.button("Reject", key=f"reject-{row['thread_id']}"):
                try:
                    decide_approval(row["thread_id"], "rejected", "rejected in the console")
                    st.warning("Rejected.")
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))
            with right:
                st.text_input("Note", key=f"note-{row['thread_id']}", label_visibility="collapsed")


def page_tickets() -> None:
    st.subheader("Tickets")
    status_filter = st.selectbox("Status", ["all", "open", "resolved", "closed"])
    try:
        rows = tickets(None if status_filter == "all" else status_filter)
    except ApiError as exc:
        st.error(str(exc))
        return

    if not rows:
        st.info("No tickets.")
        return

    st.dataframe(
        [
            {
                "id": row["id"],
                "customer": row["customer_id"],
                "subject": row["subject"],
                "priority": row["priority"],
                "category": row["category"],
                "status": row["status"],
                "created": row["created_at"],
            }
            for row in rows
        ],
        use_container_width=True,
    )

    selected = st.selectbox("Open a ticket", [row["id"] for row in rows])
    ticket = next(row for row in rows if row["id"] == selected)
    with st.container(border=True):
        st.write(f"**{ticket['subject']}**")
        st.caption(ticket.get("body") or "(no description)")
        new_status = st.selectbox("Set status", ["open", "in_progress", "resolved", "closed"], key=f"st-{selected}")
        if st.button("Update", key=f"up-{selected}"):
            try:
                set_ticket_status(selected, new_status)
                st.success("Updated.")
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))


def page_metrics() -> None:
    st.subheader("Metrics and traces")
    try:
        data = metrics()
    except ApiError as exc:
        st.error(str(exc))
        return

    conversation = data.get("conversations", {})
    outcomes = data.get("outcomes", {})
    llm = data.get("llm", {})

    first, second, third, fourth = st.columns(4)
    first.metric("Conversations", conversation.get("threads", 0))
    second.metric("Turns", conversation.get("turns", 0))
    third.metric("Avg latency", f"{conversation.get('avg_latency_ms', 0):.0f} ms")
    fourth.metric("p95 latency", f"{conversation.get('p95_latency_ms', 0):.0f} ms")

    first, second, third = st.columns(3)
    first.metric("Auto-resolution", f"{outcomes.get('auto_resolution_rate', 0):.0%}")
    second.metric("Escalation rate", f"{outcomes.get('escalation_rate', 0):.0%}")
    third.metric("Router confidence", f"{data.get('routing', {}).get('avg_intent_confidence', 0):.0%}")

    first, second, third = st.columns(3)
    first.metric("LLM requests", llm.get("total_requests", 0))
    second.metric("Rate limited", llm.get("rate_limited", 0))
    third.metric("Fallbacks", llm.get("fallbacks", 0))

    st.subheader("Intent distribution")
    distribution = data.get("routing", {}).get("intent_distribution", {})
    if distribution:
        st.bar_chart(distribution)
    else:
        st.caption("No routing decisions recorded yet.")

    st.subheader("LLM requests per model")
    per_model = llm.get("requests_per_model", {})
    if per_model:
        st.bar_chart(per_model)
    else:
        st.caption("No LLM calls recorded yet.")

    st.subheader("Node timeline")
    thread_id = st.text_input("Thread id", value=st.session_state.get("thread_id") or "")
    if thread_id and st.button("Load trace"):
        try:
            data = traces(thread_id)
        except ApiError as exc:
            st.error(str(exc))
            return
        if not data.get("trace"):
            st.info("No traces for that thread.")
            return
        st.dataframe(
            [
                {
                    "node": row["node"],
                    "ms": row["duration_ms"],
                    "model": row.get("model") or "",
                    "input": (row.get("input_summary") or "")[:120],
                    "error": row.get("error") or "",
                }
                for row in data["trace"]
            ],
            use_container_width=True,
        )
        decisions = data.get("router_decisions", [])
        if decisions:
            st.caption(
                "Router: "
                + ", ".join(
                    f"{d.get('answers', {}).get('intent')} "
                    f"({float(d.get('answers', {}).get('intent_conf') or 0):.0%})"
                    for d in decisions
                )
            )


def main() -> None:
    page, customer_id = sidebar()
    st.session_state["customer_id"] = customer_id

    if page == "Chat":
        page_chat()
    elif page == "Approvals":
        page_approvals()
    elif page == "Tickets":
        page_tickets()
    else:
        page_metrics()


if __name__ == "__main__":
    main()
