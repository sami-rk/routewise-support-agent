"""Chat endpoints: send a message, resume a thread, and stream the reply.

A turn can finish in two ways. Either the graph runs to the end and the reply
comes back, or it interrupts — for a refund that needs staff approval, or a
cancellation that needs the customer to agree — and the response says so and
carries the payload. Resuming is a separate call, which is the point: a person
gets to see the request before any money moves.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from ..agent.state import new_state
from .deps import config_for, graph_dep
from .models import ChatRequest, ChatResponse, ConfirmRequest
from .responses import citations_of, to_chat_response

router = APIRouter(tags=["chat"])


def _thread_id(requested: str | None) -> str:
    return requested or f"thread_{uuid.uuid4().hex[:12]}"


def _input_state(request: ChatRequest, thread_id: str) -> dict[str, Any]:
    """A starting state that carries the new message in `messages`.

    `add_messages` appends to the thread's history from the checkpointer, so a
    resumed conversation keeps its turns without this having to read them back.
    """
    from langchain_core.messages import HumanMessage

    state = new_state(thread_id, request.customer_id, request.message)
    state["messages"] = [HumanMessage(content=request.message)]
    return state


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest, graph=Depends(graph_dep)) -> ChatResponse:
    """Send a message and get the reply, or an interrupt waiting on a decision."""
    thread_id = _thread_id(body.thread_id)
    result = graph.invoke(_input_state(body, thread_id), config_for(thread_id))
    return to_chat_response(thread_id, result)


@router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest, graph=Depends(graph_dep)) -> StreamingResponse:
    """The same turn, streamed as server-sent events.

    Emits, in order: the routing decision, the node trace, and finally either the
    reply or the interrupt. Tokens are not streamed from the model: on the free
    tier the models are small and slow enough that per-token streaming buys little
    and costs a held connection, so the useful events are the decision and the
    node timeline.
    """
    thread_id = _thread_id(body.thread_id)

    async def events() -> AsyncIterator[bytes]:
        def send(event: str, data: dict[str, Any]) -> bytes:
            return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()

        # The checkpointer is the synchronous `SqliteSaver`, which has no async
        # methods, so the graph is streamed from a worker thread rather than on
        # the event loop. Anything awaited here would block every other request.
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def run() -> None:
            final: dict[str, Any] = {}
            try:
                for update in graph.stream(
                    _input_state(body, thread_id), config_for(thread_id), stream_mode="updates"
                ):
                    for node, patch in (update or {}).items():
                        if not isinstance(patch, dict):
                            continue
                        final.update(patch)
                        queue.put_nowait(send("node", {"node": node, "keys": sorted(patch)}))
                        if "intent" in patch:
                            queue.put_nowait(
                                send(
                                    "routing",
                                    {
                                        "intent": patch.get("intent"),
                                        "intent_conf": patch.get("intent_conf"),
                                        "urgency_band": patch.get("urgency_band"),
                                    },
                                )
                            )
                        if patch.get("response"):
                            queue.put_nowait(send("delta", {"text": patch["response"]}))
                queue.put_nowait(send("done", to_chat_response(thread_id, final).model_dump()))
            except Exception as exc:  # noqa: BLE001
                queue.put_nowait(send("error", {"message": str(exc)}))
            finally:
                queue.put_nowait(sentinel)

        # Started before the loop, so the worker is running while the events are
        # drained; awaiting it afterwards would deadlock on the first `get()`.
        future = asyncio.get_running_loop().run_in_executor(None, run)
        while True:
            event = await queue.get()
            if event is sentinel:
                break
            yield event
        await future

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/confirm", response_model=ChatResponse)
def confirm(body: ConfirmRequest, graph=Depends(graph_dep)) -> ChatResponse:
    """Answer a cancellation that is waiting on the customer.

    Only `customer_confirm` interrupts are settled here; a staff approval must go
    through `POST /approvals/{thread_id}` so it is recorded against whoever made
    it.
    """
    interrupt = interrupt_of_snapshot(graph.get_state(config_for(body.thread_id)))
    if interrupt is None or interrupt.get("mode") != "customer_confirm":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This thread is not waiting for a customer confirmation.",
        )

    decision = "approved" if body.confirmed else "rejected"
    note = body.note or ("customer confirmed" if body.confirmed else "customer declined")
    result = graph.invoke(
        Command(resume={"decision": decision, "note": note, "approved_by": "customer"}),
        config_for(body.thread_id),
    )
    return to_chat_response(body.thread_id, result)


@router.get("/threads/{thread_id}")
def thread_history(thread_id: str, graph=Depends(graph_dep)) -> dict[str, Any]:
    """The conversation so far, read back from the checkpointer."""
    snapshot = graph.get_state(config_for(thread_id))
    messages = [
        {
            "role": getattr(message, "type", "unknown"),
            "content": getattr(message, "content", str(message)),
        }
        for message in (snapshot.values.get("messages") or [])
    ]
    pending = interrupt_of_snapshot(snapshot)
    return {
        "thread_id": thread_id,
        "messages": messages,
        "response": snapshot.values.get("response") or "",
        "intent": snapshot.values.get("intent"),
        "ticket_id": snapshot.values.get("ticket_id"),
        "awaiting": pending,
        "citations": citations_of(snapshot.values),
    }


def interrupt_of_snapshot(snapshot: Any) -> dict[str, Any] | None:
    """The interrupt a thread is paused on, if any."""
    for task in getattr(snapshot, "tasks", []) or []:
        interrupts = getattr(task, "interrupts", None) or ()
        if interrupts:
            value = getattr(interrupts[0], "value", None)
            return value if isinstance(value, dict) else {"value": value}
    return None
