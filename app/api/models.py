"""Pydantic request and response models for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """`POST /chat` and `POST /chat/stream`."""

    thread_id: str | None = Field(default=None, description="Omit to start a new thread.")
    customer_id: str = Field(description="Which customer this is about.")
    message: str = Field(min_length=1, description="The customer's message.")


class ConfirmRequest(BaseModel):
    """`POST /chat/confirm`, for a cancellation waiting on the customer."""

    thread_id: str
    confirmed: bool = Field(description="True to go ahead with the cancellation.")
    note: str | None = None


class ApprovalDecision(BaseModel):
    """`POST /approvals/{thread_id}`."""

    decision: Literal["approved", "rejected"]
    note: str | None = None
    approved_by: str | None = Field(default=None, description="Recorded on the refund.")


class ChatResponse(BaseModel):
    """What `POST /chat` returns.

    `status` is the important field: `done` for a finished turn, and
    `awaiting_approval` or `awaiting_confirmation` when the graph paused and
    needs a decision, in which case `interrupt` says what is being asked.
    """

    thread_id: str
    status: Literal["done", "awaiting_approval", "awaiting_confirmation"] = "done"
    response: str = ""
    intent: str | None = None
    intent_conf: float | None = None
    escalated: bool = False
    ticket_id: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    interrupt: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
