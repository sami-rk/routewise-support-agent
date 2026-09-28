"""Small helpers shared by the node modules.

Kept in one place so the nodes themselves stay about routing and prompting
rather than message plumbing.
"""

from __future__ import annotations

from typing import Any

from ...models.llm import BUSY_MESSAGE


def message_text(response: Any) -> str:
    """The text of a model reply, whatever shape it arrived in."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Some providers return a list of content blocks.
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return " ".join(part for part in parts if part).strip()
    return str(content or "").strip()


def tool_result_message(call: dict[str, Any], content: str) -> Any:
    """A `ToolMessage` carrying one tool's result back to the model.

    The call's arguments ride along in `artifact`, so a node can read what a tool
    was asked to do without parsing the conversation. That is how a refund
    proposal reaches the approval gate.
    """
    from langchain_core.messages import ToolMessage

    return ToolMessage(
        content=content,
        tool_call_id=call.get("id", "call_0"),
        name=call.get("name", "tool"),
        artifact=dict(call.get("args") or {}),
    )


def friendly_error(exc: BaseException) -> str:
    """A message safe and useful to show a customer.

    A rate limit becomes a "we are busy" line; anything else becomes a plain
    apology without a traceback, an HTTP status, or a model name.
    """
    from ...models.llm import is_rate_limited

    if is_rate_limited(exc):
        return BUSY_MESSAGE
    return (
        "Something went wrong on my side handling that, and I would rather say so "
        "than guess. Please try again in a moment."
    )


def cited(text: str, citations: list[str]) -> str:
    """Append the sources used, if the reply has none already.

    The prompt asks the model to cite inline, which weaker models often skip, so
    this guarantees the customer can see where the facts came from.
    """
    text = (text or "").strip()
    if not citations:
        return text
    if any(f"[{c}" in text or c in text for c in citations):
        return text
    return f"{text}\n\nSources: {', '.join(citations)}"
