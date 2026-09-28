"""Tests for reading an action out of a model reply.

Free models on OpenRouter emit tool calls as text often enough that the parser
has to cope with it, and the customer must never see that JSON.
"""

from __future__ import annotations

import pytest

from app.agent.nodes.agents import (
    _proposed_action,
    strip_action_line,
    strip_tool_call_json,
)


def proposal(kind: str, arguments: dict):
    """A ToolMessage-shaped object carrying a tool call."""
    from langchain_core.messages import ToolMessage

    name = {"refund": "propose_refund", "cancel": "propose_cancellation"}[kind]
    return ToolMessage(content="ok", tool_call_id="1", name=name, artifact=arguments)


class TestStructuredToolCall:
    def test_a_refund_proposal_is_read(self) -> None:
        action = _proposed_action("", [proposal("refund", {"invoice_id": "inv_1", "amount": 19.0})])
        assert action["type"] == "refund"
        assert action["invoice_id"] == "inv_1"
        assert action["amount"] == 19.0

    def test_a_cancellation_proposal_is_read(self) -> None:
        assert _proposed_action("", [proposal("cancel", {})])["type"] == "cancel"


class TestToolCallAsText:
    """What the free models actually emit."""

    @pytest.mark.parametrize(
        "text",
        [
            '{"tool": "propose_refund", "arguments": {"invoice_id": "inv_1", "amount": 19.0}}',
            '{"name": "propose_refund", "args": {"invoice_id": "inv_1", "amount": 19.0}}',
            '{"name": "propose_refund", "arguments": {"invoice_id": "inv_1", "amount": 19.0}}',
            'proposing now: {"tool": "propose_refund", "arguments": {"invoice_id": "inv_1", "amount": 19.0}}',
        ],
    )
    def test_the_common_shapes_are_understood(self, text: str) -> None:
        action = _proposed_action(text, [])
        assert action is not None
        assert action["type"] == "refund"
        assert action["invoice_id"] == "inv_1"

    def test_arguments_arriving_as_a_json_string_are_parsed(self) -> None:
        text = '{"tool": "propose_refund", "arguments": "{\\"invoice_id\\": \\"inv_9\\", \\"amount\\": 5}"}'
        action = _proposed_action(text, [])
        assert action is not None
        assert action["invoice_id"] == "inv_9"

    def test_the_openai_function_shape_is_understood(self) -> None:
        text = '{"function": {"name": "propose_cancellation", "arguments": {}}}'
        assert _proposed_action(text, [])["type"] == "cancel"

    def test_plain_prose_is_not_an_action(self) -> None:
        assert _proposed_action("I have refunded the $19.00 to your card.", []) is None

    def test_unrelated_json_is_ignored(self) -> None:
        assert _proposed_action('Here is the data: {"invoice_id": "inv_1", "amount": 19}', []) is None

    def test_a_hallucinated_money_tool_is_refused(self) -> None:
        # Only the proposal tools count. A model inventing a direct refund call
        # gets nothing, because that tool is not in its schema.
        text = '{"tool": "create_refund", "arguments": {"invoice_id": "inv_1", "amount": 19.0}}'
        assert _proposed_action(text, []) is None

    def test_a_refund_proposal_without_an_invoice_is_refused(self) -> None:
        # The gate needs the invoice; a proposal with no details is unusable and
        # would be refused by the policy anyway.
        text = '{"tool": "propose_refund", "arguments": {"amount": 19.0}}'
        assert _proposed_action(text, []) is None

    def test_broken_json_is_ignored(self) -> None:
        assert _proposed_action('{"tool": "propose_refund", broken}', []) is None

    def test_a_structured_call_wins_over_text(self) -> None:
        action = _proposed_action(
            '{"tool": "propose_refund", "arguments": {"invoice_id": "from_text", "amount": 1}}',
            [proposal("refund", {"invoice_id": "from_tool", "amount": 2})],
        )
        assert action["invoice_id"] == "from_tool"


class TestLegacyActionLine:
    def test_the_action_line_still_works(self) -> None:
        action = _proposed_action(
            'Done.\nACTION: {"type": "refund", "invoice_id": "inv_1", "amount": 19.0}', []
        )
        assert action["type"] == "refund"


class TestStripping:
    def test_tool_call_json_is_removed_from_the_reply(self) -> None:
        text = '{"tool": "propose_refund", "arguments": {"invoice_id": "inv_1", "amount": 19.0}}'
        assert strip_tool_call_json(text).strip() == ""

    def test_normal_prose_survives_stripping(self) -> None:
        text = "Your refund of $19.00 is on its way."
        assert strip_tool_call_json(text) == text

    def test_the_action_line_is_removed(self) -> None:
        assert strip_action_line('Sure.\nACTION: {"type": "cancel"}') == "Sure."
