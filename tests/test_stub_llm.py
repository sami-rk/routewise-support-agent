"""Tests for the stub LLM itself.

The stub is load-bearing for the offline suite, so its two useful properties are
pinned here: it returns what a test scripted, and it refuses to invent an answer
when a test forgot to script one.
"""

from __future__ import annotations

import pytest

from tests.stub_llm import ScriptedLLM, StubChatModel, StubMessage


def test_returns_scripted_replies_in_order() -> None:
    model = StubChatModel(["first", "second"])
    assert model.invoke("x").content == "first"
    assert model.invoke("x").content == "second"


def test_unused_scripts_are_not_an_error() -> None:
    model = StubChatModel(["only one"])
    assert model.invoke("x").content == "only one"


def test_running_out_of_replies_is_an_error() -> None:
    model = StubChatModel(["one"])
    model.invoke("x")
    with pytest.raises(AssertionError, match="no scripted reply"):
        model.invoke("x")


def test_default_is_used_once_the_queue_empties() -> None:
    model = StubChatModel(["one"], default="fallback")
    assert model.invoke("x").content == "one"
    assert model.invoke("x").content == "fallback"
    assert model.invoke("x").content == "fallback"


def test_dict_replies_carry_tool_calls() -> None:
    model = StubChatModel([{"content": "", "tool_calls": [{"name": "get_subscription", "args": {}}]}])
    message = model.invoke("x")
    assert message.tool_calls[0]["name"] == "get_subscription"
    assert message.tool_calls[0]["args"] == {}


def test_tool_call_ids_are_generated() -> None:
    model = StubChatModel([{"content": "", "tool_calls": [{"name": "a"}, {"name": "b"}]}])
    message = model.invoke("x")
    assert message.tool_calls[0]["id"] != message.tool_calls[1]["id"]


def test_messages_pass_through() -> None:
    original = StubMessage("already a message")
    assert StubChatModel([original]).invoke("x") is original


def test_prompts_are_recorded() -> None:
    model = StubChatModel(["a", "b"])
    model.invoke("first prompt")
    model.invoke("second prompt")
    assert model.prompts == ["first prompt", "second prompt"]


def test_bind_tools_returns_a_stub_that_remembers_the_tools() -> None:
    model = StubChatModel(["ok"])
    bound = model.bind_tools(["a_tool", "b_tool"])
    assert bound.bound_tools == ["a_tool", "b_tool"]
    assert bound.invoke("x").content == "ok"


def test_scripted_context_manager_installs_and_restores() -> None:
    from app.models import llm_runner

    original = llm_runner.get_llm_chain
    with ScriptedLLM(["from the stub"]) as model:
        assert llm_runner.get_llm_chain is not original
        chain = llm_runner.get_llm_chain("faq", None)
        assert chain[0][0] == "stub-model"
        assert chain[0][1] is model
    assert llm_runner.get_llm_chain is original


def test_context_manager_restores_even_after_an_error() -> None:
    from app.models import llm_runner

    original = llm_runner.get_llm_chain
    with pytest.raises(RuntimeError):
        with ScriptedLLM(["x"]):
            raise RuntimeError("boom")
    assert llm_runner.get_llm_chain is original
