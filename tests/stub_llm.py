"""A stub chat model, so the whole test suite runs with no OpenRouter key.

It implements the two methods the nodes use — `invoke` and `bind_tools` — and
answers from a queue of scripted replies, so a test can say "the model will
propose a refund" and assert the graph then pauses for approval. Any call the
test did not script raises, which is better than a plausible-looking default
that hides a missing case.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


class StubMessage:
    """A reply shaped like a LangChain message."""

    def __init__(self, content: str, tool_calls: Sequence[dict[str, Any]] = ()) -> None:
        self.content = content
        self.type = "ai"
        self.tool_calls = [
            {"name": call["name"], "args": call.get("args", {}), "id": call.get("id", f"call_{i}")}
            for i, call in enumerate(tool_calls)
        ]
        self.usage_metadata = {"input_tokens": 12, "output_tokens": 6}
        self.response_metadata = {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StubMessage({self.content!r}, tool_calls={self.tool_calls!r})"


class StubChatModel:
    """A scripted stand-in for `ChatOpenAI`."""

    def __init__(
        self,
        replies: Sequence[str | dict[str, Any] | StubMessage] | None = None,
        *,
        default: str | StubMessage | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        self.queue: list[Any] = list(replies or [])
        self.default = default
        self.bound_tools = list(tools or [])
        self.prompts: list[Any] = []
        self.invocations = 0

    def bind_tools(self, tools: Sequence[Any]) -> "StubChatModel":
        """Tool binding returns another stub that remembers the tools."""
        clone = StubChatModel(self.queue, default=self.default, tools=list(tools))
        clone.prompts = self.prompts
        clone.default = self.default
        return clone

    def invoke(self, prompt: Any, **kwargs: Any) -> StubMessage:
        self.invocations += 1
        self.prompts.append(prompt)
        if not self.queue:
            if self.default is not None:
                return self._as_message(self.default)
            raise AssertionError(
                f"StubChatModel was called {self.invocations} time(s) with no scripted reply left. "
                "Script the reply the node should produce."
            )
        return self._as_message(self.queue.pop(0))

    def _as_message(self, reply: Any) -> StubMessage:
        if isinstance(reply, StubMessage):
            return reply
        if isinstance(reply, str):
            return StubMessage(reply)
        if isinstance(reply, dict):
            return StubMessage(reply.get("content", ""), reply.get("tool_calls", ()))
        raise TypeError(f"Cannot use {type(reply).__name__} as a stub reply")


class ScriptedLLM:
    """Installs a `StubChatModel` in place of the real factory for one test.

    Used as a context manager, so the override cannot leak into another test:

        with ScriptedLLM(["Here is the answer"]):
            graph.invoke(...)
    """

    def __init__(
        self,
        replies: Sequence[Any] | None = None,
        *,
        default: Any = None,
    ) -> None:
        self.model = StubChatModel(replies, default=default)
        self._original: Callable[..., Any] | None = None

    def __enter__(self) -> StubChatModel:
        from app.models import llm_runner

        self._original = llm_runner.get_llm_chain
        stub = self.model

        def fake_chain(*args: Any, **kwargs: Any) -> list[tuple[str, Any]]:
            return [("stub-model", stub)]

        llm_runner.get_llm_chain = fake_chain  # type: ignore[assignment]
        return stub

    def __exit__(self, *exc: Any) -> None:
        from app.models import llm_runner

        if self._original is not None:
            llm_runner.get_llm_chain = self._original  # type: ignore[assignment]
