"""Routing decisions and the `RouterModel` interface.

Two implementations satisfy the interface: `LayaRouter` (the local decision
model, see the rest of this file) and `FakeRouter` (keywords, for tests and for
machines that cannot run Laya). Everything downstream — the graph, the evals,
the API — only ever sees a `RouterDecision`, so the backend is swappable
through `ROUTER_BACKEND`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

# The intents the graph routes on. Kept as a tuple so its order is stable in
# eval reports and in the confusion matrix.
INTENTS = (
    "product_info",
    "pricing",
    "billing",
    "refund",
    "cancellation",
    "technical",
    "account",
    "smalltalk",
)

# Ordered urgency scale. Laya answers a score question with an expected value
# over these positions, which is mapped back onto 0..1.
URGENCY_LEVELS = ("not urgent", "low", "medium", "high", "critical")

# Above this probability on `needs_human`, escalate immediately.
NEEDS_HUMAN_THRESHOLD = 0.7
# Above this probability on `injection`, answer with the safe response.
INJECTION_THRESHOLD = 0.7


@dataclass
class RouterDecision:
    """One turn's routing decision, with the probabilities that produced it."""

    intent: str | None = None
    # Top probability for `intent`. This is the number the confidence gate uses,
    # not Laya's entropy-based `confidence` field.
    intent_conf: float | None = None
    # Full distribution over intents, so evals can compute calibration and the
    # UI can explain the choice.
    intent_probs: dict[str, float] = field(default_factory=dict)

    # 0..1 position on URGENCY_LEVELS.
    urgency: float | None = None
    urgency_band: str | None = None

    frustrated: bool = False
    frustrated_conf: float | None = None
    needs_human: bool = False
    needs_human_conf: float | None = None
    injection: bool = False
    injection_conf: float | None = None

    # Which checkpoint answered, and how long the decision took.
    model: str | None = None
    latency_ms: float | None = None
    # The backend that produced this decision: "laya" or "fake".
    backend: str = "fake"
    # The untouched answer payload, for traces and evals.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_unsafe(self) -> bool:
        """True when the message looks like a prompt-injection attempt."""
        return self.injection and (self.injection_conf or 0.0) > INJECTION_THRESHOLD

    @property
    def wants_human(self) -> bool:
        """True when a human is required, above the confidence threshold."""
        return self.needs_human and (self.needs_human_conf or 0.0) > NEEDS_HUMAN_THRESHOLD

    @property
    def is_angry(self) -> bool:
        return bool(self.frustrated)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, minus the raw payload's bulk."""
        data = asdict(self)
        data.pop("raw", None)
        return data

    def answers_for_log(self) -> dict[str, Any]:
        """What the router actually answered, for the `router_decisions` table."""
        return {
            "intent": self.intent,
            "intent_conf": self.intent_conf,
            "urgency": self.urgency,
            "urgency_band": self.urgency_band,
            "frustrated": self.frustrated,
            "frustrated_conf": self.frustrated_conf,
            "needs_human": self.needs_human,
            "needs_human_conf": self.needs_human_conf,
            "injection": self.injection,
            "injection_conf": self.injection_conf,
            "model": self.model,
            "backend": self.backend,
            "latency_ms": self.latency_ms,
        }


@runtime_checkable
class RouterModel(Protocol):
    """What the graph needs from a router."""

    name: str

    def predict(self, state: str) -> RouterDecision:
        """Route one customer turn.

        Args:
            state: the text to route, already truncated to the model's context.

        Returns:
            The decision, including probabilities.
        """
        ...

    def health(self) -> dict[str, Any]:
        """Whether this router is loaded and ready."""
        ...
