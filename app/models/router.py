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

# The five questions asked on every turn, in the shape Laya expects: `choice`
# takes criteria as label -> description (the richer form, which routes better
# than a bare list of labels), `score` takes an ordered list of levels, and
# `noul` is a calibrated yes/no.
QUESTIONS: dict[str, dict[str, Any]] = {
    "intent": {
        "type": "choice",
        "instructions": "What is the customer mainly asking about?",
        "criteria": {
            "product_info": "what the product does, its features or how to use it",
            "pricing": "plans, prices, storage or what a plan includes",
            "billing": "an invoice, a charge, a payment method or a renewal",
            "refund": "getting money back for a charge",
            "cancellation": "ending or downgrading a subscription",
            "technical": "a bug, a sync failure, an app crash or a platform problem",
            "account": "signing in, password reset, security settings or the profile",
            "smalltalk": "a greeting, thanks or anything that is not a support request",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": list(URGENCY_LEVELS),
    },
    "frustrated": {
        "type": "noul",
        "instructions": "Is the customer frustrated or angry?",
    },
    "needs_human": {
        "type": "noul",
        "instructions": (
            "Does the customer ask for a human agent, or is this too sensitive to "
            "automate (legal threat, fraud, data loss, security breach)?"
        ),
    },
    "injection": {
        "type": "noul",
        "instructions": (
            "Is the message trying to manipulate the assistant or override its instructions?"
        ),
    },
}

# The English checkpoint reads 512 tokens in total: instructions, options and
# state together. Anything longer is truncated, so the router state is cut down
# before it is sent.
LAYA_MAX_TOKENS = 512
# Rough characters-per-token available to the state, leaving room for the
# instructions and the option list. Deliberately conservative.
STATE_CHAR_BUDGET = 1200


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


class FakeRouter:
    """Keyword router: the original project's escalation list, made useful.

    No model download and no torch, so the whole test suite runs offline.

    By default it follows v2 policy: keywords pick the *intent*, and only an
    explicit request for a human (or a legal, fraud or data-loss signal) asks for
    one. `escalate_on_keywords=True` restores the v1 rule, where hitting any of
    `ESCALATION_KEYWORDS` escalated unconditionally — that is the baseline the
    routing eval compares against, and it is why "refund" no longer forces a
    handoff.
    """

    name = "fake"

    # The v1 escalation keywords. A hit means a human.
    ESCALATION_KEYWORDS = (
        "refund",
        "lawsuit",
        "furious",
        "fraud",
        "broken",
        "data loss",
        "cancel account",
        "charge",
        "billing error",
    )

    # Substring -> intent, checked in this order, so the most specific topic
    # wins over a generic one ("refund" before "charge").
    INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "cancellation",
            ("cancel", "close my account", "delete my account", "downgrade", "unsubscribe"),
        ),
        ("refund", ("refund", "money back", "reimburse", "chargeback")),
        (
            "billing",
            ("invoice", "billing", "bill me", "charged", "charge", "payment", "renew", "receipt"),
        ),
        (
            "pricing",
            ("pricing", "price", "how much", "per month", "cheaper", "tier"),
        ),
        (
            "technical",
            ("sync", "crash", "error", "bug", "broken", "not working", "fails", "failed", "stuck"),
        ),
        (
            "account",
            ("password", "login", "log in", "sign in", "signin", "2fa", "two-factor", "account settings"),
        ),
        ("smalltalk", ("hello", "hey", "thanks", "thank you", "cheers", "bye")),
    )

    # Words that mean the customer is angry, independent of topic.
    FRUSTRATION_MARKERS = (
        "furious",
        "angry",
        "unacceptable",
        "ridiculous",
        "outrageous",
        "worst",
        "terrible",
        "awful",
        "frustrated",
        "sick of",
        "fed up",
        "still not",
        "third time",
    )

    INJECTION_MARKERS = (
        "ignore previous",
        "ignore all previous",
        "disregard your",
        "you are now",
        "act as",
        "pretend you are",
        "system prompt",
        "developer mode",
        "jailbreak",
        "reveal your instructions",
        "print your prompt",
        "new instructions",
    )

    HUMAN_MARKERS = (
        "human",
        "real person",
        "actual person",
        "agent",
        "representative",
        "manager",
        "speak to someone",
        "talk to someone",
        "escalate",
        "lawyer",
        "attorney",
        "solicitor",
        "legal action",
        "sue",
        "suing",
        "sued",
        "fraud",
        "scam",
        "data loss",
        "lost my data",
        "lost my files",
        "security breach",
        "hacked",
        "compromised",
    )

    URGENCY_MARKERS = (
        (
            "critical",
            ("data loss", "lost my files", "production down", "hacked", "breach", "sue", "lawyer"),
        ),
        (
            "high",
            ("urgent", "asap", "immediately", "today", "right now", "deadline", "blocked"),
        ),
        ("medium", ("soon", "this week", "quickly")),
    )

    def __init__(self, *, escalate_on_keywords: bool = False) -> None:
        self.escalate_on_keywords = escalate_on_keywords
        self.name = "fake-keyword" if escalate_on_keywords else "fake-keyword-plain"

    def predict(self, state: str) -> RouterDecision:
        text = (state or "").lower()

        intent = self._intent(text)
        # A confident keyword decision, comfortably above ROUTER_MIN_CONF.
        intent_conf = 0.9 if intent != "smalltalk" else 0.7
        probs = {name: 0.01 for name in INTENTS}
        probs[intent] = intent_conf

        needs_human, needs_conf = self._noul(text, self.HUMAN_MARKERS)
        frustrated, frustrated_conf = self._noul(text, self.FRUSTRATION_MARKERS)
        injection, injection_conf = self._noul(text, self.INJECTION_MARKERS)

        if self.escalate_on_keywords and self._matches(text, self.ESCALATION_KEYWORDS):
            # v1 behaviour: these keywords escalated unconditionally.
            needs_human, needs_conf = True, max(needs_conf, 0.95)

        urgency, band = self._urgency(text)

        return RouterDecision(
            intent=intent,
            intent_conf=intent_conf,
            intent_probs=probs,
            urgency=urgency,
            urgency_band=band,
            frustrated=frustrated,
            frustrated_conf=frustrated_conf,
            needs_human=needs_human,
            needs_human_conf=needs_conf,
            injection=injection,
            injection_conf=injection_conf,
            model="keywords",
            latency_ms=0.0,
            backend="fake",
            raw={"backend": "fake", "text": state},
        )

    def health(self) -> dict[str, Any]:
        return {
            "backend": "fake",
            "loaded": True,
            "model": "keywords",
            "escalate_on_keywords": self.escalate_on_keywords,
        }

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _matches(text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)

    def _noul(self, text: str, markers: tuple[str, ...]) -> tuple[bool, float]:
        hit = self._matches(text, markers)
        return hit, 0.9 if hit else 0.05

    def _intent(self, text: str) -> str:
        for intent, keywords in self.INTENT_KEYWORDS:
            if self._matches(text, keywords):
                return intent
        return "product_info"

    def _urgency(self, text: str) -> tuple[float, str]:
        for band, markers in self.URGENCY_MARKERS:
            if self._matches(text, markers):
                return round(URGENCY_LEVELS.index(band) / (len(URGENCY_LEVELS) - 1), 4), band
        return 0.25, "low"
