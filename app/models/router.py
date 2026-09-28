"""Routing decisions and the `RouterModel` interface.

Two implementations satisfy the interface: `LayaRouter` (the local decision
model, see the rest of this file) and `FakeRouter` (keywords, for tests and for
machines that cannot run Laya). Everything downstream — the graph, the evals,
the API — only ever sees a `RouterDecision`, so the backend is swappable
through `ROUTER_BACKEND`.
"""

from __future__ import annotations

import re
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

# `needs_human` is the combination of two Laya questions plus a keyword backstop.
# The thresholds are set from the measured behaviour of the checkpoint: asking for
# a person is answered confidently, while "is this sensitive" carries a strong
# prior — a plain refund question scores 0.78 on it — so it needs a much higher bar
# and is backed by the explicit list below.
WANTS_HUMAN_THRESHOLD = 0.5
SENSITIVE_THRESHOLD = 0.85
# High-precision: these terms are unambiguous, and a message containing one is
# escalated whatever the model's probability says.
SENSITIVE_KEYWORDS = (
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
    "lost all my files",
    "lost my files",
    "security breach",
    "hacked",
    "compromised",
    "breach",
)
# Kept for callers that only need one number for "should a person look at this".
NEEDS_HUMAN_THRESHOLD = SENSITIVE_THRESHOLD
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
    "wants_human": {
        "type": "noul",
        "instructions": (
            "Does the customer explicitly ask to speak to a person, such as an agent, "
            "representative, manager or human?"
        ),
    },
    "sensitive": {
        "type": "noul",
        "instructions": (
            "Does the message mention legal action, a lawyer, fraud, data loss, "
            "lost files, or a security breach?"
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


def normalise_score(answer: dict[str, Any], levels: tuple[str, ...] = URGENCY_LEVELS) -> tuple[float, str]:
    """Map a Laya `score` answer onto 0..1 and name the band.

    A score answer is an *expected value* over the level indices, not an index
    and not a probability: a spread of 0.2 across levels 3 and 4 comes back as
    3.8. Dividing by the top of the scale turns that into the 0..1 position the
    graph state wants. The band is the level the expectation sits closest to,
    with ties rounding up, so the trace shows something a human recognises.

    Args:
        answer: one entry of `result["answers"]` for a `score` question.
        levels: the ordered scale the question was asked with.

    Returns:
        `(position, band)`. A missing or malformed answer falls back to the
        bottom of the scale rather than raising.
    """
    span = max(1, len(levels) - 1)
    raw = answer.get("score")
    if not isinstance(raw, (int, float)):
        return 0.0, levels[0]
    position = min(1.0, max(0.0, float(raw) / span))
    band_index = min(span, max(0, int(round(float(raw)))))
    return round(position, 4), levels[band_index]


def probability_of_true(answer: dict[str, Any]) -> float:
    """P(true) from a Laya `noul` answer, clamped to 0..1."""
    value = answer.get("noul")
    if not isinstance(value, (int, float)):
        return 0.0
    return round(min(1.0, max(0.0, float(value))), 4)


def combine_needs_human(
    text: str,
    wants_human_conf: float | None,
    sensitive_conf: float | None,
) -> tuple[bool, float, str | None]:
    """Decide whether a person has to look at this turn.

    Three signals, in descending order of reliability:

    1. an explicit legal, fraud or data-loss term, which is unambiguous and fires
       whatever the model said;
    2. the customer explicitly asking for a person, which the model answers well;
    3. the model's own read of whether the message is sensitive, which is the
       weakest of the three and therefore needs the highest bar — a plain refund
       request scores 0.78 on it.

    Args:
        text: the routed text.
        wants_human_conf: P(customer asked for a person).
        sensitive_conf: P(message is a legal, fraud or data-loss incident).

    Returns:
        `(needs_human, confidence, reason)`, where `reason` names the signal that
        decided it, or is None when nothing fired.
    """
    lowered = (text or "").lower()
    for keyword in SENSITIVE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}", lowered):
            return True, max(0.95, sensitive_conf or 0.0), f"keyword:{keyword}"
    if (wants_human_conf or 0.0) > WANTS_HUMAN_THRESHOLD:
        return True, wants_human_conf or 0.0, "asked_for_human"
    if (sensitive_conf or 0.0) > SENSITIVE_THRESHOLD:
        return True, sensitive_conf or 0.0, "sensitive_topic"
    return False, max(wants_human_conf or 0.0, sensitive_conf or 0.0), None


def latest_utterance(state: str) -> str:
    """The customer's most recent message inside a formatted router state.

    `build_router_state` labels the profile and earlier turns, so a routing
    signal can be taken from what the customer just said rather than from the
    whole state. Without this, a customer on the "pro plan" has every message
    read as a pricing question, and a summary mentioning an open legal issue
    escalates every turn.

    Plain text with no labels — which is what the evals pass — is returned
    unchanged.
    """
    if "Customer:" not in state:
        return state
    return state.rsplit("Customer:", 1)[1].strip()


def build_router_state(
    user_input: str,
    history: list[tuple[str, str]] | None = None,
    customer_summary: str | None = None,
    *,
    max_chars: int = STATE_CHAR_BUDGET,
    max_turns: int = 2,
) -> str:
    """Build the text Laya routes on, small enough for its 512-token context.

    The checkpoint spends that budget on instructions, options and state
    together, so the state has to be tiny: the customer's latest message, at
    most `max_turns` earlier turns, and a short profile summary. The newest
    message is never dropped — it is what the decision is about — and if the
    whole thing still does not fit, the older turns go first.

    Args:
        user_input: the customer's latest message.
        history: earlier turns as (role, text), oldest first.
        customer_summary: plan, devices, open issues. One short line.
        max_chars: total budget for the state text.
        max_turns: how many earlier turns to include.

    Returns:
        The state string to hand to the router.
    """
    parts: list[str] = []
    if customer_summary:
        # Labelled distinctly from an utterance: a "pro plan" in the profile
        # must not be read as the customer asking about pricing.
        parts.append(f"Profile: {customer_summary.strip()}")

    turns = list(history or [])
    # Most recent turns first, so a later cut drops the oldest.
    recent = list(reversed(turns))[: max(0, max_turns)]
    for role, text in reversed(recent):
        text = " ".join((text or "").split())
        if text:
            parts.append(f"{role}: {text}")

    latest = " ".join((user_input or "").split())
    if latest:
        parts.append(f"Customer: {latest}")

    state = "\n".join(parts)
    if len(state) <= max_chars:
        return state

    # Over budget: keep the summary and the latest message, trim the middle.
    head = f"Profile: {customer_summary.strip()}\n" if customer_summary else ""
    tail = f"Customer: {latest}" if latest else ""
    if len(head) + len(tail) >= max_chars:
        # Even the essentials do not fit: keep the end, which holds the message.
        return tail[-max_chars:] if tail else head[:max_chars]
    room = max_chars - len(head) - len(tail) - 1
    middle = state[len(head) : len(state) - len(tail)]
    return f"{head}{middle[-room:] if room > 0 else ''}\n{tail}"


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

    # `needs_human` is the escalation decision the graph acts on. It is the
    # combination of the two questions below and the keyword backstop, so the
    # raw signals are kept for the trace and the eval.
    needs_human: bool = False
    needs_human_conf: float | None = None
    wants_human_conf: float | None = None
    sensitive_conf: float | None = None
    # Which of the three signals decided the escalation.
    needs_human_reason: str | None = None

    injection: bool = False
    injection_conf: float | None = None

    # Which checkpoint answered, and how long the decision took.
    model: str | None = None
    latency_ms: float | None = None
    # The backend that produced this decision: "laya" or "fake".
    backend: str = "fake"
    # The text that was routed, kept so the keyword rule can be re-evaluated.
    text: str = ""
    # The untouched answer payload, for traces and evals.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_human(self) -> bool:
        """True when a person must look at this turn.

        Named from the caller's side, so the field on the decision stays
        `needs_human` and this reads as a question.
        """
        return self.needs_human

    @property
    def is_unsafe(self) -> bool:
        """True when the message looks like a prompt-injection attempt."""
        return self.injection and (self.injection_conf or 0.0) > INJECTION_THRESHOLD

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
            "wants_human_conf": self.wants_human_conf,
            "sensitive_conf": self.sensitive_conf,
            "needs_human_reason": self.needs_human_reason,
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
            ("invoice", "billing", "bill", "charged", "charge", "payment", "renew", "receipt"),
        ),
        (
            "pricing",
            ("pricing", "price", "plan", "how much", "per month", "cheaper", "tier"),
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

    # Explicit requests for a person. Legal, fraud and data-loss wording is
    # handled once, for both backends, by `combine_needs_human`.
    HUMAN_MARKERS = (
        "human",
        "real person",
        "actual person",
        "representative",
        "manager",
        "speak to someone",
        "talk to someone",
        "escalate",
    )

    URGENCY_MARKERS = (
        (
            "critical",
            (
                "data loss",
                "lost my data",
                "lost all my files",
                "lost my files",
                "production down",
                "hacked",
                "breach",
                "sue",
                "lawyer",
            ),
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
        # Signals come from what the customer just said, not from the whole
        # state: a "pro plan" in the profile must not make every message a
        # pricing question.
        text = latest_utterance(state or "").lower()

        intent = self._intent(text)
        # A confident keyword decision, comfortably above ROUTER_MIN_CONF.
        intent_conf = 0.9 if intent != "smalltalk" else 0.7
        probs = {name: 0.01 for name in INTENTS}
        probs[intent] = intent_conf

        frustrated, frustrated_conf = self._noul(text, self.FRUSTRATION_MARKERS)
        injection, injection_conf = self._noul(text, self.INJECTION_MARKERS)

        if self.escalate_on_keywords:
            # v1 in full: its own escalation list is the only signal, so a
            # legal threat phrased without those keywords was missed.
            needs_human, needs_conf = self._noul(text, self.ESCALATION_KEYWORDS)
            reason = "v1_keyword" if needs_human else None
            wants_conf, sensitive_conf = needs_conf, 0.0
        else:
            wants_hit, wants_conf = self._noul(text, self.HUMAN_MARKERS)
            sensitive_conf = 0.0
            needs_human, needs_conf, reason = combine_needs_human(
                state, wants_conf if wants_hit else 0.0, 0.0
            )

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
            wants_human_conf=wants_conf,
            sensitive_conf=sensitive_conf,
            needs_human_reason=reason,
            injection=injection,
            injection_conf=injection_conf,
            model="keywords",
            latency_ms=0.0,
            backend="fake",
            text=state,
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
        """True when any marker appears at the start of a word.

        Anchored at the start only, so "charge" still matches "charged" and
        "cancel" still matches "cancelled", while "sync" no longer matches the
        "sync" inside "CloudSync".
        """
        return any(re.search(rf"\b{re.escape(marker)}", text) for marker in markers)

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



class LayaRouter:
    """The local Laya decision model, wrapped so nothing else imports `laya`.

    Laya answers typed questions in one forward pass and returns calibrated
    probabilities, so a turn costs milliseconds and zero OpenRouter requests.
    The English checkpoint is `convaiinnovations/laya` (ModernBERT-large, 512
    tokens of context); `laya.Router` is deliberately not used, because it
    routes between checkpoints per request and multilingual routing is out of
    scope for v1.

    Every Laya-specific detail lives in this class: how the agent is loaded, how
    it is called, and how its answers are read. Everything downstream takes a
    `RouterDecision`.
    """

    name = "laya"

    def __init__(
        self,
        checkpoint: str = "convaiinnovations/laya",
        device: str = "auto",
        *,
        questions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.questions = questions or QUESTIONS
        self._agent: Any = None

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> "LayaRouter":
        """Download and build the checkpoint. Idempotent.

        CPU by default; `device="auto"` uses CUDA when it is available. Safe to
        call from the FastAPI lifespan, before the event loop starts serving.
        """
        if self._agent is not None:
            return self

        import laya  # imported here so the rest of the app never needs torch

        device = resolve_device(self.device)
        self._agent = laya.load(self.checkpoint, device=device)
        return self

    def health(self) -> dict[str, Any]:
        return {
            "backend": "laya",
            "loaded": self._agent is not None,
            "checkpoint": self.checkpoint,
            "device": self.device,
        }

    # --- inference ---------------------------------------------------------

    def predict(self, state: str) -> RouterDecision:
        """Answer the routing questions about one turn.

        The state is truncated to the checkpoint's context first, so a long
        history cannot push the customer's actual message out of the window.
        """
        self.load()
        payload = self._system_one(state)

        answers = payload.get("answers") or {}
        intent_answer = answers.get("intent") or {}
        urgency_answer = answers.get("urgency") or {}

        probs = {
            label: float(p)
            for label, p in (intent_answer.get("probabilities") or {}).items()
        }
        intent = intent_answer.get("choice")
        # The confidence gate uses the top probability, not Laya's
        # `confidence` field, which is a normalised-entropy measure.
        intent_conf = max(probs.values()) if probs else None

        urgency, band = normalise_score(urgency_answer, URGENCY_LEVELS)
        frustrated = probability_of_true(answers.get("frustrated") or {})
        wants_human = probability_of_true(answers.get("wants_human") or {})
        sensitive = probability_of_true(answers.get("sensitive") or {})
        injection = probability_of_true(answers.get("injection") or {})

        needs_human, needs_conf, reason = combine_needs_human(state, wants_human, sensitive)

        return RouterDecision(
            intent=intent if intent in INTENTS else None,
            intent_conf=round(intent_conf, 4) if intent_conf is not None else None,
            intent_probs=probs,
            urgency=urgency,
            urgency_band=band,
            frustrated=frustrated > 0.5,
            frustrated_conf=frustrated,
            needs_human=needs_human,
            needs_human_conf=needs_conf,
            wants_human_conf=wants_human,
            sensitive_conf=sensitive,
            needs_human_reason=reason,
            injection=injection > 0.5,
            injection_conf=injection,
            model=(payload.get("routing") or {}).get("model") or self.checkpoint,
            backend="laya",
            text=state,
            raw={
                "answers": answers,
                "routing": payload.get("routing"),
                "usage": payload.get("usage"),
            },
        )

    def _system_one(self, state: str) -> dict[str, Any]:
        """Call Laya, tolerating the API differences between versions.

        0.3.x exposes `Agent.system_one(state, questions)`. Earlier releases
        used `predict(state, questions)`, and the `laya.Router` wrapper still
        does. The first method that exists wins, so a version bump in either
        direction is a one-line change here rather than a rewrite of the graph.

        Args:
            state: the routing state, already truncated by the caller.

        Returns:
            The raw payload, with an `answers` key.
        """
        truncated = state[:STATE_CHAR_BUDGET]
        agent = self._agent
        for method in ("system_one", "predict"):
            call = getattr(agent, method, None)
            if callable(call):
                payload = call(truncated, self.questions)
                if payload is not None:
                    return dict(payload)
        raise RuntimeError(
            f"Laya agent {type(agent).__name__} exposes neither system_one() nor predict(); "
            "update LayaRouter._system_one for this laya version."
        )


def resolve_device(device: str) -> str:
    """Turn `ROUTER_DEVICE` into a device Laya accepts.

    `auto` picks CUDA when torch can see it, and CPU otherwise. This box has no
    GPU, so the default is CPU and the latency budget is met on CPU alone.
    """
    if device in ("cpu", "cuda"):
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"
