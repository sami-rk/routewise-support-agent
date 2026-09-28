"""Compare the three routers on the same cases, on accuracy and latency.

Section 10.3 asks for the original's keyword escalation, an LLM router and Laya
side by side. The keyword router is free to run. The LLM router is here for
completeness, but it is a real cost: it sends every case to a free-tier model
that allows about 50 requests a day, so it is opt-in.

    python -m evals.compare_routers                       # keywords + Laya
    python -m evals.compare_routers --llm                 # add the LLM router
    python -m evals.compare_routers --backend fake        # keywords only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from app.models.router import INTENTS, RouterDecision, RouterModel
from evals.run_router import Case, evaluate, load_cases, print_report

INJECTION_PROMPT = (
    "Classify the support message below into exactly one of these labels, and "
    "reply with the label and nothing else:\n"
    + ", ".join(INTENTS)
    + "\n\nMessage: {text}\n\nLabel:"
)

# The LLM router gets the same options, so the comparison is like for like.
FEW_SHOT = [
    ("I was charged twice, refund the duplicate", "refund"),
    ("what is the airspeed velocity of a swallow", "smalltalk"),
    ("my files stopped syncing", "technical"),
    ("forgot my password", "account"),
    ("I want a human", "smalltalk"),
]


class LLMRouter:
    """A router that asks a free-tier LLM for the intent.

    Included for the comparison in section 10.3, not used by the agent: it costs
    an OpenRouter request per message, which is exactly what routing with Laya
    avoids.
    """

    name = "llm"

    def __init__(self, role: str = "faq") -> None:
        self.role = role
        self.invocations = 0

    def predict(self, state: str) -> RouterDecision:
        from app.models.llm import LLMUnavailableError
        from app.models.llm_runner import invoke_with_fallback

        shots = "\n".join(f"Message: {text}\nLabel: {label}" for text, label in FEW_SHOT)
        prompt = f"{shots}\n\n{INJECTION_PROMPT.format(text=state)}"
        self.invocations += 1
        try:
            response = invoke_with_fallback(
                prompt, role=self.role, run_id="compare", thread_id="compare"
            )
        except LLMUnavailableError as exc:
            # A rate limit must not look like a confident answer.
            return RouterDecision(
                intent=None, intent_conf=0.0, backend="llm", model="unavailable",
                raw={"error": str(exc)},
            )

        from app.agent.nodes.shared import message_text

        label = message_text(response).strip().lower()
        label = label.split()[0] if label else ""
        intent = label if label in INTENTS else None
        return RouterDecision(
            intent=intent,
            intent_conf=0.8 if intent else 0.0,
            intent_probs={name: (0.8 if name == intent else 0.2 / (len(INTENTS) - 1)) for name in INTENTS},
            model="openrouter",
            backend="llm",
        )

    def health(self) -> dict[str, Any]:
        return {"backend": "llm", "loaded": True, "invocations": self.invocations}


def timed(router: RouterModel, cases: list[Case]) -> tuple[list[float], float]:
    """Latency per decision, in milliseconds."""
    timings: list[float] = []
    for case in cases:
        started = time.perf_counter()
        router.predict(case.text)
        timings.append((time.perf_counter() - started) * 1000)
    return timings, statistics.median(timings) if timings else 0.0


def summarise(name: str, result: Any, median_ms: float, elapsed_s: float) -> dict[str, Any]:
    false_escalations = len([e for e in result.human_errors if not e[2]])
    missed = len([e for e in result.human_errors if e[2]])
    return {
        "router": name,
        "intent_accuracy": round(result.accuracy, 4),
        "correct": result.correct,
        "total": result.total,
        "false_escalations": false_escalations,
        "missed_escalations": missed,
        "escalation_accuracy": round((result.total - len(result.human_errors)) / result.total, 4)
        if result.total
        else 0.0,
        "median_latency_ms": round(median_ms, 1),
        "mean_latency_ms": round(statistics.fmean(result.latency_ms), 1) if result.latency_ms else 0.0,
        "total_seconds": round(elapsed_s, 1),
        # Only an LLM router spends a free-tier request per message, which is the
        # cost that routing with Laya exists to avoid.
        "openrouter_requests_per_case": 1 if name == "llm" else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the three routers.")
    parser.add_argument("--backend", choices=("laya", "fake"), default=None)
    parser.add_argument("--llm", action="store_true", help="Also run the LLM router (costs requests).")
    parser.add_argument("--quiet", action="store_true", help="Only print the comparison table.")
    args = parser.parse_args(argv)

    cases = load_cases()
    routers: list[tuple[str, RouterModel]] = []

    from app.models.router import FakeRouter

    routers.append(("v1 keywords", FakeRouter(escalate_on_keywords=True)))
    routers.append(("v2 keywords", FakeRouter()))

    backend = args.backend
    if backend is None:
        from app.config import get_settings

        backend = get_settings().router_backend
    if backend == "laya":
        from app.models.factory import build_router

        routers.append(("laya", build_router(load=True)))
    if args.llm:
        routers.append(("llm", LLMRouter()))

    print(f"{len(cases)} cases, {len(routers)} routers\n")

    rows: list[dict[str, Any]] = []
    for name, router in routers:
        if not args.quiet:
            print(f"=== {name} ===")
        started = time.perf_counter()
        result = evaluate(router, cases, name=name)
        _, median = timed(router, cases)
        if not args.quiet:
            print_report(result, cases, show_errors=5)
        rows.append(summarise(name, result, median, time.perf_counter() - started))

    print("\n" + "=" * 96)
    header = f"{'router':<12} {'intent acc':>11} {'escal acc':>11} {'false esc':>10} {'missed':>8} {'median ms':>11} {'total s':>9}"
    print(header)
    print("-" * 96)
    for row in rows:
        print(
            f"{row['router']:<12} {row['intent_accuracy']:>10.1%} {row['escalation_accuracy']:>11.1%} "
            f"{row['false_escalations']:>10} {row['missed_escalations']:>8} "
            f"{row['median_latency_ms']:>11.1f} {row['total_seconds']:>9.1f}"
        )
    print("=" * 96)
    print("OpenRouter requests per case: keywords 0, Laya 0, LLM 1.")
    print("That is the cost routing with Laya exists to avoid.")

    output = Path(__file__).with_name("router_comparison.json")
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {output.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
