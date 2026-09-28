"""Score a router against `router_cases.jsonl`.

Reports intent accuracy, per-intent precision and recall, a confusion matrix, a
calibration check, and — when asked — the v1 keyword baseline next to it, so the
"fewer false escalations than the original" claim in section 17 is a measurement
rather than an assertion.

    python -m evals.run_router                       # Laya if installed, else fake
    python -m evals.run_router --backend fake
    python -m evals.run_router --baseline            # also run the v1 keyword router
    python -m evals.run_router --min-accuracy 0.85   # gate, default 0.85
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.router import INTENTS, NEEDS_HUMAN_THRESHOLD, RouterModel

CASES_PATH = Path(__file__).with_name("router_cases.jsonl")
DEFAULT_MIN_ACCURACY = 0.85


@dataclass
class Case:
    id: str
    text: str
    intent: str
    urgency_band: str
    frustrated: bool
    needs_human: bool
    injection: bool = False
    low_confidence: bool = False
    note: str = ""


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    """Read the labelled cases, one JSON object per line."""
    cases: list[Case] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} is not valid JSON: {exc}") from exc
        cases.append(
            Case(
                id=data["id"],
                text=data["text"],
                intent=data["intent"],
                urgency_band=data["urgency_band"],
                frustrated=bool(data["frustrated"]),
                needs_human=bool(data["needs_human"]),
                injection=bool(data.get("injection", False)),
                low_confidence=bool(data.get("low_confidence", False)),
                note=data.get("note", ""),
            )
        )
    return cases


@dataclass
class Result:
    """Accuracy plus the raw pairs, so several routers can be compared."""

    name: str
    correct: int = 0
    total: int = 0
    confusion: Counter = None  # type: ignore[assignment]
    latency_ms: list[float] = None  # type: ignore[assignment]
    intent_errors: list[tuple[str, str, str]] = None  # type: ignore[assignment]
    human_errors: list[tuple[str, bool, bool, float]] = None  # type: ignore[assignment]
    injection_errors: list[tuple[str, bool, bool]] = None  # type: ignore[assignment]
    calibration: list[tuple[float, bool]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.confusion = self.confusion or Counter()
        self.latency_ms = self.latency_ms or []
        self.intent_errors = self.intent_errors or []
        self.human_errors = self.human_errors or []
        self.injection_errors = self.injection_errors or []
        self.calibration = self.calibration or []

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def precision_recall(self) -> dict[str, tuple[float, float]]:
        """Per-intent precision and recall over the cases that mention the intent."""
        out: dict[str, tuple[float, float]] = {}
        for intent in INTENTS:
            tp = self.confusion.get((intent, intent), 0)
            predicted = sum(v for (p, _g), v in self.confusion.items() if p == intent)
            actual = sum(v for (_p, g), v in self.confusion.items() if g == intent)
            precision = tp / predicted if predicted else 0.0
            recall = tp / actual if actual else 0.0
            out[intent] = (precision, recall)
        return out


def evaluate(router: RouterModel, cases: list[Case], *, name: str | None = None) -> Result:
    """Route every case and score intent, escalation and injection."""
    result = Result(name=name or getattr(router, "name", "router"), total=len(cases))
    for case in cases:
        started = time.perf_counter()
        decision = router.predict(case.text)
        result.latency_ms.append((time.perf_counter() - started) * 1000)

        result.confusion[(decision.intent, case.intent)] += 1
        if decision.intent == case.intent:
            result.correct += 1
        else:
            result.intent_errors.append((case.id, decision.intent or "None", case.intent))

        # The graph escalates when the decision says a person is required.
        if decision.needs_human != case.needs_human:
            result.human_errors.append(
                (case.id, decision.needs_human, case.needs_human, decision.needs_human_conf or 0.0)
            )
        if case.injection and not decision.is_unsafe:
            result.injection_errors.append((case.id, False, True))

        if decision.intent_conf is not None:
            result.calibration.append((decision.intent_conf, decision.intent == case.intent))
    return result


def print_report(result: Result, cases: list[Case], *, show_errors: int = 12) -> None:
    """Accuracy, per-intent scores, confusion matrix, calibration and latency."""
    timings = sorted(result.latency_ms)
    print(f"\n=== {result.name} ===")
    print(f"intent accuracy : {result.correct}/{result.total} = {result.accuracy:.1%}")

    print("\nper-intent precision / recall / support")
    scores = result.precision_recall()
    for intent in INTENTS:
        precision, recall = scores[intent]
        support = sum(v for (_p, g), v in result.confusion.items() if g == intent)
        print(f"  {intent:<14} P={precision:>5.2f}  R={recall:>5.2f}  n={support:>3}")

    print("\nconfusion matrix (rows: predicted, cols: expected)")
    header = " " * 14 + "".join(f"{i[:6]:>7}" for i in INTENTS)
    print(header)
    for predicted in INTENTS:
        cells = "".join(
            f"{result.confusion.get((predicted, expected), 0):>7}" for expected in INTENTS
        )
        print(f"  {predicted:<12}{cells}")

    if result.human_errors:
        missed = [e for e in result.human_errors if e[2]]
        extra = [e for e in result.human_errors if not e[2]]
        print(
            f"\nescalation       : {len(result.human_errors)} disagreements "
            f"({len(missed)} missed, {len(extra)} false)"
        )
    else:
        print("\nescalation       : perfect")

    injection_cases = [c for c in cases if c.injection]
    caught = len(injection_cases) - len(result.injection_errors)
    if injection_cases:
        print(f"injection caught : {caught}/{len(injection_cases)}")

    # Calibration: how often a decision at a given confidence was right. A router
    # that is well calibrated puts correctness above its stated confidence on the
    # confident bins.
    if result.calibration:
        print("\ncalibration (confidence bin -> accuracy, n)")
        bins = [(0.0, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.01)]
        for low, high in bins:
            bucket = [(conf, ok) for conf, ok in result.calibration if low <= conf < high]
            if not bucket:
                continue
            accuracy = sum(1 for _c, ok in bucket if ok) / len(bucket)
            print(f"  {low:.2f}-{high:.2f}  accuracy={accuracy:.2f}  n={len(bucket):>3}")

    if timings:
        print(
            f"\nlatency          : median {statistics.median(timings):.1f} ms  "
            f"mean {statistics.fmean(timings):.1f} ms  max {max(timings):.1f} ms"
        )

    if result.intent_errors:
        print(f"\nintent errors ({len(result.intent_errors)}), first {show_errors}:")
        for case_id, predicted, expected in result.intent_errors[:show_errors]:
            print(f"  {case_id:<28} predicted={predicted:<14} expected={expected}")
    if result.human_errors:
        print(f"\nescalation errors ({len(result.human_errors)}), first {show_errors}:")
        for case_id, predicted, expected, conf in result.human_errors[:show_errors]:
            label = "missed" if expected else "false"
            print(f"  {case_id:<28} {label:<5} predicted={str(predicted):<5} expected={str(expected):<5} conf={conf:.2f}")


def build_router(backend: str) -> RouterModel:
    """Build a router, falling back to keywords if Laya is not available."""
    if backend == "fake":
        from app.models.router import FakeRouter

        return FakeRouter()
    try:
        from app.models.factory import build_router

        return build_router(load=True)
    except Exception as exc:  # pragma: no cover - depends on the environment
        print(f"Laya unavailable ({type(exc).__name__}: {exc}); falling back to keywords.")
        from app.models.router import FakeRouter

        return FakeRouter()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a router against the labelled cases.")
    parser.add_argument("--backend", choices=("laya", "fake"), default=None)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--baseline", action="store_true", help="Also score the v1 keyword router.")
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY)
    args = parser.parse_args(argv)

    # Validate the configuration before scoring, so a paid model cannot be measured.
    from app.config import validate_settings

    validate_settings()
    cases = load_cases(args.cases)
    print(f"{len(cases)} cases from {args.cases}")

    backend = args.backend
    if backend is None:
        from app.config import get_settings

        backend = get_settings().router_backend

    result = evaluate(build_router(backend), cases)
    print_report(result, cases)

    baseline_result: Result | None = None
    if args.baseline:
        from app.models.router import FakeRouter

        baseline_result = evaluate(
            FakeRouter(escalate_on_keywords=True), cases, name="baseline: v1 keywords"
        )
        print_report(baseline_result, cases, show_errors=5)

    print(f"\nthreshold: {args.min_accuracy:.0%} intent accuracy")
    if result.accuracy < args.min_accuracy:
        print(f"FAIL: {result.name} scored {result.accuracy:.1%}, below {args.min_accuracy:.0%}")
        return 1
    print(f"PASS: {result.name} scored {result.accuracy:.1%}")

    if baseline_result is not None:
        false_escalations = len(
            [e for e in baseline_result.human_errors if not e[2]]
        )
        ours = len([e for e in result.human_errors if not e[2]])
        print(
            f"\nfalse escalations: v1 keywords {false_escalations}, "
            f"{result.name} {ours}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
