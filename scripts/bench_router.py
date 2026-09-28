"""Measure router latency, against the 250 ms budget in section 17.

    python -m scripts.bench_router            # both backends
    python -m scripts.bench_router --backend laya --repeat 20

The number depends entirely on the machine. Laya's own figures are for a GPU;
the English checkpoint is a 421M-parameter ModernBERT-large, so a 2015-era
4-core laptop CPU runs one five-question pass in seconds, not milliseconds. The
keyword router is three orders of magnitude faster and is what `ROUTER_BACKEND=fake`
uses.
"""

from __future__ import annotations

import argparse
import statistics
import time

from app.config import get_settings
from app.models.router import FakeRouter, LayaRouter, build_router_state

MESSAGES = [
    "How much is the Pro plan and how many devices does it support?",
    "I was charged 5 days ago and want a refund",
    "Cancel my subscription",
    "my sync keeps failing on Windows and the app crashes",
    "I want a human",
]

BUDGET_MS = 250.0


def bench(router, repeat: int) -> dict[str, float]:
    # One warm-up pass, so the first-call cost is not counted.
    router.predict(build_router_state(MESSAGES[0]))
    timings: list[float] = []
    for index in range(repeat):
        state = build_router_state(MESSAGES[index % len(MESSAGES)])
        started = time.perf_counter()
        router.predict(state)
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    return {
        "median": statistics.median(timings),
        "mean": statistics.fmean(timings),
        "p95": timings[min(len(timings) - 1, int(len(timings) * 0.95))],
        "min": timings[0],
        "max": timings[-1],
    }


def report(label: str, stats: dict[str, float]) -> None:
    verdict = "within budget" if stats["median"] <= BUDGET_MS else "over budget"
    print(
        f"{label:<10} median {stats['median']:>8.1f} ms   mean {stats['mean']:>8.1f} ms   "
        f"p95 {stats['p95']:>8.1f} ms   min {stats['min']:>8.1f} ms   max {stats['max']:>8.1f} ms   "
        f"[{verdict}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure router latency.")
    parser.add_argument("--backend", choices=("laya", "fake", "both"), default="both")
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    backends = ("laya", "fake") if args.backend == "both" else (args.backend,)

    if "fake" in backends:
        report("fake", bench(FakeRouter(), args.repeat))

    if "laya" in backends:
        settings = get_settings()
        router = LayaRouter(
            checkpoint=settings.router_checkpoint, device=settings.router_device
        ).load()
        print(f"laya: {router.health()}")
        report("laya", bench(router, args.repeat))

    print(f"\nbudget: {BUDGET_MS:.0f} ms per routing decision on CPU")


if __name__ == "__main__":
    main()
