"""Draw the compiled support graph as an SVG.

    python -m scripts.plot_graph                          # write docs/support_graph.svg
    python -m scripts.plot_graph --out somewhere.svg
    python -m scripts.plot_graph --check                  # fail if the file is stale

The nodes and edges come from `build_graph().get_graph()`, so the picture cannot
drift from the code: a new node or a rewired edge changes the SVG, and `--check`
fails in CI until it is redrawn. `check_layout` also refuses to draw when an
edge does not point forwards, which would otherwise be rendered as a line
through the boxes in between.

What `--check` does and does not guarantee: it compares the structure read from
the graph against the committed file. The edge **labels** come from `EDGE_LABELS`
rather than from LangGraph, which does not expose the condition text, so rewording
a label here will not fail the check. The labels are the one hand-maintained part.

Output is deterministic — no timestamps, no generated ids — so regenerating an
unchanged graph produces a byte-identical file.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "docs" / "support_graph.svg"

# So `python scripts/plot_graph.py` works as well as `python -m
# scripts.plot_graph`: run as a file, the repo root is not on sys.path and
# `import app` fails, which is how CI invokes it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- appearance ------------------------------------------------------------

NODE_W, NODE_H = 152, 48
COL_GAP, ROW_GAP = 232, 92
MARGIN_X, MARGIN_TOP = 32, 104
LEGEND_TOP_GAP = 74
CHAR_W = 5.55  # approximate width of a character at 10.5px, for label plates

# Node role -> (fill, stroke, text colour).
STYLES: dict[str, tuple[str, str, str]] = {
    "entry": ("#e8eef7", "#4a6fa5", "#1c2b40"),
    "routing": ("#ede4f7", "#7a5ba6", "#2f2140"),
    "retrieval": ("#dff2ef", "#3f8f85", "#16332f"),
    "agent": ("#e2f2e2", "#4f8f4f", "#1c331c"),
    "hitl": ("#fdf0d9", "#c08a2e", "#3d2c0d"),
    "action": ("#fce5da", "#c0703f", "#3d2515"),
    "output": ("#eef0f2", "#7b8794", "#232a31"),
}

SUBTITLES: dict[str, str] = {
    "__start__": "customer message",
    "load_context": "profile, memory, trim",
    "laya_router": "intent, urgency, noul",
    "guardrail": "injection, then route",
    "retrieve_kb": "FAISS search + floor",
    "faq_agent": "product, pricing, how-to",
    "billing_agent": "invoices, refunds, cancel",
    "technical_agent": "sync and platform",
    "clarify": "one clarifying question",
    "approval_gate": "interrupt() for a decision",
    "execute_action": "refund or cancel",
    "create_ticket": "real ticket row",
    "escalate": "empathetic handoff",
    "respond": "reply + citations",
    "update_memory": "merge durable facts",
    "__end__": "",
}

ROLES: dict[str, str] = {
    "__start__": "output",
    "load_context": "entry",
    "laya_router": "routing",
    "guardrail": "routing",
    "retrieve_kb": "retrieval",
    "faq_agent": "agent",
    "billing_agent": "agent",
    "technical_agent": "agent",
    "clarify": "hitl",
    "approval_gate": "hitl",
    "create_ticket": "action",
    "escalate": "action",
    "execute_action": "action",
    "respond": "output",
    "update_memory": "output",
    "__end__": "output",
}

# Column-major, left to right. A node that consumes another's output gets its own
# column so every edge points forwards; escalate is alone in one because it feeds
# create_ticket.
COLUMNS: list[list[str]] = [
    ["__start__"],
    ["load_context"],
    ["laya_router"],
    ["guardrail"],
    ["retrieve_kb"],
    ["faq_agent", "billing_agent", "technical_agent"],
    ["clarify", "approval_gate"],
    ["escalate"],
    ["execute_action", "create_ticket"],
    ["respond"],
    ["update_memory"],
    ["__end__"],
]

# Hand-maintained, because LangGraph does not expose the condition text.
#
# These are deliberately short. The gap between two adjacent columns is about
# 80px, and a full sentence does not fit in it without covering a node, so the
# diagram carries the decision and the README table carries the exact condition.
EDGE_LABELS: dict[tuple[str, str], str] = {
    ("guardrail", "retrieve_kb"): "routed",
    ("guardrail", "escalate"): "person",
    ("guardrail", "clarify"): "unsure",
    ("guardrail", "respond"): "injection",
    ("retrieve_kb", "faq_agent"): "faq intents",
    ("retrieve_kb", "billing_agent"): "billing, refund, cancel",
    ("retrieve_kb", "technical_agent"): "technical",
    ("retrieve_kb", "clarify"): "unsure",
    ("retrieve_kb", "escalate"): "person",
    ("retrieve_kb", "respond"): "injection",
    ("faq_agent", "create_ticket"): "ticket",
    ("faq_agent", "respond"): "answer",
    ("technical_agent", "create_ticket"): "ticket",
    ("technical_agent", "respond"): "answer",
    ("billing_agent", "approval_gate"): "action",
    ("billing_agent", "respond"): "no action",
    ("approval_gate", "execute_action"): "approved",
    ("approval_gate", "respond"): "rejected",
}

# Solid = always taken. Dashed = a conditional edge in LangGraph.
EDGE_COLOURS = {"solid": "#55606e", "dashed": "#98a2b3"}

TITLE = "CloudSync Pro support graph"


# --- reading the graph -----------------------------------------------------


def read_graph() -> tuple[list[str], list[tuple[str, str, bool]]]:
    """Nodes and edges from the compiled graph, in a stable order."""
    from app.agent.graph import build_graph

    spec = build_graph().get_graph()
    nodes = sorted({n for n in list(spec.nodes) + ["__start__", "__end__"]} - {"__enter__"})
    edges = sorted({(e.source, e.target, bool(e.conditional)) for e in spec.edges})
    return nodes, edges


def column_of(name: str) -> int:
    for index, column in enumerate(COLUMNS):
        if name in column:
            return index
    raise KeyError(name)


def _is_obvious(source: str, target: str) -> bool:
    """Edges whose meaning is the node name itself, so a label would be noise."""
    return (source, target) in {
        ("__start__", "load_context"),
        ("load_context", "laya_router"),
        ("laya_router", "guardrail"),
        ("escalate", "create_ticket"),
        ("execute_action", "respond"),
        ("clarify", "respond"),
        ("create_ticket", "respond"),
        ("respond", "update_memory"),
        ("update_memory", "__end__"),
    }


def check_layout(nodes: list[str], edges: list[tuple[str, str, bool]]) -> list[str]:
    """Everything wrong with the layout, so the diagram cannot be silently partial."""
    problems: list[str] = []
    placed = {name for column in COLUMNS for name in column}

    for node in nodes:
        if node not in placed:
            problems.append(f"node {node!r} has no place in COLUMNS")
    for node in sorted(placed):
        if node not in nodes:
            problems.append(f"COLUMNS lists {node!r}, which is not a node of the graph")
    for source, target, _conditional in edges:
        if source in placed and target in placed and column_of(target) <= column_of(source):
            problems.append(
                f"edge {source} -> {target} does not point forwards, so it would be "
                "drawn through the boxes in between; move one of them to a later column"
            )
        if (source, target) not in EDGE_LABELS and not _is_obvious(source, target):
            problems.append(f"edge {source} -> {target} has no label in EDGE_LABELS")
    return problems


# --- geometry --------------------------------------------------------------


def node_box(name: str) -> tuple[float, float]:
    """(x, y) of a node's top-left corner."""
    column_index = column_of(name)
    row = COLUMNS[column_index].index(name)
    return MARGIN_X + column_index * COL_GAP, MARGIN_TOP + row * ROW_GAP


def svg_size() -> tuple[int, int]:
    width = MARGIN_X * 2 + (len(COLUMNS) - 1) * COL_GAP + NODE_W
    height = MARGIN_TOP + max(len(c) for c in COLUMNS) * ROW_GAP + LEGEND_TOP_GAP + 78
    return int(width), int(height)


def _bezier_mid(p0, p1, p2, p3) -> tuple[float, float]:
    """The point at t=0.5 of a cubic, which is the middle of the drawn curve.

    Labels go here rather than on a straight line between the boxes, so a bowed
    edge's label sits in its own lane instead of on top of a node.
    """
    return (
        (p0[0] + 3 * p1[0] + 3 * p2[0] + p3[0]) / 8.0,
        (p0[1] + 3 * p1[1] + 3 * p2[1] + p3[1]) / 8.0,
    )


def edge_path(source: str, target: str) -> tuple[str, float, float]:
    """A cubic bezier between two nodes, and where its label goes.

    An edge that skips columns is bowed away from the boxes in between, upward
    when the target is higher on screen and downward when it is lower, so the two
    directions do not share a lane. The magnitude grows with the skip so that
    parallel edges of different lengths do not land on the same curve.
    """
    sx, sy = node_box(source)
    tx, ty = node_box(target)
    start = (sx + NODE_W, sy + NODE_H / 2)
    end = (tx, ty + NODE_H / 2)
    columns = column_of(target) - column_of(source)

    if columns <= 1:
        bend = (end[0] - start[0]) * 0.5
        control1 = (start[0] + bend, start[1])
        control2 = (end[0] - bend, end[1])
    else:
        lane = min(46 + columns * 17, 150) * (-1 if end[1] <= start[1] else 1)
        bend = (end[0] - start[0]) * 0.34
        control1 = (start[0] + bend, start[1] + lane)
        control2 = (end[0] - bend, end[1] + lane)

    path = (
        f"M {start[0]:.0f},{start[1]:.0f} "
        f"C {control1[0]:.0f},{control1[1]:.0f} "
        f"{control2[0]:.0f},{control2[1]:.0f} "
        f"{end[0]:.0f},{end[1]:.0f}"
    )
    mx, my = _bezier_mid(start, control1, control2, end)
    return path, mx, my - 6


# --- rendering -------------------------------------------------------------


def _label(x: float, y: float, text: str, *, size: float = 10.5, colour: str = "#55606e") -> str:
    """Text on an opaque plate.

    A stroke halo would be tidier, but `paint-order` is not honoured everywhere,
    and a label crossing an edge is unreadable. The plate is portable.
    """
    escaped = html.escape(text)
    plate_w = len(text) * CHAR_W * (size / 10.5) + 8
    return (
        f'<rect x="{x - plate_w / 2:.1f}" y="{y - size:.1f}" width="{plate_w:.1f}" '
        f'height="{size + 6}" rx="3" fill="#ffffff" fill-opacity="0.92"/>'
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{colour}" '
        f'text-anchor="middle">{escaped}</text>'
    )


def render(nodes: list[str], edges: list[tuple[str, str, bool]]) -> str:
    width, height = svg_size()
    conditional_count = sum(1 for _s, _t, c in edges if c)
    out: list[str] = []
    add = out.append

    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif, system-ui, '
        f'-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif">'
    )
    add(
        '<defs>'
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        f'fill="{EDGE_COLOURS["solid"]}"/></marker>'
        '<marker id="arrow-soft" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        f'fill="{EDGE_COLOURS["dashed"]}"/></marker>'
        "</defs>"
    )
    add(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')

    add(f'<text x="{MARGIN_X}" y="36" font-size="20" font-weight="600" fill="#1c2b40">{TITLE}</text>')
    add(
        f'<text x="{MARGIN_X}" y="58" font-size="12" fill="#6b7580">'
        f"{len(nodes) - 2} nodes, {len(edges)} edges, {conditional_count} of them conditional. "
        f"Solid = always taken, dashed = chosen by a routing function. "
        f"Generated from build_graph() by scripts/plot_graph.py.</text>"
    )

    # Edges first so the boxes sit on top of them.
    for source, target, conditional in edges:
        path, lx, ly = edge_path(source, target)
        colour = EDGE_COLOURS["dashed" if conditional else "solid"]
        dash = ' stroke-dasharray="6 4"' if conditional else ""
        marker = "arrow-soft" if conditional else "arrow"
        add(
            f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.5"{dash} '
            f'marker-end="url(#{marker})"/>'
        )
        label = EDGE_LABELS.get((source, target))
        if label:
            add(_label(lx, ly, label))

    for name in nodes:
        x, y = node_box(name)
        cx = x + NODE_W / 2
        fill, stroke, text_colour = STYLES[ROLES[name]]
        add(
            f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="9" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
        )
        add(
            f'<text x="{cx:.0f}" y="{y + 20}" font-size="12.5" font-weight="600" '
            f'fill="{text_colour}" text-anchor="middle" '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">'
            f"{html.escape(name)}</text>"
        )
        subtitle = SUBTITLES.get(name, "")
        if subtitle:
            add(
                f'<text x="{cx:.0f}" y="{y + 36}" font-size="9.5" fill="#5f6a76" '
                f'text-anchor="middle">{html.escape(subtitle)}</text>'
            )

    add(_legend(y=height - 108))
    add(
        f'<text x="{MARGIN_X}" y="{height - 16}" font-size="10.5" fill="#6b7580">'
        "Node names, edges and directions are read from the compiled StateGraph. "
        "Edge labels come from scripts/plot_graph.py, since LangGraph does not expose "
        "the condition text.</text>"
    )
    add("</svg>")
    return "\n".join(out) + "\n"


def _legend(*, y: int) -> str:
    """Node roles on the left, edge styles on the right, on one row."""
    parts: list[str] = []
    roles = [
        ("entry", "entry"),
        ("routing", "routing"),
        ("retrieval", "retrieval"),
        ("agent", "agents"),
        ("hitl", "human in the loop"),
        ("action", "action"),
        ("output", "output"),
    ]
    x = MARGIN_X
    for role, caption in roles:
        fill, stroke, _ = STYLES[role]
        parts.append(
            f'<rect x="{x}" y="{y - 10}" width="17" height="13" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<text x="{x + 24}" y="{y + 1}" font-size="11" fill="#55606e">{caption}</text>'
        )
        x += 30 + len(caption) * 6.1

    x += 24
    for caption, colour, dashed in (
        ("unconditional edge", EDGE_COLOURS["solid"], False),
        ("conditional edge", EDGE_COLOURS["dashed"], True),
    ):
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append(
            f'<line x1="{x}" y1="{y - 3}" x2="{x + 34}" y2="{y - 3}" stroke="{colour}" '
            f'stroke-width="1.7"{dash} marker-end="url(#{"arrow-soft" if dashed else "arrow"})"/>'
        )
        parts.append(
            f'<text x="{x + 43}" y="{y + 1}" font-size="11" fill="#55606e">{caption}</text>'
        )
        x += 43 + len(caption) * 6.1
    return "".join(parts)


# --- entry point -----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the support graph as an SVG.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="Fail if the file is stale.")
    args = parser.parse_args(argv)

    nodes, edges = read_graph()

    problems = check_layout(nodes, edges)
    if problems:
        print("The layout needs updating in scripts/plot_graph.py:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    svg = render(nodes, edges)

    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist; run: python -m scripts.plot_graph", file=sys.stderr)
            return 1
        if args.out.read_text(encoding="utf-8") != svg:
            print(
                f"{args.out} is stale: the graph and the diagram disagree. "
                "Run: python -m scripts.plot_graph",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out.name} is current: {len(nodes)} nodes, {len(edges)} edges")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")
    conditional = sum(1 for _s, _t, c in edges if c)
    print(f"wrote {args.out} ({len(nodes)} nodes, {len(edges)} edges, {conditional} conditional)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
