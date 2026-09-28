"""The graph diagram stays in step with the compiled graph.

The SVG is generated from `build_graph().get_graph()`, so a test that checks the
generator is really testing that the committed picture cannot drift. CI also
runs `python scripts/plot_graph.py --check`, which compares the committed file
byte for byte.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from scripts import plot_graph

SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def graph() -> tuple[list[str], list[tuple[str, str, bool]]]:
    return plot_graph.read_graph()


def read_svg() -> str:
    return plot_graph.DEFAULT_OUT.read_text(encoding="utf-8")


class TestReadingTheGraph:
    def test_every_node_of_the_graph_is_read(self, graph) -> None:
        nodes, _edges = graph
        # 14 nodes plus the two pseudo-nodes LangGraph adds.
        assert len(nodes) == 16
        assert "__start__" in nodes and "__end__" in nodes

    def test_every_edge_is_read(self, graph) -> None:
        _nodes, edges = graph
        assert len(edges) == 27
        assert sum(1 for _s, _t, c in edges if c) == 18

    def test_reading_is_stable(self, graph) -> None:
        assert plot_graph.read_graph() == graph


class TestLayout:
    def test_the_layout_is_complete_and_correct(self, graph) -> None:
        assert plot_graph.check_layout(*graph) == []

    def test_a_new_node_is_reported(self, graph) -> None:
        nodes, edges = graph
        problems = plot_graph.check_layout(nodes + ["invented_node"], edges)
        assert any("invented_node" in p for p in problems)

    def test_a_removed_node_is_reported(self, graph) -> None:
        nodes, edges = graph
        problems = plot_graph.check_layout([n for n in nodes if n != "billing_agent"], edges)
        assert any("billing_agent" in p for p in problems)

    def test_a_backward_edge_is_reported(self, graph) -> None:
        nodes, edges = graph
        # respond -> guardrail would be drawn through everything in between.
        problems = plot_graph.check_layout(nodes, edges + [("respond", "guardrail", True)])
        assert any("does not point forwards" in p for p in problems)

    def test_an_unlabelled_conditional_edge_is_reported(self, graph) -> None:
        nodes, edges = graph
        # Forward-pointing, so the direction check passes, but it is neither a
        # labelled edge nor one of the obvious linear steps.
        fake = ("execute_action", "update_memory", True)
        problems = plot_graph.check_layout(nodes, edges + [fake])
        assert any("no label" in p for p in problems)

    def test_column_of_rejects_an_unknown_node(self) -> None:
        with pytest.raises(KeyError):
            plot_graph.column_of("nope")


class TestSvg:
    def test_the_svg_is_well_formed(self) -> None:
        root = ET.fromstring(read_svg())
        assert root.tag == f"{SVG_NS}svg"

    def test_every_node_is_drawn_with_its_name(self, graph) -> None:
        nodes, _edges = graph
        text = read_svg()
        for node in nodes:
            assert f">{node}</text>" in text, f"{node} is not labelled"

    def test_every_edge_is_drawn(self, graph) -> None:
        _nodes, edges = graph
        root = ET.fromstring(read_svg())
        paths = root.findall(f".//{SVG_NS}path")
        # 27 edges plus the two arrowheads in the marker definitions.
        assert len(paths) == len(edges) + 2

    def test_conditional_edges_are_dashed(self, graph) -> None:
        _nodes, edges = graph
        text = read_svg()
        dashed = text.count('stroke-dasharray="6 4"')
        # Every conditional edge is dashed.
        assert dashed >= sum(1 for _s, _t, c in edges if c)

    def test_every_labelled_edge_has_its_label(self) -> None:
        text = read_svg()
        for (source, target), label in plot_graph.EDGE_LABELS.items():
            assert f">{label}</text>" in text, f"label for {source} -> {target} is missing"

    def test_regenerating_is_byte_identical(self, graph) -> None:
        # A timestamp or a random id would make --check fail on every run.
        assert plot_graph.render(*graph) == read_svg()

    def test_the_check_passes(self) -> None:
        assert plot_graph.main(["--check"]) == 0
