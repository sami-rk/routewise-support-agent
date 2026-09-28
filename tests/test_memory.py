"""Tests for long-term customer memory."""

from __future__ import annotations

import pytest

from app.db.session import close_connection, query_one, set_db_path
from app.memory.long_term import (
    MAX_WORDS,
    NOTHING,
    clear_memory,
    customer_facts,
    get_memory,
    is_worth_storing,
    set_memory,
    trim_summary,
)
from scripts.seed_db import seed


@pytest.fixture
def db(tmp_path):
    set_db_path(tmp_path / "memory.db")
    seed()
    yield
    close_connection()


class TestTrim:
    def test_short_text_is_untouched(self) -> None:
        assert trim_summary("Uses the Pro plan on Windows.") == "Uses the Pro plan on Windows."

    def test_whitespace_is_collapsed(self) -> None:
        assert trim_summary("  a   b\n\nc  ") == "a b c"

    def test_long_text_is_cut_to_the_limit(self) -> None:
        summary = ". ".join(f"Sentence number {i}" for i in range(80))
        trimmed = trim_summary(summary)
        assert len(trimmed.split()) <= MAX_WORDS

    def test_a_cut_prefers_a_sentence_boundary(self) -> None:
        summary = " ".join(f"word{i}" for i in range(60)) + ". " + " ".join(
            f"tail{i}" for i in range(120)
        )
        trimmed = trim_summary(summary)
        assert trimmed.endswith(".")

    def test_a_truncated_sentence_still_ends_cleanly(self) -> None:
        trimmed = trim_summary(" ".join(f"word{i}" for i in range(400)))
        assert not trimmed.endswith(",") and not trimmed.endswith(";")
        assert trimmed.endswith(".")

    def test_empty_stays_empty(self) -> None:
        assert trim_summary("") == ""
        assert trim_summary("   ") == ""


class TestWorthStoring:
    def test_a_real_summary_is_worth_storing(self) -> None:
        assert is_worth_storing("On the Pro plan, syncs five devices, Windows desktop.") is True

    def test_the_nothing_marker_is_rejected(self) -> None:
        assert is_worth_storing(NOTHING) is False
        assert is_worth_storing("nothing") is False

    def test_an_empty_summary_is_rejected(self) -> None:
        assert is_worth_storing("") is False

    def test_a_two_word_stub_is_rejected(self) -> None:
        assert is_worth_storing("thanks bye") is False


class TestStorage:
    def test_memory_round_trips(self, db) -> None:
        set_memory("cus_ada", "On the Pro plan. Syncs from two machines.")
        assert "Pro plan" in get_memory("cus_ada")

    def test_memory_is_replaced_not_appended(self, db) -> None:
        set_memory("cus_ada", "First summary about the customer.")
        set_memory("cus_ada", "Second summary about the customer.")
        stored = get_memory("cus_ada")
        assert "Second" in stored
        assert "First" not in stored
        assert query_one("SELECT COUNT(*) AS n FROM customer_memory WHERE customer_id='cus_ada'")["n"] == 1

    def test_stored_memory_respects_the_word_limit(self, db) -> None:
        set_memory("cus_ada", " ".join(f"w{i}" for i in range(400)))
        assert len(get_memory("cus_ada").split()) <= MAX_WORDS

    def test_no_memory_reads_as_empty(self, db) -> None:
        assert get_memory("cus_grace") == ""

    def test_missing_customer_id_reads_as_empty(self) -> None:
        assert get_memory("") == ""
        assert get_memory(None) == ""

    def test_clearing_forgets(self, db) -> None:
        set_memory("cus_ada", "Something worth remembering about this customer.")
        assert clear_memory("cus_ada") is True
        assert get_memory("cus_ada") == ""

    def test_clearing_nothing_reports_false(self, db) -> None:
        assert clear_memory("cus_grace") is False

    def test_clearing_a_missing_customer_is_safe(self) -> None:
        assert clear_memory("") is False

    def test_memory_is_removed_with_the_customer(self, db) -> None:
        set_memory("cus_ada", "Something worth remembering about this customer.")
        from app.db.session import execute

        execute("DELETE FROM customers WHERE id = 'cus_ada'")
        assert get_memory("cus_ada") == ""


class TestCustomerFacts:
    def test_facts_include_the_plan_and_devices(self, db) -> None:
        facts = customer_facts("cus_ada")
        assert facts["name"] == "Ada Lovelace"
        assert facts["plan"] == "pro"
        assert facts["devices"] == "3 of 5"

    def test_unlimited_devices_are_described(self, db) -> None:
        assert customer_facts("cus_barbara")["devices"] == "7 of unlimited"

    def test_memory_is_included_once_stored(self, db) -> None:
        assert customer_facts("cus_ada")["memory"] is None
        set_memory("cus_ada", "Uses Windows and the desktop app daily.")
        assert "Windows" in customer_facts("cus_ada")["memory"]

    def test_unknown_customer_gives_empty_facts(self, db) -> None:
        facts = customer_facts("cus_nobody")
        assert facts["name"] is None and facts["plan"] is None
