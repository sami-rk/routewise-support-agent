"""Tests for ticket creation, priority and status."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.db.session import close_connection, query_one, set_db_path
from app.tools.tickets import (
    PRIORITIES,
    UnknownTicketError,
    create_ticket,
    get_ticket,
    get_ticket_status,
    list_tickets,
    priority_for,
    update_ticket_status,
)
from scripts.seed_db import seed

UTC = timezone.utc
NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def db(tmp_path):
    set_db_path(tmp_path / "tickets.db")
    seed()
    yield
    close_connection()


class TestCreateTicket:
    def test_ticket_is_stored_with_a_real_id(self, db) -> None:
        ticket = create_ticket("cus_ada", "Sync is broken", "It fails every time.", now=NOW)
        assert ticket["id"].startswith("tkt_")
        assert ticket["ticket_id"] == ticket["id"]
        assert get_ticket(ticket["id"]) is not None

    def test_defaults(self, db) -> None:
        ticket = create_ticket("cus_ada", "Question", now=NOW)
        assert ticket["status"] == "open"
        assert ticket["priority"] == "normal"
        assert ticket["category"] == "general"

    def test_subject_is_trimmed_and_capped(self, db) -> None:
        ticket = create_ticket("cus_ada", "  " + "x" * 500 + "  ", now=NOW)
        assert len(ticket["subject"]) == 200
        assert ticket["subject"].startswith("x")

    def test_empty_subject_gets_a_default(self, db) -> None:
        assert create_ticket("cus_ada", "   ", now=NOW)["subject"] == "Support request"

    def test_unknown_category_falls_back(self, db) -> None:
        assert create_ticket("cus_ada", "S", category="nonsense", now=NOW)["category"] == "general"


class TestPriority:
    @pytest.mark.parametrize(
        ("urgency", "expected"),
        [(0.0, "low"), (0.2, "low"), (0.3, "normal"), (0.6, "high"), (0.8, "urgent"), (1.0, "urgent")],
    )
    def test_urgency_maps_to_priority(self, urgency: float, expected: str) -> None:
        assert priority_for(urgency) == expected

    def test_missing_urgency_is_normal(self) -> None:
        assert priority_for(None) == "normal"

    @pytest.mark.parametrize(
        "subject",
        ["Data loss after the update", "I lost my files", "Account hacked", "Security breach", "I will sue"],
    )
    def test_serious_wording_is_urgent_regardless_of_urgency(self, subject: str) -> None:
        assert priority_for(0.0, subject) == "urgent"

    def test_priority_comes_from_urgency_when_not_forced(self, db) -> None:
        assert create_ticket("cus_ada", "Slow sync", urgency=0.9, now=NOW)["priority"] == "urgent"
        assert create_ticket("cus_ada", "Nice question", urgency=0.1, now=NOW)["priority"] == "low"

    def test_forced_priority_wins(self, db) -> None:
        assert create_ticket("cus_ada", "S", priority="high", urgency=0.0, now=NOW)["priority"] == "high"

    def test_every_priority_is_a_known_value(self, db) -> None:
        for urgency in (0.0, 0.5, 1.0):
            assert create_ticket("cus_ada", "S", urgency=urgency, now=NOW)["priority"] in PRIORITIES


class TestStatus:
    def test_status_is_reported(self, db) -> None:
        ticket = create_ticket("cus_ada", "S", now=NOW)
        status = get_ticket_status(ticket["id"])
        assert status["status"] == "open"
        assert status["subject"] == "S"

    def test_unknown_ticket_raises(self, db) -> None:
        with pytest.raises(UnknownTicketError):
            get_ticket_status("tkt_nope")

    def test_status_can_be_updated(self, db) -> None:
        ticket = create_ticket("cus_ada", "S", now=NOW)
        updated = update_ticket_status(ticket["id"], "resolved")
        assert updated["status"] == "resolved"
        assert get_ticket_status(ticket["id"])["status"] == "resolved"

    def test_updating_an_unknown_ticket_raises(self, db) -> None:
        with pytest.raises(UnknownTicketError):
            update_ticket_status("tkt_nope", "closed")


class TestListing:
    def test_lists_all_tickets_newest_first(self, db) -> None:
        first = create_ticket("cus_ada", "First", now=NOW)
        second = create_ticket("cus_ada", "Second", now=NOW)
        ids = [t["id"] for t in list_tickets(limit=10)]
        assert second["id"] in ids and first["id"] in ids

    def test_filters_by_status(self, db) -> None:
        create_ticket("cus_ada", "Open one", now=NOW)
        closed = create_ticket("cus_ada", "Closed one", now=NOW)
        update_ticket_status(closed["id"], "closed")
        open_ids = [t["id"] for t in list_tickets("open", limit=50)]
        assert closed["id"] not in open_ids

    def test_filters_by_customer(self, db) -> None:
        mine = create_ticket("cus_ada", "Mine", now=NOW)
        theirs = create_ticket("cus_grace", "Theirs", now=NOW)
        ids = [t["id"] for t in list_tickets(customer_id="cus_ada", limit=50)]
        assert mine["id"] in ids
        assert theirs["id"] not in ids

    def test_limit_is_capped(self, db) -> None:
        assert len(list_tickets(limit=10_000)) <= 200


def test_ticket_belongs_to_a_real_customer(db) -> None:
    ticket = create_ticket("cus_ada", "S", now=NOW)
    row = query_one("SELECT customer_id FROM tickets WHERE id = ?", (ticket["id"],))
    assert row["customer_id"] == "cus_ada"
