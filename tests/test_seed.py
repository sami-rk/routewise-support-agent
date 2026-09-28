"""Tests for the demo seed data.

The refund rule is the interesting part: the seed must produce invoices on both
sides of the 14-day window so `check_refund_eligibility` has something to
decide, including a customer whose latest charge is old.
"""

from __future__ import annotations

import pytest

from app.db.session import close_connection, query_all, query_one, set_db_path
from scripts.seed_db import CUSTOMERS, seed

pytestmark = pytest.mark.usefixtures("seeded_db")


@pytest.fixture
def seeded_db(tmp_path):
    set_db_path(tmp_path / "seed.db")
    seed()
    yield
    close_connection()


def test_seed_creates_ten_customers(seeded_db) -> None:
    assert query_one("SELECT COUNT(*) AS n FROM customers")["n"] == len(CUSTOMERS) == 10


def test_every_plan_is_represented(seeded_db) -> None:
    plans = {row["plan"] for row in query_all("SELECT DISTINCT plan FROM subscriptions")}
    assert plans == {"basic", "pro", "business"}


def test_prices_match_the_published_plans(seeded_db) -> None:
    prices = {row["plan"]: row["price_monthly"] for row in query_all("SELECT plan, price_monthly FROM subscriptions")}
    assert prices == {"basic": 9.0, "pro": 19.0, "business": 49.0}


def test_business_plan_has_unlimited_devices(seeded_db) -> None:
    business = query_one("SELECT devices_allowed FROM subscriptions WHERE plan = 'business'")
    # 0 is the sentinel for unlimited, as in the knowledge base.
    assert business["devices_allowed"] == 0


def test_one_subscription_is_cancelled(seeded_db) -> None:
    cancelled = query_all("SELECT id FROM subscriptions WHERE status = 'cancelled'")
    assert len(cancelled) == 1


def test_invoices_exist_on_both_sides_of_the_refund_window(seeded_db) -> None:
    rows = query_all(
        "SELECT customer_id, "
        "  CAST(julianday('now') - julianday(charged_at) AS INTEGER) AS age_days "
        "FROM invoices WHERE id LIKE '%\\_current' ESCAPE '\\'"
    )
    ages = {row["customer_id"]: row["age_days"] for row in rows}
    assert len(ages) == 10
    assert any(age <= 14 for age in ages.values()), "need an invoice inside the window"
    assert any(age > 14 for age in ages.values()), "need an invoice outside the window"


def test_a_one_day_invoice_exists_for_the_auto_approval_path(seeded_db) -> None:
    recent = query_one(
        "SELECT id FROM invoices "
        "WHERE CAST(julianday('now') - julianday(charged_at) AS INTEGER) = 1"
    )
    assert recent is not None


def test_older_invoices_are_far_outside_the_window(seeded_db) -> None:
    old = query_all(
        "SELECT CAST(julianday('now') - julianday(charged_at) AS INTEGER) AS age_days "
        "FROM invoices WHERE id LIKE '%\\_old' ESCAPE '\\'"
    )
    assert all(row["age_days"] > 14 for row in old)


def test_tickets_are_seeded_with_priority_and_category(seeded_db) -> None:
    rows = query_all("SELECT priority, category FROM tickets")
    assert len(rows) == 3
    assert {row["priority"] for row in rows} <= {"low", "normal", "high", "urgent"}
    assert {row["category"] for row in rows} <= {"technical", "billing", "account", "general"}


def test_tickets_belong_to_real_customers(seeded_db) -> None:
    known = {cid for cid, *_ in CUSTOMERS}
    owners = {row["customer_id"] for row in query_all("SELECT customer_id FROM tickets")}
    assert owners <= known


def test_seeding_twice_is_idempotent(seeded_db) -> None:
    seed()
    assert query_one("SELECT COUNT(*) AS n FROM customers")["n"] == 10
    assert query_one("SELECT COUNT(*) AS n FROM invoices")["n"] == 20


def test_reseeding_clears_refunds(seeded_db) -> None:
    # A refund from a previous run would make the demo customer look
    # already-refunded, and the refund flow would decline them.
    from app.db.session import execute
    from app.tools.refunds import create_refund

    execute("UPDATE invoices SET charged_at = datetime('now') WHERE id = 'inv_ada_current'")
    create_refund("inv_ada_current", 19.0, "from a previous run")
    assert query_one("SELECT COUNT(*) AS n FROM refunds")["n"] == 1

    seed()
    assert query_one("SELECT COUNT(*) AS n FROM refunds")["n"] == 0


def test_reseeding_clears_pending_approvals(seeded_db) -> None:
    from app.db.session import execute

    execute(
        "INSERT INTO pending_actions (thread_id, customer_id, action, payload, mode) "
        "VALUES ('t1', 'cus_ada', 'refund', '{}', 'staff_approve')"
    )
    seed()
    assert query_one("SELECT COUNT(*) AS n FROM pending_actions")["n"] == 0


def test_customer_emails_are_unique(seeded_db) -> None:
    emails = [row["email"] for row in query_all("SELECT email FROM customers")]
    assert len(emails) == len(set(emails))
