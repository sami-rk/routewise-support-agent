"""Tests for the refund policy and the 14-day window.

The window boundary is the interesting part, so the tests pin "14 days old is
inside, 15 is outside" and check the truncation of partial days, the
`REFUND_AUTO_LIMIT` split, and that a refund cannot be issued twice.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.config import Settings
from app.db.session import close_connection, query_one, set_db_path
from app.tools.refunds import (
    RefundError,
    UnknownInvoiceError,
    cancel_subscription,
    check_refund_eligibility,
    create_refund,
    find_refundable_invoice,
)
from scripts.seed_db import seed

UTC = timezone.utc
NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def db(tmp_path):
    set_db_path(tmp_path / "refunds.db")
    seed()
    yield
    close_connection()


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, refund_window_days=14, refund_auto_limit=20.0)


def charge_invoice(invoice_id: str, days_ago: float) -> None:
    """Backdate an invoice so the window can be tested precisely."""
    charged = (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
    from app.db.session import execute

    execute("UPDATE invoices SET charged_at = ?, status = 'paid' WHERE id = ?", (charged, invoice_id))


class TestWindow:
    def test_fresh_charge_is_eligible(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 0)
        result = check_refund_eligibility("inv_ada_current", settings=settings, now=NOW)
        assert result["eligible"] is True
        assert result["age_days"] == 0
        assert result["max_amount"] == pytest.approx(19.0)

    def test_last_day_of_the_window_is_inside(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 14)
        result = check_refund_eligibility("inv_ada_current", settings=settings, now=NOW)
        assert result["eligible"] is True, "day 14 is the last eligible day"
        assert result["age_days"] == 14

    def test_one_day_past_the_window_is_outside(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 15)
        result = check_refund_eligibility("inv_ada_current", settings=settings, now=NOW)
        assert result["eligible"] is False
        assert result["max_amount"] == 0.0
        assert "14 days" in result["reason"]

    def test_partial_days_truncate_rather_than_round(self, db, settings) -> None:
        # 14 days and 23 hours ago is still day 14, so still inside.
        charge_invoice("inv_ada_current", 14 + (23 / 24))
        assert check_refund_eligibility("inv_ada_current", settings=settings, now=NOW)["eligible"] is True

        # 15 days and 1 hour ago is day 15, so outside.
        charge_invoice("inv_ada_current", 15 + (1 / 24))
        assert check_refund_eligibility("inv_ada_current", settings=settings, now=NOW)["eligible"] is False

    def test_a_custom_window_is_honoured(self, db) -> None:
        thirty = Settings(_env_file=None, refund_window_days=30, refund_auto_limit=20.0)
        charge_invoice("inv_ada_current", 20)
        assert check_refund_eligibility("inv_ada_current", settings=thirty, now=NOW)["eligible"] is True

    def test_unknown_invoice_raises(self, db, settings) -> None:
        with pytest.raises(UnknownInvoiceError):
            check_refund_eligibility("inv_nope", settings=settings, now=NOW)

    def test_unpaid_charge_is_not_refundable(self, db, settings) -> None:
        from app.db.session import execute

        execute("UPDATE invoices SET status = 'failed' WHERE id = 'inv_ada_current'")
        result = check_refund_eligibility("inv_ada_current", settings=settings, now=NOW)
        assert result["eligible"] is False
        assert "failed" in result["reason"]


class TestAutoLimit:
    def test_small_refund_does_not_need_approval(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 1)
        result = check_refund_eligibility("inv_ada_current", settings=settings, now=NOW)
        assert result["eligible"] is True
        assert result["needs_approval"] is False
        assert result["amount"] == pytest.approx(19.0)

    def test_large_refund_needs_approval(self, db, settings) -> None:
        charge_invoice("inv_barbara_current", 2)
        result = check_refund_eligibility("inv_barbara_current", settings=settings, now=NOW)
        assert result["eligible"] is True
        assert result["needs_approval"] is True
        assert "specialist" in result["reason"]

    def test_exactly_the_limit_needs_no_approval(self, db, settings) -> None:
        from app.db.session import execute

        execute("UPDATE invoices SET amount = 20.0 WHERE id = 'inv_ada_current'")
        charge_invoice("inv_ada_current", 1)
        result = check_refund_eligibility("inv_ada_current", settings=settings, now=NOW)
        assert result["needs_approval"] is False

    def test_one_cent_over_the_limit_needs_approval(self, db, settings) -> None:
        from app.db.session import execute

        execute("UPDATE invoices SET amount = 20.01 WHERE id = 'inv_ada_current'")
        charge_invoice("inv_ada_current", 1)
        assert check_refund_eligibility("inv_ada_current", settings=settings, now=NOW)["needs_approval"] is True


class TestCreateRefund:
    def test_refund_is_recorded(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 2)
        result = create_refund("inv_ada_current", 19.0, "customer asked", settings=settings, now=NOW)
        assert result["status"] == "succeeded"
        assert result["approved_by"] == "automatic"
        row = query_one("SELECT * FROM refunds WHERE id = ?", (result["refund_id"],))
        assert row is not None
        assert row["amount"] == pytest.approx(19.0)

    def test_refund_outside_the_window_is_refused(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 20)
        with pytest.raises(RefundError, match="outside the window"):
            create_refund("inv_ada_current", 19.0, "too late", settings=settings, now=NOW)

    def test_refund_over_the_auto_limit_is_refused_without_approval(self, db, settings) -> None:
        charge_invoice("inv_barbara_current", 2)
        with pytest.raises(RefundError, match="needs a support specialist"):
            create_refund("inv_barbara_current", 49.0, "big", settings=settings, now=NOW)

    def test_large_refund_succeeds_with_an_approver(self, db, settings) -> None:
        charge_invoice("inv_barbara_current", 2)
        result = create_refund(
            "inv_barbara_current", 49.0, "approved", approved_by="staff:kim", settings=settings, now=NOW
        )
        assert result["approved_by"] == "staff:kim"

    def test_refund_cannot_exceed_the_charge(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 1)
        with pytest.raises(RefundError, match="more than"):
            create_refund("inv_ada_current", 500.0, "greedy", settings=settings, now=NOW)

    def test_refund_must_be_positive(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 1)
        with pytest.raises(RefundError, match="positive"):
            create_refund("inv_ada_current", 0, "zero", settings=settings, now=NOW)

    def test_second_refund_uses_the_remaining_amount(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 1)
        create_refund("inv_ada_current", 10.0, "part one", settings=settings, now=NOW)
        eligibility = check_refund_eligibility("inv_ada_current", settings=settings, now=NOW)
        assert eligibility["already_refunded"] == pytest.approx(10.0)
        assert eligibility["max_amount"] == pytest.approx(9.0)

    def test_cannot_refund_the_same_money_twice(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 1)
        create_refund("inv_ada_current", 19.0, "full", settings=settings, now=NOW)
        eligibility = check_refund_eligibility("inv_ada_current", settings=settings, now=NOW)
        assert eligibility["eligible"] is False
        assert "already been refunded" in eligibility["reason"]
        with pytest.raises(RefundError):
            create_refund("inv_ada_current", 1.0, "again", settings=settings, now=NOW)


class TestFindRefundableInvoice:
    def test_finds_the_newest_eligible_charge(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 2)
        charge_invoice("inv_ada_old", 200)
        found = find_refundable_invoice("cus_ada", settings=settings, now=NOW)
        assert found is not None
        assert found["invoice_id"] == "inv_ada_current"

    def test_skips_ineligible_charges(self, db, settings) -> None:
        charge_invoice("inv_ada_current", 60)
        charge_invoice("inv_ada_old", 300)
        assert find_refundable_invoice("cus_ada", settings=settings, now=NOW) is None


class TestCancelSubscription:
    def test_cancellation_records_the_change(self, db) -> None:
        result = cancel_subscription("cus_ada", approved_by="cus_ada", now=NOW)
        assert result["status"] == "cancelled"
        assert result["plan"] == "pro"
        row = query_one("SELECT status FROM subscriptions WHERE id = 'sub_ada'")
        assert row["status"] == "cancelled"

    def test_cannot_cancel_twice(self, db) -> None:
        cancel_subscription("cus_ada", now=NOW)
        with pytest.raises(RefundError, match="already cancelled"):
            cancel_subscription("cus_ada", now=NOW)

    def test_unknown_customer_has_nothing_to_cancel(self, db) -> None:
        with pytest.raises(RefundError, match="could not find a subscription"):
            cancel_subscription("cus_nobody")
