"""Seed the support database with demo customers, invoices and tickets.

Ten customers across Basic, Pro and Business, with invoices on both sides of the
14-day refund window so the refund rule can be exercised, one cancelled
subscription, and a few tickets. Running it twice leaves the same data.

    python -m scripts.seed_db
    python -m scripts.seed_db --db-path ./data/other.db
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import validate_settings
from app.db.session import close_connection, execute, init_db, query_one, set_db_path

UTC = timezone.utc

# (id, name, email, plan, price, devices_allowed, devices_used, status, days_since_charge)
CUSTOMERS = [
    ("cus_ada", "Ada Lovelace", "ada@example.com", "pro", 19.0, 5, 3, "active", 5),
    ("cus_grace", "Grace Hopper", "grace@example.com", "basic", 9.0, 2, 1, "active", 3),
    ("cus_alan", "Alan Turing", "alan@example.com", "business", 49.0, 0, 12, "active", 20),
    ("cus_katherine", "Katherine Johnson", "kj@example.com", "pro", 19.0, 5, 2, "active", 45),
    ("cus_margaret", "Margaret Hamilton", "margaret@example.com", "basic", 9.0, 2, 2, "active", 1),
    ("cus_linus", "Linus Torvalds", "linus@example.com", "pro", 19.0, 5, 4, "cancelled", 60),
    ("cus_barbara", "Barbara Liskov", "barbara@example.com", "business", 49.0, 0, 7, "active", 8),
    ("cus_donald", "Donald Knuth", "donald@example.com", "basic", 9.0, 2, 1, "active", 12),
    ("cus_edsger", "Edsger Dijkstra", "edsger@example.com", "pro", 19.0, 5, 5, "active", 30),
    ("cus_radia", "Radia Perlman", "radia@example.com", "business", 49.0, 0, 40, "active", 16),
]

TICKETS = [
    ("cus_alan", "Files vanished after the March update", "Synced folders are empty.", "high", "technical"),
    ("cus_ada", "Duplicate charge in March", "Billed twice for the same month.", "normal", "billing"),
    ("cus_edsger", "Cannot sign in after password reset", "The reset link says it is expired.", "normal", "account"),
]


def iso(days_ago: float, *, base: datetime | None = None) -> str:
    """A UTC timestamp `days_ago` days before now, in SQLite's text format."""
    now = base or datetime.now(UTC)
    return (now - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


def seed(db_path: Path | None = None) -> dict[str, int]:
    """Create the schema and insert the demo data. Returns row counts.

    Customers, subscriptions, invoices and tickets are replaced. Refunds and
    pending approvals are cleared, because they are history rather than seed
    data: leaving a refund from a previous run would make the demo customers
    look already-refunded and the refund flow would decline them.
    """
    set_db_path(db_path) if db_path else init_db()
    init_db()

    now = datetime.now(UTC)
    counts = {"customers": 0, "subscriptions": 0, "invoices": 0, "tickets": 0, "refunds": 0}

    execute("DELETE FROM refunds")
    execute("DELETE FROM pending_actions")

    for (
        cid,
        name,
        email,
        plan,
        price,
        devices_allowed,
        devices_used,
        status,
        days_ago,
    ) in CUSTOMERS:
        sub_id = f"sub_{cid.removeprefix('cus_')}"
        execute(
            "INSERT OR REPLACE INTO customers (id, name, email, created_at) VALUES (?, ?, ?, ?)",
            (cid, name, email, iso(400, base=now)),
        )
        execute(
            "INSERT OR REPLACE INTO subscriptions "
            "(id, customer_id, plan, price_monthly, devices_allowed, devices_used, status, "
            " renews_at, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sub_id,
                cid,
                plan,
                price,
                devices_allowed,
                devices_used,
                status,
                iso(-25, base=now),  # renews in the future
                iso(365, base=now),
            ),
        )
        execute(
            "INSERT OR REPLACE INTO invoices "
            "(id, customer_id, subscription_id, amount, currency, status, charged_at) "
            "VALUES (?, ?, ?, ?, 'USD', 'paid', ?)",
            (f"inv_{cid.removeprefix('cus_')}_current", cid, sub_id, price, iso(days_ago, base=now)),
        )
        # An older invoice, well outside the refund window.
        execute(
            "INSERT OR REPLACE INTO invoices "
            "(id, customer_id, subscription_id, amount, currency, status, charged_at) "
            "VALUES (?, ?, ?, ?, 'USD', 'paid', ?)",
            (f"inv_{cid.removeprefix('cus_')}_old", cid, sub_id, price, iso(days_ago + 90, base=now)),
        )
        counts["customers"] += 1
        counts["subscriptions"] += 1
        counts["invoices"] += 2

    for index, (cid, subject, body, priority, category) in enumerate(TICKETS, start=1):
        execute(
            "INSERT OR REPLACE INTO tickets "
            "(id, customer_id, subject, body, priority, category, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)",
            (f"tkt_seed_{index}", cid, subject, body, priority, category, iso(2, base=now), iso(1, base=now)),
        )
        counts["tickets"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the CloudSync Pro support database.")
    parser.add_argument("--db-path", default=None, help="Target database (default: DB_PATH from settings).")
    args = parser.parse_args()
    validate_settings()

    counts = seed(Path(args.db_path) if args.db_path else None)
    customers = query_one("SELECT COUNT(*) AS n FROM customers")
    invoices = query_one("SELECT COUNT(*) AS n FROM invoices")
    print(f"Seeded {customers['n']} customers, {invoices['n']} invoices, {counts['tickets']} tickets.")
    print("Refund-window coverage:")
    for row in _window_report():
        print(f"  {row['customer_id']:<16} {row['age_days']:>3} days old  ${row['amount']:>6.2f}  {row['verdict']}")
    close_connection()


def _window_report() -> list[dict]:
    """One invoice per customer with its refund-window verdict, for the summary."""
    rows = []
    for cid, *_ in CUSTOMERS:
        row = query_one(
            "SELECT amount, charged_at, julianday('now') - julianday(charged_at) AS age_days "
            "FROM invoices WHERE customer_id = ? AND id LIKE '%_current'",
            (cid,),
        )
        if row is None:
            continue
        age = int(row["age_days"])
        rows.append(
            {
                "customer_id": cid,
                "amount": row["amount"],
                "age_days": age,
                "verdict": "inside 14 days" if age <= 14 else "outside 14 days",
            }
        )
    return rows


if __name__ == "__main__":
    main()
