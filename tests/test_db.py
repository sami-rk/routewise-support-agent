"""Tests for the SQLite schema and session helpers.

Every test runs against a temporary database file, so the developer's real
`data/support.db` is never touched.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from app.db.session import (
    close_connection,
    execute,
    get_connection,
    init_db,
    query_all,
    query_one,
    set_db_path,
    transaction,
)

EXPECTED_TABLES = {
    "customers",
    "subscriptions",
    "invoices",
    "refunds",
    "tickets",
    "customer_memory",
    "traces",
    "router_decisions",
    "llm_calls",
    "pending_actions",
}


@pytest.fixture
def db(tmp_path):
    """An initialised, empty database that is torn down afterwards."""
    set_db_path(tmp_path / "test.db")
    init_db()
    yield
    close_connection()


def table_names(db_path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def test_init_db_creates_every_table(tmp_path) -> None:
    path = tmp_path / "fresh.db"
    set_db_path(path)
    init_db()
    assert EXPECTED_TABLES <= table_names(path)
    close_connection()


def test_init_db_is_idempotent(db) -> None:
    init_db()  # must not raise on an already-initialised database
    execute("INSERT INTO customers (id, name, email) VALUES ('cus_x', 'X', 'x@example.com')")
    init_db()  # and must not drop existing rows
    assert query_one("SELECT * FROM customers WHERE id = 'cus_x'") is not None


def test_row_factory_returns_mapping_rows(db) -> None:
    execute(
        "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
        ("cus_1", "Ada Lovelace", "ada@example.com"),
    )
    row = query_one("SELECT * FROM customers WHERE id = ?", ("cus_1",))
    assert row is not None
    assert row["name"] == "Ada Lovelace"
    assert row["email"] == "ada@example.com"


def test_query_all_returns_a_list(db) -> None:
    for i in range(3):
        execute(
            "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
            (f"cus_{i}", f"Customer {i}", f"c{i}@example.com"),
        )
    rows = query_all("SELECT id FROM customers ORDER BY id")
    assert [r["id"] for r in rows] == ["cus_0", "cus_1", "cus_2"]


def test_query_one_returns_none_when_absent(db) -> None:
    assert query_one("SELECT * FROM customers WHERE id = ?", ("nope",)) is None


def test_transaction_rolls_back_on_error(db) -> None:
    with pytest.raises(RuntimeError):
        with transaction() as conn:
            conn.execute(
                "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
                ("cus_roll", "Roll Back", "rb@example.com"),
            )
            raise RuntimeError("boom")
    assert query_one("SELECT * FROM customers WHERE id = ?", ("cus_roll",)) is None


def test_transaction_commits_on_success(db) -> None:
    with transaction() as conn:
        conn.execute(
            "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
            ("cus_ok", "Commit Me", "ok@example.com"),
        )
    assert query_one("SELECT * FROM customers WHERE id = ?", ("cus_ok",)) is not None


def test_foreign_keys_are_enforced(db) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        execute(
            "INSERT INTO invoices (id, customer_id, amount, charged_at) VALUES (?, ?, ?, ?)",
            ("inv_orphan", "no_such_customer", 19.0, "2026-01-01"),
        )


def test_deleting_a_customer_cascades(db) -> None:
    execute("INSERT INTO customers (id, name, email) VALUES ('c1', 'A', 'a@x.com')")
    execute(
        "INSERT INTO subscriptions (id, customer_id, plan, price_monthly, devices_allowed) "
        "VALUES ('s1', 'c1', 'pro', 19.0, 5)"
    )
    execute(
        "INSERT INTO invoices (id, customer_id, amount, charged_at) VALUES ('i1', 'c1', 19.0, '2026-01-01')"
    )
    execute("INSERT INTO tickets (id, customer_id, subject) VALUES ('t1', 'c1', 'Help')")
    execute("DELETE FROM customers WHERE id = 'c1'")
    assert query_all("SELECT * FROM subscriptions") == []
    assert query_all("SELECT * FROM invoices") == []
    assert query_all("SELECT * FROM tickets") == []


def test_each_thread_gets_its_own_connection(db) -> None:
    main_conn = get_connection()
    seen: list[int] = []

    def worker() -> None:
        seen.append(id(get_connection()))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert len(seen) == 1
    assert seen[0] != id(main_conn)


def test_set_db_path_creates_the_parent_directory(tmp_path) -> None:
    nested = tmp_path / "deep" / "nested" / "support.db"
    set_db_path(nested)
    init_db()
    assert nested.exists()
    close_connection()
