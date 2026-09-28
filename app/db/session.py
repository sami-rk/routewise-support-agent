"""SQLite access: schema, connections and small query helpers.

Connections are per-thread because the agent graph and the FastAPI thread pool
both call into the database from more than one thread, and SQLite connections
are not safe to share across them.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_local = threading.local()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def get_connection() -> sqlite3.Connection:
    """The calling thread's connection to the support database."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect(get_settings().db_path)
        _local.conn = conn
    return conn


def set_db_path(db_path: Path | str) -> None:
    """Point the current thread at another database. Used by tests."""
    close_connection()
    path = Path(db_path)
    if not path.is_absolute():
        path = (Path(get_settings().db_path).parent / path).resolve()
    _local.conn = _connect(path)


def close_connection() -> None:
    """Close this thread's connection, if it has one."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


def init_db(db_path: Path | str | None = None) -> None:
    """Create every table. Safe to call repeatedly."""
    if db_path is not None:
        set_db_path(db_path)
    get_connection().executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    get_connection().commit()


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Run a block inside one transaction, committing or rolling back."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def query_one(sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
    return get_connection().execute(sql, params).fetchone()


def query_all(sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
    return get_connection().execute(sql, params).fetchall()


def execute(sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
    conn = get_connection()
    cursor = conn.execute(sql, params)
    conn.commit()
    return cursor
