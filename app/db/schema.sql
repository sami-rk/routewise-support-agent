-- CloudSync Pro support agent: SQLite schema.
--
-- Column lists follow section 13 of the spec. `llm_calls` is an addition: the
-- per-model request counts, fallback rate and rate-limit hits in section 10.2
-- need one row per attempt, which a per-node trace cannot express.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    email      TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS subscriptions (
    id              TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    plan            TEXT NOT NULL,
    price_monthly   REAL NOT NULL,
    devices_allowed INTEGER NOT NULL,
    devices_used    INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'active',
    renews_at       TEXT,
    started_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_customer ON subscriptions(customer_id);

CREATE TABLE IF NOT EXISTS invoices (
    id              TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    subscription_id TEXT REFERENCES subscriptions(id) ON DELETE SET NULL,
    amount          REAL NOT NULL,
    currency        TEXT NOT NULL DEFAULT 'USD',
    status          TEXT NOT NULL DEFAULT 'paid',
    charged_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices(customer_id, charged_at DESC);

CREATE TABLE IF NOT EXISTS refunds (
    id         TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
    amount     REAL NOT NULL,
    reason     TEXT,
    status     TEXT NOT NULL DEFAULT 'succeeded',
    approved_by TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_refunds_invoice ON refunds(invoice_id);

CREATE TABLE IF NOT EXISTS tickets (
    id          TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    subject     TEXT NOT NULL,
    body        TEXT,
    priority    TEXT NOT NULL DEFAULT 'normal',
    category    TEXT NOT NULL DEFAULT 'general',
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status, priority);

CREATE TABLE IF NOT EXISTS customer_memory (
    customer_id TEXT PRIMARY KEY REFERENCES customers(id) ON DELETE CASCADE,
    summary     TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS traces (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    thread_id      TEXT NOT NULL,
    node           TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    duration_ms    REAL NOT NULL DEFAULT 0,
    input_summary  TEXT,
    output_summary TEXT,
    model          TEXT,
    tokens_in      INTEGER NOT NULL DEFAULT 0,
    tokens_out     INTEGER NOT NULL DEFAULT 0,
    error          TEXT
);

CREATE INDEX IF NOT EXISTS idx_traces_thread ON traces(thread_id, id);

CREATE TABLE IF NOT EXISTS router_decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id    TEXT NOT NULL,
    message      TEXT NOT NULL,
    answers_json TEXT NOT NULL,
    probs_json   TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_router_decisions_thread ON router_decisions(thread_id, id);

CREATE TABLE IF NOT EXISTS llm_calls (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT,
    thread_id  TEXT,
    role       TEXT NOT NULL,
    model      TEXT NOT NULL,
    attempt    INTEGER NOT NULL DEFAULT 0,
    status     TEXT NOT NULL,
    error      TEXT,
    duration_ms REAL NOT NULL DEFAULT 0,
    tokens_in  INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON llm_calls(model);

-- Customer confirmation of a cancellation can arrive as a fresh turn, so the
-- pending action has to outlive a single graph run.
CREATE TABLE IF NOT EXISTS pending_actions (
    thread_id   TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    action      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    mode        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'awaiting',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_pending_actions_status ON pending_actions(status, created_at);
