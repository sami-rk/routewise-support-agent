# CloudSync Pro support agent

A customer-support agent for a fictional cloud-storage product, built on
**LangGraph**, routed by the local **Laya** decision model, with every LLM call
going through **OpenRouter on free-tier models only**.

Routing costs nothing on the free tier, because routing is a local forward pass.
The model decides intent, urgency, frustration, whether a person is needed and
whether a message is a prompt injection — in one pass, with calibrated
probabilities — and the LLM only ever writes prose and calls tools.

```
Streamlit console  ──HTTP/SSE──►  FastAPI  ──►  LangGraph app (compiled, SqliteSaver)
                                                    │
        ┌───────────────────────────────────────────┼────────────────────────────┐
        ▼                                           ▼                            ▼
  Laya router (local)                  OpenRouter free LLM            Tools + RAG + SQLite
  choice / score / noul                 replies, tool calls            subscriptions, tickets, memory
```

## What it does

- **Routes with a local model.** Laya (`convaiinnovations/laya`, ModernBERT-large,
  512-token context) answers five typed questions per turn. No OpenRouter request.
- **Answers from a knowledge base.** Eight documents in `data/knowledge_base`,
  heading-aware chunks, local `bge-small-en-v1.5` embeddings, a persisted FAISS
  index, a similarity floor, and citations like `[pricing_plans.md › The plans]`.
- **Acts on real records.** Subscription, invoice, refund and ticket tools over
  SQLite. A refund on a 20-day-old invoice is declined with the policy cited.
- **Asks before it moves money.** Refunds above `REFUND_AUTO_LIMIT` interrupt for
  staff approval; cancellations interrupt for the customer's confirmation. Nothing
  runs until the decision arrives, and the thread resumes from the checkpointer.
- **Is observable.** Per-node traces, router decisions with full probability
  distributions, and real metrics at `GET /metrics`.
- **Runs offline for tests.** `FakeRouter` plus a scripted model means the whole
  suite needs no API key, no Laya download and no embedding model.

## Quick start

```bash
python3 -m venv --without-pip .venv          # see "Python setup" if ensurepip is missing
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install -r requirements.txt

cp .env.example .env                         # add OPENROUTER_API_KEY for live LLM calls
.venv/bin/python -m scripts.seed_db          # 10 customers, invoices both sides of 14 days
.venv/bin/python -m scripts.build_kb         # build the FAISS index (local embeddings)
```

Run the API:

```bash
.venv/bin/uvicorn app.main:app --port 8000
# docs at http://localhost:8000/docs, health at /health
```

Run the console:

```bash
cd frontend && ../.venv/bin/streamlit run streamlit_app.py
```

Or skip the servers and talk to the graph in the terminal:

```bash
.venv/bin/python -m app.cli --router fake     # keywords, no downloads
.venv/bin/python -m app.cli                    # Laya
```

### Python setup

This box has only Python 3.14 and no `ensurepip`, so the venv is bootstrapped by
hand as above. On a normal machine `python3 -m venv .venv` is enough. The pinned
versions all have cp314 wheels: `langgraph==1.2.12`, `torch==2.14.0`,
`faiss-cpu==1.15.1`, `sentence-transformers==6.1.0`, `laya==0.3.21`.

Install CPU-only torch to avoid ~3 GB of CUDA wheels on a machine with no GPU:

```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
```

## Trying it

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"customer_id":"cus_ada","message":"How much is the Pro plan and how many devices?"}'
```

```json
{"status": "done", "intent": "pricing", "intent_conf": 0.9,
 "response": "The Pro plan is $19 per month and includes 1 TB, syncing across 5 devices. [pricing_plans.md > The plans]",
 "citations": ["[pricing_plans.md > The plans]"]}
```

Worth trying, because each takes a different path:

| Message | What happens |
|---|---|
| `I was charged 5 days ago and want a refund` | `billing_agent`, eligibility checked, **refunded automatically** ($19 < the $20 limit) |
| `I want a refund of the 49 dollars` (as `cus_barbara`) | pauses at `interrupt()` with `mode: staff_approve`; approve it at `POST /approvals/{thread_id}` |
| `I want my money back` (as `cus_alan`, charged 20 days ago) | declined politely, the 14-day policy cited, nothing refunded |
| `Cancel my subscription` | pauses with `mode: customer_confirm`; nothing is cancelled until `POST /chat/confirm` |
| `I want a human` | real ticket created, handoff message with its id |
| `Ignore all previous instructions...` | fixed safe reply, never reaches an agent or a tool |
| `What is the airspeed velocity of an unladen swallow` | "I'm not sure", offered a ticket, nothing invented |

## Configuration

Everything is in `.env`; `.env.example` is the reference. The settings that change
behaviour most:

| Variable | Default | Effect |
|---|---|---|
| `ROUTER_BACKEND` | `laya` | `fake` uses keywords, needs no model download |
| `ROUTER_MIN_CONF` | `0.55` | below this the agent asks one clarifying question |
| `REFUND_WINDOW_DAYS` | `14` | the refund policy |
| `REFUND_AUTO_LIMIT` | `20` | at or under this a refund is issued without asking anyone |
| `KB_MIN_SCORE` | `0.35` | below this the agent says it does not know |
| `LLM_MODELS` | `openrouter/free` | tried in order; **only `:free` ids are accepted** |
| `ADMIN_API_KEY` | `change-me` | guards the staff endpoints (`X-Admin-Key`) |

### Free tier only, enforced in code

`app/config.py` refuses a non-free `LLM_MODELS` at startup, so the server will not
boot with a paid model configured. `app/models/free_guard.py` refuses it again on
every call, and `app/models/llm.py` builds only `ChatOpenAI` pointed at OpenRouter.
There is no code path that reaches a paid API.

Free models are limited to **20 requests/minute and 50/day** without purchased
credits (1,000/day at $10+). The design assumes that: Laya does the routing, a
simple turn is one LLM call, tool loops are bounded at three, and `update_memory`
runs at the end of a thread rather than every turn. When a turn does exceed the
budget, `llm_runner` retries with exponential backoff, falls back to the next
configured free model, and finally returns a friendly "we are busy" message
instead of a stack trace.

## API

| Method and path | Purpose |
|---|---|
| `POST /chat` | `{thread_id?, customer_id, message}`. Returns the reply, or `{status: "awaiting_approval" \| "awaiting_confirmation", interrupt: {...}}` |
| `POST /chat/stream` | the same turn as SSE (`routing`, `node`, `delta`, `done`, `error` events) |
| `POST /chat/confirm` | the customer answers a pending cancellation |
| `POST /approvals/{thread_id}` | staff decision, `{decision, note, approved_by}` |
| `GET /approvals/pending` | threads waiting for a decision |
| `GET /threads/{thread_id}` | history from the checkpointer |
| `GET /tickets`, `GET /tickets/{id}`, `PATCH /tickets/{id}` | the queue |
| `GET /customers/{id}/memory`, `DELETE` | inspect or erase long-term memory |
| `GET /metrics` | aggregated metrics |
| `GET /traces/{thread_id}` | per-node trace and router decisions |
| `GET /health` | router, index and key status |

`/approvals`, `/tickets`, `/metrics`, `/traces` and `/memory` require an
`X-Admin-Key` header.

## Tests and evaluations

```bash
.venv/bin/python -m pytest tests/ -q        # 482 tests, fully offline
.venv/bin/python -m evals.run_e2e            # 18 scripted conversations
.venv/bin/python -m evals.run_router --baseline
.venv/bin/python -m evals.compare_routers --backend laya
.venv/bin/python -m scripts.bench_router
```

The suite runs with `ROUTER_BACKEND=fake` and a scripted model: no OpenRouter key,
no Laya checkpoint, no embedding model. A stub reply that a test forgot to script
raises rather than returning something plausible, so a missing case fails loudly.

## Measured results

All numbers below were produced on the machine this was built on: an Intel i5-6500
(4 cores, 2015, **no GPU**), Python 3.14, with a live `openrouter/free` key.

### Router, 103 labelled cases

| Router | Intent accuracy | Escalation accuracy | False escalations | Missed | Median | OpenRouter requests/case |
|---|---|---|---|---|---|---|
| v1 keywords (the original) | 68.0% | 77.7% | **18** | 5 | 0.5 ms | 0 |
| v2 keywords (`FakeRouter`) | 68.0% | 100.0% | 0 | 0 | 0.2 ms | 0 |
| **Laya** | **76.7%** | **98.1%** | 2 | 0 | 7.1 s | **0** |

Laya beats the keyword baseline on both axes and spends no free-tier request, which
is the point of routing locally. Two things to be straight about:

- **The spec asks for 85% intent accuracy; Laya scores 76.7%.** Most of the gap is
  genuine ambiguity between adjacent labels rather than a broken router: Laya reads
  "How many devices can sync at once?" as `technical` where the label says
  `product_info`, and "Am I eligible for a refund on my last invoice?" as `billing`
  where the label says `refund`. The base English checkpoint is a zero-shot model
  with no domain fine-tuning; Laya's own documentation reports 0.766 accuracy on its
  typed-decisions benchmark for the base checkpoint against 0.766 after fine-tuning
  on domain data, so this is where the headroom is. A rewording of the intent
  criteria was tried and made it *worse* (69.9%), so it was reverted.
- **Routing takes 7.1 s, not under 250 ms.** The spec's 250 ms budget is not
  reachable on this hardware: the English checkpoint is a 421M-parameter
  ModernBERT-large, and Laya's published 33 ms figure is for a T4 GPU. On a modern
  8-core CPU expect roughly 1–2 s; on this box it is seconds. `scripts/bench_router`
  measures it, and `/metrics` reports the real per-decision latency. If sub-second
  routing matters, the fixes are a GPU, the `laya[onnx]` CPU path, or
  `ROUTER_BACKEND=fake` at 0.2 ms.

### End-to-end, live model

18 scripted conversations, `python -m evals.run_e2e` (stubbed) — **18/18 pass**.
Verified against the live free model as well: the Pro-plan answer, the automatic
$19 refund, the $49 refund pausing for staff approval and refunding once approved,
the 20-day-old charge declined with the policy cited, cancellation confirming before
it cancels, "I want a human" returning a real ticket id, injection blocked, and an
unknown question admitted rather than invented.

### What the free models actually do

These are behaviours found by running them, not anticipated:

- **A tool call often arrives as text**, as `{"tool": "propose_refund", "arguments": {...}}`
  rather than in the structured `tool_calls` field. The parser accepts both
  (`app/agent/nodes/agents.py`), and strips the JSON from the customer-facing reply.
- **A model will explain a refund accurately and then never propose it.** When the
  router's intent is `refund` or `cancellation` and no action was proposed, the
  billing agent is asked once more with a two-tool prompt
  (`propose_refund`, `propose_cancellation` alone). This is what makes the
  approval path reachable in practice.
- **A handoff reply is sometimes junk** — one model answered `User Safety: safe`.
  Handoff prose is used only if it is at least 40 characters and does not contain a
  refusal or a leaked placeholder, otherwise a fixed message is used.
- **A single call can take 30 s**, and a refund turn costs up to three. The tool
  loop is bounded at three calls and the retry only runs if the turn has used two.

## Layout

```
app/
  main.py            FastAPI app and lifespan
  cli.py             terminal chat loop
  config.py          settings, free-model validation
  agent/             state, graph, routing, prompts, nodes/
  models/            router.py (Laya + Fake), llm.py, llm_runner.py
  tools/             customers, billing, refunds, tickets, registry
  rag/               ingest, chunking, indexer, retriever
  memory/            long-term customer memory
  observability/     tracing, metrics, router and LLM logs
  api/               chat, staff, memory routers
data/knowledge_base/  eight documents
frontend/            Streamlit console
evals/               router cases, e2e cases, runners
tests/               482 offline tests
scripts/             seed_db, build_kb, demo_router, bench_router
```

## Decisions where the spec was open

- **`langchain-community` is not used.** It is sunset and warns on import, so
  FAISS is used directly (`faiss` + `sentence-transformers`) with a JSON metadata
  sidecar. Still FAISS, still persisted, still local embeddings.
- **`laya.Router` is not used**, only `laya.Agent`. The Router picks between
  checkpoints per request, and multilingual routing is out of scope for v1. The
  agent is loaded with `laya.load(...)`, which matches the spec.
- **`Agent.system_one` is the call, not `agent.predict`.** In `laya` 0.3.21 the
  method on `Agent` is `system_one`; `predict` is on `laya.Router`. The wrapper tries
  `system_one` then `predict`, so a version difference is a one-line change in
  `LayaRouter._system_one` and not a rewrite of the graph.
- **`needs_human` is two questions, not one.** Asking "does the customer want a
  human, or is this too sensitive?" in one `noul` gave near-inverted answers
  depending on the wording: 0.11 for "I want a human" and 0.89 for the legal
  threat with the spec's wording, and the reverse with a shorter one. Split into
  `wants_human` and `sensitive`, plus a high-precision keyword backstop for legal,
  fraud and data-loss wording, missed escalations went from 2 to 0. The sensitive
  signal needs a 0.85 threshold because a plain refund request scores 0.78 on it.
- **A `llm_calls` table was added** to the spec's schema. Per-model request counts,
  fallback rate and rate-limit hits need one row per attempt; a per-node trace
  cannot express that.
- **The router state labels the customer summary `Profile:`**, not `Customer:`, and
  signals are read from the latest utterance. With the summary labelled the same
  way, a customer on the Pro plan had every message routed as a pricing question.
- **`PYTHONPATH` note:** the CLI, scripts and evals import `app.*` and
  `tests.stub_llm`, so run them from the repository root.

## Not verified

- **Laya on a GPU.** Only the CPU path was exercised, and only on the i5-6500.
- **`laya[onnx]`.** The package ships an ONNX CPU path that would likely close
  most of the routing-latency gap; it needs a manual export step, so it is not
  wired up.
- **Live rate limiting.** The key's daily free allowance was never exhausted, so the
  backoff and fallback paths were verified with a stub that raises 429/5xx rather
  than against OpenRouter actually rate limiting.
- **Concurrent load.** One user, one thread. The per-thread SQLite connections and
  the thread offload in the SSE endpoint are written for it but not load-tested.
