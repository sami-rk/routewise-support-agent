#!/usr/bin/env bash
# First-run setup for the api container, then serve.
#
# The database and the knowledge base index are not in the image: one is data
# and the other is a generated artefact, and both are supposed to live in the
# volume. So on the first start they are created here, and a marker in the volume
# means later starts skip straight to the server.
#
#   docker compose up --build      first run, waits for the index
#   docker compose up              later runs, starts immediately
#   docker compose down -v         throw the volume away and start over

set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
# Where the application is installed. /app in the image, overridable so the setup
# half can be exercised outside a container.
APP_DIR="${APP_DIR:-/app}"
DB_PATH="${DB_PATH:-${DATA_DIR}/support.db}"
CHECKPOINT_DB_PATH="${CHECKPOINT_DB_PATH:-${DATA_DIR}/checkpoints.db}"
KB_DIR="${KB_DIR:-${DATA_DIR}/knowledge_base}"
MARKER="${DATA_DIR}/.initialised"

log() { printf '[entrypoint] %s\n' "$*" >&2; }

# The knowledge base documents are code, not data, so they are copied into the
# volume the first time. Without them the retriever has nothing to search and
# every answer is an honest "I am not sure", which looks like a broken install
# rather than an empty one.
install_documents() {
    mkdir -p "${KB_DIR}"
    if [ -z "$(ls -A "${KB_DIR}" 2>/dev/null)" ]; then
        log "copying the knowledge base into ${KB_DIR}"
        cp "${APP_DIR}/data/knowledge_base/"*.md "${KB_DIR}/"
    fi
}

initialise() {
    if [ -f "${MARKER}" ]; then
        log "already initialised, skipping setup"
        return
    fi

    log "first start: preparing the database and the knowledge base"
    install_documents

    if [ ! -f "${KB_DIR}/../kb_index/index.faiss" ]; then
        # The embedding model is local, so this needs no API key, but it does
        # download ~130 MB on the first run. That is the wait.
        log "building the FAISS index (downloads the embedding model once)"
        python -m scripts.build_kb --kb-dir "${KB_DIR}" --index-dir "${KB_DIR}/../kb_index"
    else
        log "knowledge base index already present"
    fi

    log "seeding the demo customers"
    python -m scripts.seed_db --db-path "${DB_PATH}"

    touch "${MARKER}"
    log "setup finished"
}

main() {
    initialise
    log "serving on 0.0.0.0:${PORT:-8000}"
    exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
}

main "$@"
