#!/usr/bin/env bash
# Start the stack again from an empty volume.
#
#   ./scripts/dev_reset.sh            stop, delete the volume, start again
#   ./scripts/dev_reset.sh --no-build stop, delete the volume, start without rebuilding
#
# Deletes the database, the FAISS index and the checkpoints, so the next start
# re-seeds the demo customers and rebuilds the index. The source code and the
# image are left alone.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed or not on PATH" >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "docker compose (v2) is required: https://docs.docker.com/compose/install/" >&2
    exit 1
fi

build_flag="--build"
if [ "${1:-}" = "--no-build" ]; then
    build_flag=""
fi

echo "==> stopping the stack and removing the volume"
docker compose down -v

echo "==> starting again (the first start re-seeds and rebuilds the index)"
# shellcheck disable=SC2086  # build_flag is deliberately empty or --build
docker compose up ${build_flag}
