# The API and the agent: LangGraph, Laya, local embeddings, SQLite.
#
# CPU-only torch, because this image would otherwise be ~3 GB of CUDA wheels on
# a machine that mostly has no GPU. Set ROUTER_BACKEND=laya to use the local
# decision model; the default is the keyword router, which needs no checkpoint
# download and no 1.7 GB of weights.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    APP_DIR=/app \
    DATA_DIR=/data \
    KB_DIR=/data/knowledge_base \
    DB_PATH=/data/support.db \
    CHECKPOINT_DB_PATH=/data/checkpoints.db \
    PORT=8000

WORKDIR /app

# build-essential is needed to install faiss-cpu's dependency chain, then dropped
# from the final layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so a code change does not reinstall them.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/knowledge_base/ ./data/knowledge_base/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# The volume is mounted here; the entrypoint fills it on the first start.
VOLUME ["/data"]
EXPOSE 8000

# Defined once, here, so plain `docker run` is checked as well as compose. The
# start period is generous because the first start builds the FAISS index, which
# downloads the embedding model.
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=5 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code == 200 else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
