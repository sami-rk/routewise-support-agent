# The API. CPU-only torch, because the image is otherwise 3 GB of CUDA wheels
# for a service that only needs a router and a sentence encoder on the CPU.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && pip install -r requirements.txt

COPY . .

# The knowledge base index is built at start-up rather than baked in, so the
# image does not carry a generated artefact. Override KB_DIR to mount your own.
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
