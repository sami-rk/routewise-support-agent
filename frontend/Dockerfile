# The Streamlit console. It talks to the API over HTTP only, so the two can be
# scaled separately and the image needs neither torch nor the model cache.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    API_BASE_URL=http://api:8000

WORKDIR /app

RUN pip install streamlit==1.64.0 httpx

COPY frontend/ ./frontend/

EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit_app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0"]
