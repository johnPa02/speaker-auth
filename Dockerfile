# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501

# System dependencies for audio + builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libsndfile1 \
    libasound2 \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv package manager
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv on PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copy project metadata first for better layer caching
COPY pyproject.toml uv.lock ./

# Sync project dependencies into a local venv and add missing runtime deps
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --no-dev

# Copy source last
COPY web_app.py core.py ws_server.py ./
COPY README.md ./

# Ensure data/model dirs exist and are writable
RUN mkdir -p /app/enrolled /app/pretrained_models

EXPOSE 8501 8000

# Use uv to run inside the managed venv
ENV PATH="/app/.venv/bin:${PATH}"
# Default to Streamlit. Run API with:
#   uv run uvicorn ws_server:app --host 0.0.0.0 --port 8000
CMD ["uv", "run", "streamlit", "run", "web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
