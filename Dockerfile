FROM ghcr.io/astral-sh/uv:0.5.20 AS uv_bin
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install uv binary from official image
COPY --from=uv_bin /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY pyproject.toml README.md ./

# Install python dependencies with uv
RUN uv pip install --system -e ".[dev]"

# Copy application source code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY docs/ ./docs/

EXPOSE 8000 9090

# Default entrypoint runs FastAPI application with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
