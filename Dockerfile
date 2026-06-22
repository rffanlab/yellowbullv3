FROM python:3.12-slim AS base

WORKDIR /app

# System deps for building some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser || true
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
