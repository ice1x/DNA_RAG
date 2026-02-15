# ---------------------------------------------------------------------------
# DNA RAG — Multi-stage Docker build
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project metadata first for layer caching
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install the package with API extras
RUN pip install --no-cache-dir ".[api]"

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "dna_rag.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]

# ---------------------------------------------------------------------------
# CLI (for batch jobs / cron)
# ---------------------------------------------------------------------------
FROM base AS cli

ENTRYPOINT ["dna-rag"]
