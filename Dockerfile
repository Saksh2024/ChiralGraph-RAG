# =====================================================================
# ChiralGraph-RAG Production Multi-Stage Containerfile
# 3D Conformation & Stereoisomer-Aware Drug Discovery Engine
# =====================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_PORT=8000

WORKDIR /app

# Install system dependencies required for RDKit, scientific libraries, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libxrender1 \
    libxext6 \
    libsm6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a separate layer for optimal caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and web dashboard assets
COPY src/ /app/src/
COPY ui/ /app/ui/
COPY tests/ /app/tests/
COPY pytest.ini /app/pytest.ini
COPY .env.example /app/.env

# Create non-root application user for container security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose FastAPI application port
EXPOSE 8000

# Container Healthcheck verifying FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production Uvicorn ASGI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
