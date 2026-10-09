# ==============================================================================
# WeatherCast AI - Production Multi-Stage Dockerfile
# Optimized for AWS ECS Fargate / Container Run
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Wheel Compiler Stage
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Prevent writing .pyc files & unbuffer standard output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install minimal OS build dependencies needed for building wheels (C/C++ extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY backend/requirements.txt /build/requirements.txt

# Create wheels in isolated directory (including all transitive dependencies)
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r /build/requirements.txt


# ------------------------------------------------------------------------------
# Stage 2: Final Lean Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runner

# Configure runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DJANGO_SETTINGS_MODULE=config.settings.production \
    PYTHONPATH="/app:/app/backend"

WORKDIR /app

# Install minimal runtime shared libraries (curl for health check and libgomp for XGBoost OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install wheels from builder stage
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt && \
    rm -rf /wheels

# Create dedicated non-root security user & group
RUN groupadd -r appgroup && useradd -r -u 1001 -g appgroup appuser

# Copy application directories with proper ownership
COPY --chown=appuser:appgroup backend /app/backend
COPY --chown=appuser:appgroup frontend /app/frontend
COPY --chown=appuser:appgroup ml/models /app/ml/models

# Prepare working directories & staticfiles with non-root ownership
RUN mkdir -p /app/backend/staticfiles && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose production port
EXPOSE 8000

# Container Healthcheck (Checks internal /api/health/ endpoint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/api/health/ || exit 1

# Production Gunicorn WSGI Server entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "2", "--timeout", "30", "--access-logfile", "-", "--error-logfile", "-", "--chdir", "/app/backend", "config.wsgi:application"]
