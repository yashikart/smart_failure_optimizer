# Multi-stage build for optimized production image

# ========================================
# Stage 1: Builder
# ========================================
FROM python:3.9-slim as builder

LABEL maintainer="hydraulic-monitor@example.com"
LABEL description="Hydraulic System Condition Monitoring - MLOps Application"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# ========================================
# Stage 2: Production
# ========================================
FROM python:3.9-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application files
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser new_app.py .
COPY --chown=appuser:appuser requirements.txt .

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R appuser:appuser /app/models /app/data /app/logs

# Copy models if they exist (optional)
COPY --chown=appuser:appuser models/ ./models/ 2>/dev/null || true

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run Streamlit (select entry via APP_FILE env; defaults to app.py)
ENV APP_FILE=app.py
CMD ["sh", "-c", "streamlit run ${APP_FILE} --server.port=8501 --server.address=0.0.0.0"]

