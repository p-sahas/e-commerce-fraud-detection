# =============================================================================
# Dockerfile — E-Commerce Fraud Detection Streaming Services
# =============================================================================
# Used by: producer, inference, analytics containers in docker-compose.yml
# Build: docker build -t fraud-detection .
# =============================================================================

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        kafka-python>=2.0.2 \
        mlflow>=2.12.0 \
        pandas>=2.0.0 \
        numpy>=1.26.0 \
        scikit-learn>=1.4.0 \
        lightgbm>=4.3.0 \
        pyyaml>=6.0 \
        python-dotenv>=1.0.0

# Copy application source
COPY pipelines/  pipelines/
COPY src/        src/
COPY utils/      utils/
COPY config/     config/

# Create runtime directories
RUN mkdir -p data/raw artifacts/models artifacts/predictions analytics

# Default: run in demo mode (overridden by docker-compose command:)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python"]
CMD ["pipelines/streaming_inference_pipeline.py", "demo"]
