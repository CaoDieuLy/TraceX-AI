# Metadata Service Dockerfile for Docker/Coolify deployment

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies with retry
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/services/metadata-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Increment to force Docker to skip its layer cache for app code below.
ARG CACHEBUST=2

# Copy application code
COPY backend/services/metadata-service/app ./app

# Copy backend config (homography_registry.json, etc.)
COPY backend/config ./backend/config

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=5).read()"

# Run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
