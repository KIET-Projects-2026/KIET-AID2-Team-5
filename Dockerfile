# Use Python 3.10 slim image
FROM python:3.10-slim

# Install system dependencies required for OpenCV and video processing
# Also install Node.js for yt-dlp JavaScript runtime (required for YouTube)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    curl \
    ca-certificates \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create required directories for data storage
RUN mkdir -p data/uploads data/output data/violations data/logs

# Set pip configuration for faster installs
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_NO_CACHE_DIR=1

# Copy and install Python dependencies first (for better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy ALL application source files (add missing files)
COPY backend_complete.py .

# Copy frontend directory
COPY frontend/ ./frontend/

# YOLOv8 model will be auto-downloaded by ultralytics on first run
# Don't copy it - saves space and build time

# Set only essential environment variable
# ENVIRONMENT will be set by Render's environment variables, not hardcoded
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application with uvicorn
# Increased timeout for large file uploads and YouTube extraction
CMD ["uvicorn", "backend_complete:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "300", "--timeout-graceful-shutdown", "30"]