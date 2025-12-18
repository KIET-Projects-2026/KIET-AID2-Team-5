# Use Python 3.10 slim image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create required directories
RUN mkdir -p data/uploads data/output data/violations data/logs

# Set pip timeout
ENV PIP_DEFAULT_TIMEOUT=300

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend_complete.py .
COPY .env .

# Copy frontend files
COPY frontend/ ./frontend/

# Copy YOLOv8 model (if exists locally)
# If yolov8n.pt is not present, the build will fail; otherwise, it will be copied
COPY yolov8n.pt .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "backend_complete:app", "--host", "0.0.0.0", "--port", "8000"]