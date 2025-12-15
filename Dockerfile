# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# MongoDB configuration (override with -e or docker-compose)
ENV MONGODB_URL=mongodb://localhost:27017
ENV DATABASE_NAME=traffic_monitoring
ENV SECRET_KEY=your-super-secret-key-change-in-production-2024

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY backend_complete.py .
COPY yolov8n.pt .

# Copy frontend files
COPY frontend/ ./frontend/

# Create necessary directories
RUN mkdir -p data/violations data/logs data/uploads data/output

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "backend_complete:app", "--host", "0.0.0.0", "--port", "8000"]
