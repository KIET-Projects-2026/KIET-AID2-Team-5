# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create required directories
RUN mkdir -p data/uploads data/output data/violations data/logs

# Increase pip timeout for slow networks
ENV PIP_DEFAULT_TIMEOUT=300

# Copy and install dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Copy frontend files
COPY frontend/ ./frontend/

# Create necessary directories
RUN mkdir -p data/violations data/logs data/uploads data/output

# Expose port
EXPOSE 8000

CMD ["uvicorn", "backend_complete:app", "--host", "0.0.0.0", "--port", "8000"]
