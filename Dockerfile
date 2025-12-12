FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set pip timeout and index to be more tolerant on slow networks
ENV PIP_DEFAULT_TIMEOUT=300

# Copy and install lighter runtime deps first
COPY requirements.runtime.txt .
RUN pip install --no-cache-dir -r requirements.runtime.txt

# Install AI deps separately (so if this fails, you can retry only this layer)
COPY requirements.ai.txt .
RUN pip install --no-cache-dir -r requirements.ai.txt

# Copy the whole project
COPY . .

EXPOSE 8000
CMD ["uvicorn", "backend_complete:app", "--host", "0.0.0.0", "--port", "8000"]
