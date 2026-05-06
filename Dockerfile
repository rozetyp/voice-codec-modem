FROM python:3.12-slim

# ffmpeg + libopus + libopencore-amrnb (encoder/decoder for our channels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source. .dockerignore narrows what gets copied.
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOST=0.0.0.0

# Railway sets $PORT; default to 8080 locally.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
