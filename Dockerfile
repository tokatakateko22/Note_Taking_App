FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for SQLite FTS5 (usually built-in, but just in case)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment vars
ENV PYTHONPATH=/app
ENV DB_PATH=/data/notes.db

# Ensure data directory exists
RUN mkdir -p /data

# Default command: run the CLI REPL
CMD ["python", "-m", "src.cli"]
