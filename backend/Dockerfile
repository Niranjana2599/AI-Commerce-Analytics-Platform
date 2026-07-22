# Use a small, supported Python base image for the FastAPI service.
FROM python:3.13-slim

# Prevent Python bytecode files and ensure logs appear immediately in Docker logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Run all following commands from the application root.
WORKDIR /app

# Copy dependencies first so Docker can reuse this layer when application code changes.
COPY backend/requirements.txt /tmp/requirements.txt

# Install only the Python packages needed to run the API.
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy backend code after dependencies to preserve Docker layer caching.
COPY backend /app/backend
COPY src /app/src
COPY pipelines /app/pipelines
COPY scripts /app/scripts

# Make the Render startup script executable inside the Linux image.
RUN chmod +x /app/scripts/start_render_backend.sh

# Document the internal FastAPI port.
EXPOSE 8000

# Ask Docker whether the API health endpoint is responding.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD sh -c "python -c \"from urllib.request import urlopen; import os; urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/api/v1/health', timeout=3)\"" || exit 1

# Start one Uvicorn worker; Compose supplies data/model volumes and environment values.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
