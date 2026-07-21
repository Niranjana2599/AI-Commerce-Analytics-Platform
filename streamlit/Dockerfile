# Use a small, supported Python base image for the Streamlit service.
FROM python:3.11-slim

# Prevent bytecode files and flush Streamlit logs directly to Docker logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Run all following commands from the frontend directory.
WORKDIR /app/streamlit

# Copy dependencies first so Docker reuses this layer when UI code changes.
COPY streamlit/requirements.txt /tmp/requirements.txt

# Install only the frontend packages.
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the Streamlit application after the dependency layer.
COPY streamlit /app/streamlit

# Document the internal Streamlit port.
EXPOSE 8501

# Check that Streamlit is serving its built-in health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

# Run Streamlit on every container interface; Compose injects the backend service URL.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
