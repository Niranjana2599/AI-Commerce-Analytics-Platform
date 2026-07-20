# Run from the repository root: docker build -f backend/Dockerfile -t ai-commerce-api .
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY src /app/src
COPY models /app/models
COPY data/processed/master_df.parquet /app/data/processed/master_df.parquet

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
