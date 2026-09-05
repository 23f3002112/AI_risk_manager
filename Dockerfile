FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY data/ /app/data/
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Expose port (default 8000 for FastAPI)
ENV PORT=8000
EXPOSE $PORT

# Command to run FastAPI server
CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port $PORT"]
