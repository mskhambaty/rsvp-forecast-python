FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create practical models (Random Forest + Linear Regression)
RUN python create_practical_model.py

# Expose the port (Render will override this with its own PORT env var)
EXPOSE 8000

# Start the FastAPI application with Uvicorn
# Using PORT env var that Render provides
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
