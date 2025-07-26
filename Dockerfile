# Use Python 3.11 slim image (more likely to be available)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy models directory (will be created during CI)
COPY models/ ./models/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "src/predict.py"]