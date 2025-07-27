# Use Python 3.9 slim image (standard and reliable)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python scripts
COPY train.py predict.py ./

# Copy models directory (will be created during CI)
COPY models/ ./models/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "predict.py"]