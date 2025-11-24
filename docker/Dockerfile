# H-ΛCDM Analysis Framework - Docker Image
# ===========================================
#
# This Dockerfile creates a reproducible environment for running
# the Holographic Lambda Model (H-ΛCDM) analysis framework.
#
# Usage:
#   docker build -f docker/Dockerfile -t hlcdm-analysis .
#   docker run -v $(pwd)/downloaded_data:/app/downloaded_data \
#              -v $(pwd)/processed_data:/app/processed_data \
#              -v $(pwd)/results:/app/results \
#              hlcdm-analysis --gamma validate

FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p downloaded_data processed_data results

# Set permissions
RUN chmod +x main.py

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash hlcdm && \
    chown -R hlcdm:hlcdm /app
USER hlcdm

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import numpy; print('H-ΛCDM environment ready')" || exit 1

# Default command
CMD ["python", "main.py", "--help"]
