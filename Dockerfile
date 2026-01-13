FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download models
RUN python3 scripts/download_models.py || echo "Model download script not found"

# Expose port for API
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/models
ENV CUDA_VISIBLE_DEVICES=0

# Create volumes for data
VOLUME ["/app/data", "/app/models"]

# Default command
CMD ["python3", "main.py", "--input", "0", "--model", "yolov8n"]
