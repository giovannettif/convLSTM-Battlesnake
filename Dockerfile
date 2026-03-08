# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/apt-get/lists/*

# Copy requirements and install python dependencies
# (Note: We use a simplified list here based on what we've been using)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    numpy \
    pydantic \
    requests

# Copy the project files
COPY . .

# Ensure the Battlesnake CLI is executable
RUN chmod +x ./battlesnake

# Set environment variables for the cluster
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# The entrypoint will usually be overridden by the SLURM script
# but we can set a default for testing
CMD ["python3", "my_nn_snake/core/train.py", "--total-steps", "20000000"]
