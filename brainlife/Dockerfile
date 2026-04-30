# Use JAX-ready CUDA image
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Install python and system deps
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip3 install uv

WORKDIR /app

# Copy project files
COPY . /app

# Install project dependencies
# Note: we install the package in editable mode or just install deps
RUN uv pip install --system ".[cuda12]"

# Brainlife expected entry point
ENTRYPOINT ["/app/brainlife/main.py"]
