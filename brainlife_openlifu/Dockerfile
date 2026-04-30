# Use CUDA image compatible with JAX
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install uv

WORKDIR /app

# Copy the local openlifu-python source
COPY . /app

# Install openlifu-python and its dependencies (including the j-Wave fork)
RUN uv pip install --system ".[cuda12]"

# Make main executable
RUN chmod +x /app/brainlife_openlifu/main.py

ENTRYPOINT ["/app/brainlife_openlifu/main.py"]
