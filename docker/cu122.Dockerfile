# Base image with CUDA 12.2.2 on Ubuntu 22.04
FROM nvcr.io/nvidia/cuda:12.2.2-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip and install base Python tools
RUN python -m pip install --upgrade pip setuptools wheel

# install AWS
RUN pip install awscli boto3

# Install PyTorch (CUDA 12.2)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone the repository (you can replace the URL below)
ARG REPO_URL=https://github.com/Motor-Ai/pin_slam.git
RUN git clone ${REPO_URL} /pin_slam

# Set working directory
WORKDIR /pin_slam

# Install Python dependencies
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Default command
CMD ["python"]