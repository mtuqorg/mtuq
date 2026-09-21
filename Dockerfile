# Development Dockerfile for mtuq
FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies including OpenMPI for parallel execution
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libnetcdf-dev \
    libhdf5-dev \
    pkg-config \
    git \
    vim \
    curl \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none

# Copy the full source to install dependencies from pyproject.toml
COPY . /workspace

# Install Python dependencies from pyproject.toml (single source of truth)
# Use [dev] extras to include development dependencies like mpi4py
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[dev]"

# Create a non-root user for development
RUN useradd -m -s /bin/bash developer && \
    chown -R developer:developer /workspace

USER developer

# Default command
CMD ["/bin/bash"]

