# Development Dockerfile for Bioinformatics Toolkit
FROM docker.io/jupyter/datascience-notebook:python-3.10

USER root

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV R_LIBS_SITE=/usr/local/lib/R/site-library
ENV R_LIBS_USER=/opt/conda/lib/R/library

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libhdf5-dev \
    libigraph-dev \
    libboost-all-dev \
    libgsl-dev \
    gnupg \
    lsb-release \
    python3-dev \
    awscli \
    git \
    git-lfs \
    libfftw3-dev \
    libudunits2-dev \
    libgeos-dev \
    libgdal-dev \
    libproj-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Mamba for faster package management
RUN conda install -y -c conda-forge mamba

# Copy environment.yml file
COPY environment.yml /tmp/environment.yml

# Create conda environment from environment.yml using Mamba
RUN mamba env update -n base -f /tmp/environment.yml && \
    mamba clean -afy && \
    rm /tmp/environment.yml

# Install problematic packages separately with better error handling
# First install core dependencies
RUN pip install --no-cache-dir numpy h5py

# Install packages one by one with error handling
# RUN pip install --no-cache-dir loompy==3.0.6 || echo "loompy installation failed, continuing..."
# RUN pip install --no-cache-dir scanpy-scripts || echo "scanpy-scripts installation failed, continuing..."
# RUN pip install --no-cache-dir squidpy || echo "squidpy installation failed, continuing..."
# RUN pip install --no-cache-dir spatialdata-io || echo "spatialdata-io installation failed, continuing..."
# RUN pip install --no-cache-dir --no-deps "napari<0.5.0" || echo "napari installation failed, continuing..."

# Copy R requirements file
COPY r-requirements.R /tmp/r-requirements.R

# Install R packages from requirements file
RUN Rscript /tmp/r-requirements.R && \
    rm /tmp/r-requirements.R

# Set up the project structure
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/results /tmp/data

# Use the existing jovyan user (UID 1000) that comes with the Jupyter image
# instead of creating a new user
RUN chown -R jovyan:users /app
USER jovyan

# Set working directory for mounted volumes
WORKDIR /data

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]