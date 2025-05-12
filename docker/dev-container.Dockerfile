# Stage 1: Dependency solver and package downloader
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 as solver

# Set environment variables for solver stage
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/conda/bin:$PATH

# Install minimal system dependencies needed for conda
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
# Install Miniconda/Mamba in solver stage
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/25.3.0-1/Miniforge3-25.3.0-1-Linux-x86_64.sh -O /tmp/mambaforge.sh \
    && bash /tmp/mambaforge.sh -b -p /opt/conda \
    && rm /tmp/mambaforge.sh \
    && chmod -R 777 /opt/conda

# Copy environment files to solver
COPY environment-py.yml /tmp/environment-py.yml
COPY environment-r.yml /tmp/environment-r.yml

# PRE-SOLVE dependencies and download packages without installing
RUN mamba create --name sctools-py --dry-run -f /tmp/environment-py.yml && \
    mamba create --name sctools-r --dry-run -f /tmp/environment-r.yml

# Stage 2: Final build with pre-downloaded packages
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/conda/bin:$PATH

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libhdf5-dev \
    libigraph-dev \
    libboost-dev \
    libgsl-dev \
    git \
    git-lfs \
    libfftw3-dev \
    libudunits2-dev \
    libgeos-dev \
    libgdal-dev \
    libproj-dev \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
# Install Miniconda/Mamba
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/25.3.0-1/Miniforge3-25.3.0-1-Linux-x86_64.sh -O /tmp/mambaforge.sh \
    && bash /tmp/mambaforge.sh -b -p /opt/conda \
    && rm /tmp/mambaforge.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo '. /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc \
    && echo 'conda activate base' >> ~/.bashrc \
    && chmod -R 777 /opt/conda

# Copy the pre-downloaded packages from solver stage
COPY --from=solver /opt/conda/pkgs /opt/conda/pkgs

# Set up basic CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Copy environment files
COPY environment-py.yml /tmp/environment-py.yml
COPY environment-r.yml /tmp/environment-r.yml

# Create sctools-py conda environment - should be faster now with pre-downloaded packages
RUN mamba env create -f /tmp/environment-py.yml && \
    conda clean -a

# Create sctools-r conda environment - should be faster now with pre-downloaded packages
RUN mamba env create -f /tmp/environment-r.yml && \
    conda clean -a && \
    rm /tmp/environment-py.yml /tmp/environment-r.yml && \
    conda run -n sctools-r R -e "IRkernel::installspec(name = 'sctools-r', displayname = 'R (sctools-r)')"

# Install NVIDIA Container Toolkit essentials
RUN conda run -n sctools-py pip install --no-cache-dir cupy-cuda12x

# Make Python environment available as Jupyter kernel
RUN conda run -n sctools-py python -m ipykernel install --user --name sctools-py --display-name "Python (sctools-py)"

# Create a user with the same UID as the host user to avoid permission issues
RUN useradd -m -s /bin/bash -N -u 1000 developer

# Set up the project structure
RUN mkdir -p /app/data /app/results /tmp/data /data
RUN chown -R developer:developer /app /tmp/data /data /opt/conda
# Fix conda permissions for nb_conda_kernels
RUN chmod -R 755 /opt/conda/bin/conda

# Switch to the developer user
USER developer
WORKDIR /data

# Create a startup script to activate the sctools-py environment and start Jupyter
RUN echo '#!/bin/bash\n\
eval "$(conda shell.bash hook)"\n\
# Set CUDA environment variables\n\
export CUDA_HOME=/usr/local/cuda\n\
export CUDA_PATH=/usr/local/cuda\n\
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH\n\
\n\
# Activate the conda environment\n\
conda activate sctools-py\n\
\n\
# Verify CUDA availability\n\
echo "CUDA Environment Information:"\n\
python -c "import torch; print(\'CUDA Available: \', torch.cuda.is_available()); print(\'CUDA Devices: \', torch.cuda.device_count()); print(\'CUDA Device Name: \', torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'None\')" || echo "PyTorch CUDA check failed"\n\
\n\
# Install local packages in development mode if setup.py exists\n\
if [ -f "/data/repo/setup.py" ]; then\n\
    echo "Installing sctools packages in development mode..."\n\
    pip install -e /data/repo\n\
fi\n\
\n\
# Start Jupyter Lab\n\
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token="" --NotebookApp.password=""' > /home/developer/start-jupyter.sh && \
    chmod +x /home/developer/start-jupyter.sh

# Default command to start Jupyter Lab in the sctools-py environment
CMD ["/home/developer/start-jupyter.sh"]