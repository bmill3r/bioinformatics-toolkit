# Dockerfile for Bioinformatics Toolkit
# This Dockerfile sets up an environment for single-cell and spatial transcriptomics analysis

# Start with a base Python and R image
FROM jupyter/datascience-notebook:python-3.10

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
    libfftw3-dev \
    libudunits2-dev \
    libgeos-dev \
    libgdal-dev \
    libproj-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    scanpy \
    anndata \
    loompy \
    leiden \
    umap-learn \
    pynndescent \
    scikit-learn \
    statsmodels \
    numba \
    pandas \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    plotly \
    bokeh \
    datashader \
    colorcet \
    s3fs \
    boto3 \
    h5py \
    zarr \
    fastremap \
    ipywidgets \
    scikit-image \
    igraph \
    louvain \
    squidpy \
    tables \
    pyarrow \
    tqdm \
    jupyter \
    jupyterlab \
    rpy2 \
    pygam \
    pysal

# Install R packages
RUN R -e "install.packages(c('BiocManager', 'devtools', 'remotes'), repos='http://cran.rstudio.com/')" && \
    R -e "BiocManager::install(c('SingleCellExperiment', 'scran', 'scater', 'BiocParallel', 'DropletUtils', 'sctransform', 'limma', 'edgeR'))" && \
    R -e "BiocManager::install('monocle3')" && \
    R -e "install.packages(c('Seurat', 'SeuratObject', 'SeuratDisk', 'leiden', 'ggplot2', 'dplyr', 'tidyr', 'patchwork', 'spatstat', 'Matrix', 'MASS', 'data.table', 'cowplot', 'ggrepel', 'reticulate', 'hdf5r', 'Rtsne'), repos='http://cran.rstudio.com/')" && \
    R -e "BiocManager::install('AUCell')" && \
    R -e "BiocManager::install('MAST')" && \
    R -e "devtools::install_github('cole-trapnell-lab/leidenbase')" && \
    R -e "devtools::install_github('mojaveazure/seurat-disk')"

# Install optional packages for spatial transcriptomics
RUN pip install --no-cache-dir \
    squidpy \
    spatialdata-io \
    spatialde \
    scvi-tools \
    napari[all] \
    vitessce

# Set up the project structure
WORKDIR /app

# Copy the project files
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/data /app/results /tmp/data

# Install our package
COPY pyproject.toml setup.py /app/
COPY python/sctools/ /app/python/sctools/
RUN pip install -e .

# Create a non-root user
RUN useradd -m -u 1000 biouser
RUN chown -R biouser:biouser /app
USER biouser

# Set up entry points
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/entrypoint-r.sh /entrypoint-r.sh
RUN chmod +x /entrypoint.sh /entrypoint-r.sh

# Set working directory for mounted volumes
WORKDIR /data

# Default entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
