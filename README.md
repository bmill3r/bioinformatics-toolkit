# Repository Organization for Bioinformatics Toolkit

This document explains how to organize the codebase for both Python and R implementations of our comprehensive bioinformatics toolkit for single-cell and spatial analysis.

## Repository Structure

To maintain clean organization while supporting both Python and R implementations, we recommend the following structure:

```
bioinformatics-toolkit/
├── LICENSE
├── README.md                     # Main project documentation
├── CONTRIBUTING.md               # Contribution guidelines
├── pyproject.toml                # Python package configuration
├── setup.py                      # Python package installation
├── DESCRIPTION                   # R package description
├── NAMESPACE                     # R namespace file
├── environment.yml               # Conda environment for Python
├── r-environment.yml             # Conda environment for R
├── docker/                       # Docker-related files
│   ├── Dockerfile                # Main Dockerfile
│   ├── docker-compose.yml        # Docker Compose configuration
│   ├── s3_utils.py               # Python S3 utilities for Docker
│   └── s3_utils.R                # R S3 utilities for Docker
├── podman/                       # Podman-related files
│   └── podman_build.sh           # Podman build script
├── python/                       # Python implementation
│   ├── sctools/                  # Python package
│   │   ├── __init__.py           # Package initialization
│   │   ├── qc.py                 # SingleCellQC class
│   │   ├── normalization.py      # Normalization class
│   │   ├── feature_selection.py  # FeatureSelection class
│   │   ├── dim_reduction.py      # DimensionalityReduction class
│   │   ├── visualization.py      # EnhancedVisualization class
│   │   ├── spatial.py            # SpatialAnalysis class
│   │   ├── geneset.py            # GeneSetScoring class
│   │   └── utils/                # Utility functions
│   │       ├── __init__.py
│   │       └── s3_utils.py       # S3Utils class
│   └── tests/                    # Python unit tests
│       ├── __init__.py
│       ├── test_qc.py
│       └── ...
├── R/                            # R implementation
│   ├── SingleCellQC.R            # QC functions
│   ├── Normalization.R           # Normalization functions
│   ├── FeatureSelection.R        # Feature selection functions
│   ├── DimensionalityReduction.R # Dimensionality reduction functions
│   ├── Visualization.R           # Visualization functions
│   ├── SpatialAnalysis.R         # Spatial analysis functions
│   ├── GeneSetScoring.R          # Gene set scoring functions
│   └── S3Utils.R                 # S3 utilities
├── man/                          # R package documentation
│   ├── SingleCellQC.Rd           # Generated from roxygen comments
│   └── ...
├── docs/                         # Documentation for both implementations
│   ├── python/                   # Python docs
│   │   ├── index.md              # Python documentation index
│   │   └── ...
│   ├── r/                        # R docs
│   │   ├── index.md              # R documentation index
│   │   └── ...
│   └── tutorials/                # General tutorials
│       ├── python_tutorial.md    # Python tutorial
│       ├── r_tutorial.md         # R tutorial
│       └── ...
├── examples/                     # Example scripts and notebooks
│   ├── python/                   # Python examples
│   │   ├── basic_workflow.ipynb  # Jupyter notebook with basic workflow
│   │   └── ...
│   └── r/                        # R examples
│       ├── basic_workflow.Rmd    # R Markdown with basic workflow
│       └── ...
└── data/                         # Small example datasets
    ├── README.md                 # Data descriptions and sources
    └── mini_pbmc/                # Small example dataset
```

## Python Package Organization

The Python implementation is structured as a standard Python package with submodules for each major component. Each module corresponds to a class or set of related functions:

1. **qc.py**: Contains the `SingleCellQC` class for quality control
2. **normalization.py**: Contains the `Normalization` class
3. **feature_selection.py**: Contains the `FeatureSelection` class
4. **dim_reduction.py**: Contains the `DimensionalityReduction` class
5. **visualization.py**: Contains the `EnhancedVisualization` class
6. **spatial.py**: Contains the `SpatialAnalysis` class
7. **geneset.py**: Contains the `GeneSetScoring` class
8. **utils/s3_utils.py**: Contains the `S3Utils` class

### Python Imports and Usage

```python
# Import individual components
from sctools.qc import SingleCellQC
from sctools.normalization import Normalization
from sctools.feature_selection import FeatureSelection
from sctools.dim_reduction import DimensionalityReduction
from sctools.visualization import EnhancedVisualization
from sctools.spatial import SpatialAnalysis
from sctools.geneset import GeneSetScoring
from sctools.utils.s3_utils import S3Utils

# Example workflow
qc = SingleCellQC()
adata = qc.load_data("path/to/data")
qc.calculate_qc_metrics()
qc.filter_cells()

norm = Normalization(qc.adata)
norm.log_norm()

fs = FeatureSelection(norm.adata)
fs.find_highly_variable_genes()

dr = DimensionalityReduction(fs.adata)
dr.run_pca()
dr.run_umap()

viz = EnhancedVisualization(dr.adata)
viz.scatter(x='X_umap-0', y='X_umap-1', color='leiden')
```

## R Package Organization

The R implementation uses a more functional approach, with each R file containing a main class constructor and associated methods. This aligns with common R package design patterns:

1. **SingleCellQC.R**: Contains the `SingleCellQC()` constructor function and methods
2. **Normalization.R**: Contains the `Normalization()` constructor function and methods
3. **FeatureSelection.R**: Contains the `FeatureSelection()` constructor function and methods
4. **DimensionalityReduction.R**: Contains the `DimensionalityReduction()` constructor function and methods
5. **Visualization.R**: Contains the `Visualization()` constructor function and methods
6. **SpatialAnalysis.R**: Contains the `SpatialAnalysis()` constructor function and methods
7. **GeneSetScoring.R**: Contains the `GeneSetScoring()` constructor function and methods
8. **S3Utils.R**: Contains the `S3Utils()` constructor function and methods

### R Imports and Usage

```R
# Load the package (if installed)
library(sctools)

# Or source individual files during development
source("R/SingleCellQC.R")
source("R/Normalization.R")
source("R/FeatureSelection.R")
source("R/DimensionalityReduction.R")
source("R/Visualization.R")
source("R/S3Utils.R")

# To register the R kernel with Jupyter (needed when environment is first created)
# IRkernel::installspec(name = 'sctools-r', displayname = 'R (sctools-r)')

# Example workflow
qc <- SingleCellQC()
seurat_obj <- qc$load_data("path/to/data")
qc$calculate_qc_metrics()
qc$filter_cells()

norm <- Normalization(qc$seurat)
norm$log_norm()

fs <- FeatureSelection(norm$seurat)
fs$find_variable_features()

dr <- DimensionalityReduction(fs$seurat)
dr$run_pca()
dr$run_umap()

viz <- Visualization(dr$seurat)
viz$dim_plot(reduction = "umap", group_by = "seurat_clusters")
```

## File and Function Relationships

Here's how the components relate to each other in both implementations:

### Python

1. `SingleCellQC` -> loads data, calculates QC metrics, filters cells/genes
2. Output of `SingleCellQC` -> Input to `Normalization`
3. Output of `Normalization` -> Input to `FeatureSelection`
4. Output of `FeatureSelection` -> Input to `DimensionalityReduction`
5. Output of `DimensionalityReduction` -> Input to `EnhancedVisualization` and `GeneSetScoring`
6. `SpatialAnalysis` can be used after `SingleCellQC` or later steps

### R

1. `SingleCellQC()` -> loads data, calculates QC metrics, filters cells/genes
2. Output of `SingleCellQC()` -> Input to `Normalization()`
3. Output of `Normalization()` -> Input to `FeatureSelection()`
4. Output of `FeatureSelection()` -> Input to `DimensionalityReduction()`
5. Output of `DimensionalityReduction()` -> Input to `Visualization()` and `GeneSetScoring()`
6. `SpatialAnalysis()` can be used after `SingleCellQC()` or later steps

## S3 Utilities Integration

The S3 utilities in both Python and R implementations can be used at any stage of the workflow to load data from or save results to AWS S3 buckets.

### Python S3 Integration

```python
# Initialize S3 utilities
s3 = S3Utils(profile_name="my-profile")

# Load data from S3
adata = s3.read_h5ad('my-bucket', 'path/to/data.h5ad')
qc = SingleCellQC()
qc.load_data(adata)

# Later in the workflow, save results to S3
s3.write_h5ad(dr.adata, 'my-bucket', 'results/processed_data.h5ad')

# Save a figure to S3
fig = viz.scatter(x='X_umap-0', y='X_umap-1', color='leiden', return_fig=True)
s3.save_figure(fig, 'my-bucket', 'results/umap.png')
```

### R S3 Integration

```R
# Initialize S3 utilities
s3 <- S3Utils(profile_name = "my-profile")

# Load data from S3
seurat_obj <- s3$read_seurat('my-bucket', 'path/to/data.rds')
qc <- SingleCellQC()
qc$load_data(seurat_obj)

# Later in the workflow, save results to S3
s3$write_seurat(dr$seurat, 'my-bucket', 'results/processed_data.rds')

# Save a figure to S3
p <- viz$dim_plot(reduction = "umap", group_by = "seurat_clusters")
s3$save_plot(p, 'my-bucket', 'results/umap.png')
```

## Building and Installing

### Python Package Installation

```bash
# Install from source
git clone https://github.com/yourusername/bioinformatics-toolkit.git
cd bioinformatics-toolkit
pip install -e .

# Or create conda environments
# For Python environment:
conda env create -f environment-py.yml

# For R environment:
conda env create -f environment-r.yml

# Important: After creating the R environment, you need to register the R kernel with Jupyter
# so it appears in Jupyter Lab sessions:
conda activate sctools-r
R -e "IRkernel::installspec(name = 'sctools-r', displayname = 'R (sctools-r)')"

# Note: For Docker deployment, include this registration step in your setup script
# when building the Docker image.

conda env create -f environment.yml
conda activate sctools-py
```

### R Package Installation

```R
# Install from source
# In R console:
install.packages("devtools")
devtools::install_github("yourusername/bioinformatics-toolkit", subdir = "R")

# Or create conda environment
# In terminal:
conda env create -f r-environment.yml
conda activate sctools-r
```

## Docker Usage

```bash
# Build the Docker image
cd bioinformatics-toolkit
docker build -t sctools:latest .

# Run Python environment
docker run -it -v $(pwd):/data sctools:latest python

# Run R environment
docker run -it -v $(pwd):/data --entrypoint /entrypoint-r.sh sctools:latest R
```

## Podman Usage

```bash
# Build with Podman
cd bioinformatics-toolkit
podman build -t sctools:latest .

# Run Python environment
podman run -it -v $(pwd):/data:Z sctools:latest python

# Run R environment
podman run -it -v $(pwd):/data:Z --entrypoint /entrypoint-r.sh sctools:latest R
```

This repository structure and organization ensures a clean, maintainable codebase that supports both Python and R implementations while providing clear documentation and examples for users of either language.
