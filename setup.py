"""Setup script for the sctools package"""

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Main setup configuration
setup(
    name="sctools",
    version="0.1.0",
    author="Your Name",
    author_email="youremail@example.com",
    description="A comprehensive bioinformatics toolkit for single-cell and spatial transcriptomics analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sctools",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/sctools/issues",
        "Documentation": "https://sctools.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/sctools",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.8",
    install_requires=[
        "scanpy>=1.9.0",
        "anndata>=0.8.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "plotly>=5.5.0",
        "boto3>=1.20.0",
        "s3fs>=2022.1.0",
        "h5py>=3.6.0",
        "zarr>=2.10.0",
        "leidenalg>=0.8.0",
        "umap-learn>=0.5.0",
        "pysal>=2.3.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "black>=22.1.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "spatial": [
            "squidpy>=1.2.0",
            "napari[all]>=0.4.16",
            "spatialdata-io",
        ],
        "velocity": [
            "scvelo>=0.2.4",
            "cellrank>=1.5.1",
        ],
        "gpu": [
            "cupy-cuda11x",
            "rapids-core-cuda11x",
        ],
    },
    entry_points={
        "console_scripts": [
            "sctools=sctools.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
