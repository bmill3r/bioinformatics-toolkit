"""
sctools: A comprehensive bioinformatics toolkit for single-cell and spatial transcriptomics analysis.

This package provides tools for quality control, normalization, dimensionality reduction,
feature selection, visualization, and spatial analysis of single-cell data.
"""

# Version
__version__ = "0.1.0"

# Import main classes for easier access
from .qc import SingleCellQC
from .normalization import Normalization
from .feature_selection import FeatureSelection
from .dim_reduction import DimensionalityReduction
from .visualization import EnhancedVisualization
from .spatial import SpatialAnalysis
from .geneset import GeneSetScoring
from .utils.s3_utils import S3Utils
