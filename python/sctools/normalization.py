#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Normalization: Methods for normalizing single-cell RNA-seq data

This module provides the Normalization class for normalizing single-cell RNA-seq data
using various state-of-the-art methods. It supports standard log normalization, 
scran-based pooling normalization, sctransform variance-stabilizing transformation,
and centered log-ratio normalization.

Normalization is a critical step in single-cell analysis to correct for technical
variations such as sequencing depth differences between cells, enabling meaningful
comparisons of gene expression across cells.

Key features:
- Standard log normalization with customizable scale factor
- scran normalization for accurate size factor estimation
- sctransform variance-stabilizing transformation
- Centered log-ratio normalization for compositional data
- Support for both in-place and copy operations

Upstream dependencies:
- SingleCellQC for quality control and filtering before normalization

Downstream applications:
- FeatureSelection for finding highly variable genes
- DimensionalityReduction for PCA, UMAP, etc.
- GeneSetScoring for pathway and signature analysis

Author: Your Name
Date: Current Date
Version: 0.1.0
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from typing import Union, Optional
import warnings


class Normalization:
    """
    Class for normalizing single-cell data using various methods.
    
    This class provides implementations of several normalization methods for
    single-cell RNA-seq data, each addressing different aspects of technical variation.
    The choice of normalization method can significantly impact downstream analysis.
    
    Attributes:
        adata (AnnData): AnnData object containing the single-cell data.
    """
    
    def __init__(self, adata: ad.AnnData):
        """
        Initialize with AnnData object containing raw counts.
        
        Args:
            adata (AnnData): AnnData object containing gene expression data.
                            This should be quality-controlled data, typically
                            output from the SingleCellQC class.
        """
        self.adata = adata
        
    def log_norm(self, 
               scale_factor: float = 10000, 
               log_base: float = 2, 
               inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Perform standard log normalization (library size normalization).
        
        This function normalizes cells by total counts, multiplies by a scale factor,
        and applies a log transformation. The normalized values are:
        log(1 + (count * scale_factor / total_counts))
        
        This is the most common normalization method for single-cell RNA-seq data.
        
        Args:
            scale_factor: Scale factor for library size normalization
            log_base: Base for the logarithm (2 for log2, 10 for log10, e for ln)
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns a normalized AnnData object
        """
        # Work with either the original object or a copy
        adata = self.adata if inplace else self.adata.copy()
        
        print(f"Performing log normalization (scale_factor={scale_factor}, log_base={log_base})")
        
        # Library size normalization (divides counts by library size and multiplies by scale factor)
        sc.pp.normalize_total(adata, target_sum=scale_factor)
        
        # Log transformation (adds 1 to avoid log(0))
        sc.pp.log1p(adata, base=log_base)
        
        # Store normalization parameters for reference
        adata.uns['normalization'] = {
            'method': 'log',
            'scale_factor': scale_factor,
            'log_base': log_base
        }
        
        # Return or update in place
        if not inplace:
            return adata
        else:
            self.adata = adata
            
    def scran_norm(self, 
                 n_pools: int = 10, 
                 min_mean: float = 0.1,
                 inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Perform normalization using the scran method (pooling-based size factors).
        
        This function implements the scran normalization algorithm from Lun et al.,
        which estimates cell-specific size factors using pools of cells to reduce
        the impact of zero counts. It's more robust for sparse datasets.
        
        Args:
            n_pools: Number of pools for scran normalization
            min_mean: Minimum mean expression for genes to be used in normalization
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns a normalized AnnData object
            
        Note:
            Requires rpy2 and R packages (scran, BiocParallel, SingleCellExperiment).
            Falls back to standard log normalization if dependencies are not available.
        """
        # Work with either the original object or a copy
        adata = self.adata if inplace else self.adata.copy()
        
        print(f"Performing scran normalization (n_pools={n_pools})")
        
        try:
            # Import R interface libraries
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
            from rpy2.robjects import numpy2ri, pandas2ri
            from rpy2.robjects.conversion import localconverter
            
            # Enable automatic conversion between R and Python objects
            numpy2ri.activate()
            pandas2ri.activate()
            
            # Import necessary R packages
            importr('scran')
            importr('BiocParallel')
            importr('SingleCellExperiment')
            
            # Define R function for scran normalization
            ro.r('''
            normalize_scran <- function(counts, n_pools, min_mean) {
                library(scran)
                library(BiocParallel)
                library(SingleCellExperiment)
                
                # Create SingleCellExperiment
                # Note: transpose because R uses genes as rows
                sce <- SingleCellExperiment(list(counts=t(counts)))
                
                # Calculate size factors using pooling
                clusters <- quickCluster(sce, min.mean=min.mean, n.cores=1)
                sce <- computeSumFactors(sce, clusters=clusters, min.mean=min.mean, n.cores=1)
                
                # Get size factors
                size_factors <- sizeFactors(sce)
                
                return(size_factors)
            }
            ''')
            
            # Get raw counts
            if sparse.issparse(adata.X):
                counts = adata.X.toarray()
            else:
                counts = adata.X
                
            # Run scran normalization in R
            normalize_scran = ro.globalenv['normalize_scran']
            size_factors = np.array(normalize_scran(counts, n_pools, min_mean))
            
            # Apply size factors
            adata.obs['size_factors'] = size_factors
            
            # Make a copy of the expression matrix to avoid modifying the original
            adata.X = adata.X.copy()
            
            # Normalize each cell by its size factor
            for i in range(adata.n_obs):
                if sparse.issparse(adata.X):
                    # For sparse matrix
                    adata.X[i] = adata.X[i].multiply(1.0 / size_factors[i])
                else:
                    # For dense array
                    adata.X[i] = adata.X[i] / size_factors[i]
                
            # Log transform the normalized values
            sc.pp.log1p(adata)
            
            # Store normalization parameters
            adata.uns['normalization'] = {
                'method': 'scran',
                'n_pools': n_pools,
                'min_mean': min_mean
            }
            
            # Clean up R interface
            numpy2ri.deactivate()
            pandas2ri.deactivate()
            
        except ImportError:
            # Fall back to standard normalization if R interface not available
            print("rpy2 or required R packages not available. Falling back to scanpy's normalize_total.")
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            
            # Store normalization parameters with note
            adata.uns['normalization'] = {
                'method': 'log',
                'scale_factor': 1e4,
                'log_base': 2,
                'note': 'Fallback from scran due to missing dependencies'
            }
            
        # Return or update in place
        if not inplace:
            return adata
        else:
            self.adata = adata
            
    def sctransform(self, 
                   n_genes: int = 3000, 
                   n_cells: int = 5000,
                   inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Perform normalization using the sctransform method.
        
        This function implements the sctransform normalization approach from Hafemeister & Satija,
        which models the mean-variance relationship of gene expression data using
        regularized negative binomial regression.
        
        Args:
            n_genes: Maximum number of genes to use (for large datasets)
            n_cells: Maximum number of cells to use (for large datasets)
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns a normalized AnnData object
            
        Note:
            Requires rpy2 and R packages (sctransform, Matrix).
            Falls back to standard log normalization if dependencies are not available.
        """
        # Work with either the original object or a copy
        adata = self.adata if inplace else self.adata.copy()
        
        print(f"Performing sctransform normalization")
        
        try:
            # Import R interface libraries
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
            from rpy2.robjects import numpy2ri, pandas2ri
            from rpy2.robjects.conversion import localconverter
            
            # Enable automatic conversion between R and Python objects
            numpy2ri.activate()
            pandas2ri.activate()
            
            # Import necessary R packages
            importr('sctransform')
            importr('Matrix')
            
            # Define R function for sctransform
            ro.r('''
            normalize_sctransform <- function(counts, n_genes, n_cells) {
                library(sctransform)
                library(Matrix)
                
                # Convert to sparse matrix if needed
                counts_matrix <- counts
                if (!inherits(counts_matrix, "dgCMatrix")) {
                    counts_matrix <- as(counts_matrix, "dgCMatrix")
                }
                
                # Subsample for very large datasets
                if (ncol(counts_matrix) > n_genes || nrow(counts_matrix) > n_cells) {
                    genes_use <- sample(1:ncol(counts_matrix), min(n_genes, ncol(counts_matrix)))
                    cells_use <- sample(1:nrow(counts_matrix), min(n_cells, nrow(counts_matrix)))
                    counts_matrix <- counts_matrix[cells_use, genes_use]
                }
                
                # Run sctransform
                vst_out <- vst(counts_matrix, return_corrected_umi=TRUE, verbose=FALSE)
                
                # Extract results
                pearson_residuals <- vst_out$y
                corrected_counts <- vst_out$umi_corrected
                
                return(list(
                    pearson_residuals = pearson_residuals,
                    corrected_counts = corrected_counts
                ))
            }
            ''')
            
            # Get raw counts
            if sparse.issparse(adata.X):
                counts = adata.X.copy()
            else:
                # Convert to sparse matrix for efficiency
                counts = sparse.csr_matrix(adata.X)
                
            # Run sctransform
            normalize_sctransform = ro.globalenv['normalize_sctransform']
            result = normalize_sctransform(counts, n_genes, n_cells)
            
            # Extract results
            corrected_counts = np.array(result[1])
            
            # Store original counts in a layer
            adata.layers['counts'] = adata.X.copy()
            
            # Replace expression matrix with corrected counts
            adata.X = corrected_counts
            
            # Store normalization parameters
            adata.uns['normalization'] = {
                'method': 'sctransform',
                'n_genes': n_genes,
                'n_cells': n_cells
            }
            
            # Clean up R interface
            numpy2ri.deactivate()
            pandas2ri.deactivate()
            
        except ImportError:
            # Fall back to standard normalization if R interface not available
            print("rpy2 or sctransform not available. Falling back to scanpy's normalize_total.")
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            
            # Store normalization parameters with note
            adata.uns['normalization'] = {
                'method': 'log',
                'scale_factor': 1e4,
                'log_base': 2,
                'note': 'Fallback from sctransform due to missing dependencies'
            }
            
        # Return or update in place
        if not inplace:
            return adata
        else:
            self.adata = adata
            
    def clr_norm(self, 
               eps: float = 1.0,
               inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Perform centered log-ratio normalization.
        
        This function implements the centered log-ratio (CLR) normalization, which
        is particularly useful for compositional data. It normalizes each value
        by the geometric mean of all values for that cell.
        
        Formula: log(value / geometric_mean_of_all_values_in_cell)
        
        Args:
            eps: Pseudo-count to add to avoid log(0)
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns a normalized AnnData object
        """
        # Work with either the original object or a copy
        adata = self.adata if inplace else self.adata.copy()
        
        print(f"Performing centered log-ratio normalization")
        
        # Get raw counts
        if sparse.issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X.copy()
            
        # Add pseudocount to avoid log(0)
        X += eps
        
        # Calculate geometric mean for each cell (mean of log values, then exponentiate)
        geo_means = np.exp(np.mean(np.log(X), axis=1))
        
        # Apply CLR normalization
        for i in range(X.shape[0]):
            # Divide by the geometric mean, then take log
            X[i] = np.log(X[i] / geo_means[i])
            
        # Update the expression matrix
        adata.X = X
        
        # Store normalization parameters
        adata.uns['normalization'] = {
            'method': 'clr',
            'eps': eps
        }
        
        # Return or update in place
        if not inplace:
            return adata
        else:
            self.adata = adata
    
    def run_normalization(self, 
                        method: str = 'log', 
                        **kwargs) -> ad.AnnData:
        """
        Run normalization using the specified method.
        
        This is a convenience function that runs the appropriate normalization
        function based on the specified method name.
        
        Args:
            method: Normalization method ('log', 'scran', 'sctransform', 'clr')
            **kwargs: Additional parameters for the specific normalization method
            
        Returns:
            The normalized AnnData object (self.adata)
            
        Raises:
            ValueError: If an unsupported normalization method is specified
        """
        # Call the appropriate normalization method based on the method name
        if method.lower() == 'log':
            self.log_norm(**kwargs)
        elif method.lower() == 'scran':
            self.scran_norm(**kwargs)
        elif method.lower() == 'sctransform':
            self.sctransform(**kwargs)
        elif method.lower() == 'clr':
            self.clr_norm(**kwargs)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
        return self.adata
