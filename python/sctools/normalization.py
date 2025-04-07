"""
Single-Cell RNA-seq Normalization Module

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
    
    The Normalization class is typically used after quality control and before feature selection:
    
    - Upstream dependencies:
      * SingleCellQC for data loading, QC metrics calculation, and filtering
    
    - Downstream applications:
      * FeatureSelection for identifying highly variable genes
      * DimensionalityReduction for dimensionality reduction
      * GeneSetScoring for pathway activity scoring
    
    Attributes:
        adata (AnnData): AnnData object containing the single-cell data.
    
    Examples:
        >>> # After QC and filtering
        >>> from sctools.normalization import Normalization
        >>> norm = Normalization(adata)
        >>> 
        >>> # Standard log normalization
        >>> norm.log_norm(scale_factor=10000)
        >>> 
        >>> # SCTransform (if R packages available)
        >>> norm.sctransform()
        >>> 
        >>> # Get normalized data for downstream analysis
        >>> normalized_adata = norm.adata
    """
    
    def __init__(self, adata: ad.AnnData):
        """
        Initialize with AnnData object.
        
        Parameters:
            adata (AnnData): AnnData object containing gene expression data.
            
        Examples:
            >>> norm = Normalization(adata)
        """
        self.adata = adata
        
    def log_norm(self, 
               scale_factor: float = 10000, 
               log_base: float = 2, 
               inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Perform standard log normalization (library size normalization).
        
        This method normalizes cells by their total counts, multiplies by a scale factor,
        and applies a log transformation. This is the most common normalization method
        for single-cell RNA-seq data.
        
        Parameters:
            scale_factor : float
                Scale factor for library size normalization. The counts are 
                normalized to sum to this value for each cell before log transformation.
            log_base : float
                Base for the logarithm (2 for log2, 10 for log10, math.e for ln).
            inplace : bool
                If True, modify self.adata, else return a normalized copy.
                
        Returns:
            Optional[AnnData]
                Normalized AnnData object if inplace is False.
                
        Examples:
            >>> norm = Normalization(adata)
            >>> 
            >>> # Default log normalization (log2 with scale factor 10000)
            >>> norm.log_norm()
            >>> 
            >>> # Custom scale factor and log base
            >>> norm.log_norm(scale_factor=1e6, log_base=10)
            >>> 
            >>> # Return a copy instead of modifying in place
            >>> normalized_copy = norm.log_norm(inplace=False)
        
        Notes:
            - This is equivalent to Seurat's NormalizeData() function with default parameters.
            - The normalized values are: log(1 + (count * scale_factor / total_counts))
            - Stores the normalization info in adata.uns['normalization']
        """
        adata = self.adata if inplace else self.adata.copy()
        
        print(f"Performing log normalization (scale_factor={scale_factor}, log_base={log_base})")
        
        # Library size normalization and log transformation
        sc.pp.normalize_total(adata, target_sum=scale_factor)
        sc.pp.log1p(adata, base=log_base)
        
        # Store normalization parameters
        adata.uns['normalization'] = {
            'method': 'log',
            'scale_factor': scale_factor,
            'log_base': log_base
        }
        
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
        
        This method implements the scran normalization algorithm, which estimates
        cell-specific size factors using pools of cells to reduce the impact of
        zero counts. This approach is more robust than simple library size normalization
        for datasets with high sparsity.
        
        Parameters:
            n_pools : int
                Number of pools for scran normalization.
            min_mean : float
                Minimum mean expression for genes to be used in normalization.
            inplace : bool
                If True, modify self.adata, else return a normalized copy.
                
        Returns:
            Optional[AnnData]
                Normalized AnnData object if inplace is False.
                
        Examples:
            >>> norm = Normalization(adata)
            >>> 
            >>> # Default scran normalization
            >>> try:
            ...     norm.scran_norm()
            ... except ImportError:
            ...     print("rpy2 not available, falling back to log normalization")
            ...     norm.log_norm()
        
        Notes:
            - Requires rpy2 and R packages (scran, BiocParallel, SingleCellExperiment).
            - Falls back to standard log normalization if dependencies are not available.
            - Stores cell-specific size factors in adata.obs['size_factors'].
            - More robust than log_norm() for datasets with many zeros or high technical noise.
            - Stores the normalization info in adata.uns['normalization']
        """
        adata = self.adata if inplace else self.adata.copy()
        
        print(f"Performing scran normalization (n_pools={n_pools})")
        
        try:
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
            from rpy2.robjects import numpy2ri, pandas2ri
            from rpy2.robjects.conversion import localconverter
            
            # Enable automatic conversion
            numpy2ri.activate()
            pandas2ri.activate()
            
            # Import R packages
            importr('scran')
            importr('BiocParallel')
            importr('SingleCellExperiment')
            
            # Convert to SingleCellExperiment
            ro.r('''
            normalize_scran <- function(counts, n_pools, min_mean) {
                library(scran)
                library(BiocParallel)
                library(SingleCellExperiment)
                
                # Create SingleCellExperiment
                sce <- SingleCellExperiment(list(counts=t(counts)))
                
                # Calculate size factors
                clusters <- quickCluster(sce, min.mean=min_mean, n.cores=1)
                sce <- computeSumFactors(sce, clusters=clusters, min.mean=min_mean, n.cores=1)
                
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
                
            # Run scran normalization
            normalize_scran = ro.globalenv['normalize_scran']
            size_factors = np.array(normalize_scran(counts, n_pools, min_mean))
            
            # Apply size factors
            adata.obs['size_factors'] = size_factors
            adata.X = adata.X.copy()  # Make a copy of .X to avoid modifying the original
            
            # Normalize by size factors and log transform
            for i in range(adata.n_obs):
                adata.X[i] = adata.X[i] / size_factors[i]
                
            sc.pp.log1p(adata)
            
            # Store normalization parameters
            adata.uns['normalization'] = {
                'method': 'scran',
                'n_pools': n_pools,
                'min_mean': min_mean
            }
            
            # Clean up
            numpy2ri.deactivate()
            pandas2ri.deactivate()
            
        except ImportError:
            print("rpy2 not available. Falling back to scanpy's normalize_total.")
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            
            # Store normalization parameters
            adata.uns['normalization'] = {
                'method': 'log',
                'scale_factor': 1e4,
                'log_base': 2,
                'note': 'Fallback from scran due to missing dependencies'
            }
            
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
        
        This method implements the sctransform normalization approach, which models the
        mean-variance relationship of gene expression data using regularized negative
        binomial regression. It's particularly effective for handling technical noise
        and improving the signal-to-noise ratio.
        
        Parameters:
            n_genes : int
                Maximum number of genes to use (for large datasets).
            n_cells : int
                Maximum number of cells to use (for large datasets).
            inplace : bool
                If True, modify self.adata, else return a normalized copy.
                
        Returns:
            Optional[AnnData]
                Normalized AnnData object if inplace is False.
                
        Examples:
            >>> norm = Normalization(adata)
            >>> 
            >>> # Default sctransform normalization
            >>> try:
            ...     norm.sctransform()
            ... except ImportError:
            ...     print("rpy2 or sctransform not available, falling back to log normalization")
            ...     norm.log_norm()
        
        Notes:
            - Requires rpy2 and R packages (sctransform, Matrix).
            - Falls back to standard log normalization if dependencies are not available.
            - For very large datasets, subsampling is performed (n_genes, n_cells parameters).
            - Stores original counts in adata.layers['counts'].
            - Considered state-of-the-art for many applications but more computationally intensive.
            - Stores the normalization info in adata.uns['normalization']
        """
        adata = self.adata if inplace else self.adata.copy()
        
        print(f"Performing sctransform normalization")
        
        try:
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
            from rpy2.robjects import numpy2ri, pandas2ri
            from rpy2.robjects.conversion import localconverter
            
            # Enable automatic conversion
            numpy2ri.activate()
            pandas2ri.activate()
            
            # Import R packages
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
                counts = sparse.csr_matrix(adata.X)
                
            # Run sctransform
            normalize_sctransform = ro.globalenv['normalize_sctransform']
            result = normalize_sctransform(counts, n_genes, n_cells)
            
            # Extract results
            corrected_counts = np.array(result[1])
            
            # Store original and normalized data
            adata.layers['counts'] = adata.X.copy()
            adata.X = corrected_counts
            
            # Store normalization parameters
            adata.uns['normalization'] = {
                'method': 'sctransform',
                'n_genes': n_genes,
                'n_cells': n_cells
            }
            
            # Clean up
            numpy2ri.deactivate()
            pandas2ri.deactivate()
            
        except ImportError:
            print("rpy2 or sctransform not available. Falling back to scanpy's normalize_total.")
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            
            # Store normalization parameters
            adata.uns['normalization'] = {
                'method': 'log',
                'scale_factor': 1e4,
                'log_base': 2,
                'note': 'Fallback from sctransform due to missing dependencies'
            }
            
        if not inplace:
            return adata
        else:
            self.adata = adata
            
    def clr_norm(self, 
               eps: float = 1.0,
               inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Perform centered log-ratio normalization.
        
        This method implements the centered log-ratio (CLR) normalization, which
        is particularly useful for compositional data, such as scRNA-seq where the
        total mRNA content per cell is largely arbitrary. It normalizes each value
        by the geometric mean of all values for that cell.
        
        Parameters:
            eps : float
                Pseudo-count to add to avoid log(0).
            inplace : bool
                If True, modify self.adata, else return a normalized copy.
                
        Returns:
            Optional[AnnData]
                Normalized AnnData object if inplace is False.
                
        Examples:
            >>> norm = Normalization(adata)
            >>> 
            >>> # Perform CLR normalization
            >>> norm.clr_norm()
            >>> 
            >>> # CLR with custom pseudo-count
            >>> norm.clr_norm(eps=0.5)
        
        Notes:
            - Useful for compositional data (e.g., CITE-seq ADT data).
            - The formula is: log(value / geometric_mean_of_all_values_in_cell)
            - CLR makes the distribution of each cell have a mean of zero.
            - Stores the normalization info in adata.uns['normalization']
        """
        adata = self.adata if inplace else self.adata.copy()
        
        print(f"Performing centered log-ratio normalization")
        
        # Get raw counts
        if sparse.issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X.copy()
            
        # Add pseudocount
        X += eps
        
        # Calculate geometric mean for each cell
        geo_means = np.exp(np.mean(np.log(X), axis=1))
        
        # Apply CLR normalization
        for i in range(X.shape[0]):
            X[i] = np.log(X[i] / geo_means[i])
            
        adata.X = X
        
        # Store normalization parameters
        adata.uns['normalization'] = {
            'method': 'clr',
            'eps': eps
        }
        
        if not inplace:
            return adata
        else:
            self.adata = adata
    
    def run_normalization(self, 
                        method: str = 'log', 
                        **kwargs) -> ad.AnnData:
        """
        Run normalization using the specified method.
        
        This is a convenience method that runs the appropriate normalization
        function based on the specified method name.
        
        Parameters:
            method : str
                Normalization method ('log', 'scran', 'sctransform', 'clr').
            **kwargs : dict
                Additional parameters for the specific normalization method.
                
        Returns:
            AnnData
                Normalized AnnData object.
                
        Raises:
            ValueError: If an unsupported normalization method is specified.
                
        Examples:
            >>> norm = Normalization(adata)
            >>> 
            >>> # Log normalization with custom parameters
            >>> norm.run_normalization('log', scale_factor=1e6)
            >>> 
            >>> # SCTransform normalization
            >>> norm.run_normalization('sctransform', n_genes=2000)
        
        Notes:
            - This is a convenience wrapper around the specific normalization methods.
            - The method parameter is case-insensitive.
        """
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
