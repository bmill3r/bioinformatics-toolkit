#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DimensionalityReduction: Dimensionality Reduction for Single-Cell RNA-seq Data

This module provides the DimensionalityReduction class for performing various
dimensionality reduction techniques on single-cell RNA-seq data. It includes
implementations of PCA, UMAP, and t-SNE, with utilities for visualization and analysis
of the reduced-dimensional representations.

Dimensionality reduction is essential for visualizing and analyzing high-dimensional
single-cell data, revealing underlying structure and relationships between cells.

Key features:
- Principal Component Analysis (PCA) with variance analysis
- Uniform Manifold Approximation and Projection (UMAP)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Visualization of explained variance in PCA
- Support for highly variable gene selection

Upstream dependencies:
- SingleCellQC for quality control and filtering
- Normalization for data normalization
- FeatureSelection for identifying highly variable genes

Downstream applications:
- EnhancedVisualization for advanced plotting
- Clustering algorithms for cell type identification
- Trajectory inference for developmental processes

Author: Your Name
Date: Current Date
Version: 0.1.0
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple, Dict
import warnings


class DimensionalityReduction:
    """
    Class for dimensionality reduction of single-cell data.
    
    This class provides methods for reducing the dimensionality of single-cell RNA-seq 
    data using various techniques like PCA, UMAP, and t-SNE. These methods help visualize 
    and analyze high-dimensional gene expression data in lower-dimensional spaces.
    
    Attributes:
        adata (AnnData): AnnData object containing the single-cell data.
    """
    
    def __init__(self, adata: ad.AnnData):
        """
        Initialize with AnnData object containing normalized and feature-selected data.
        
        Args:
            adata (AnnData): AnnData object with normalized gene expression data.
                            This should be normalized data with selected features,
                            typically output from the FeatureSelection class.
        """
        self.adata = adata
        
    def run_pca(self, 
               n_comps: int = 50,
               use_highly_variable: bool = True,
               svd_solver: str = 'arpack',
               random_state: int = 42,
               return_info: bool = False,
               inplace: bool = True) -> Optional[Union[ad.AnnData, Tuple[ad.AnnData, Dict]]]:
        """
        Run Principal Component Analysis (PCA) on the data.
        
        This function performs PCA on the gene expression data, optionally focusing
        on highly variable genes. PCA is typically the first dimensionality reduction
        step in single-cell analysis.
        
        Args:
            n_comps: Number of principal components to compute
            use_highly_variable: Whether to use only highly variable genes
            svd_solver: SVD solver to use ('arpack', 'randomized', 'auto', etc.)
            random_state: Random seed for reproducibility
            return_info: Whether to return PCA information dictionary
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns AnnData with PCA results.
            If return_info=True, also returns PCA information dictionary.
            
        Note:
            The PCA results are stored in adata.obsm['X_pca'] and the loadings
            in adata.varm['PCs'] if available.
        """
        print(f"Running PCA with {n_comps} components")
        
        # Create a copy if not inplace
        adata = self.adata if inplace else self.adata.copy()
        
        # Check for highly variable genes if requested
        if use_highly_variable and 'highly_variable' in adata.var.columns:
            # Use only highly variable genes for PCA
            print(f"Using {sum(adata.var.highly_variable)} highly variable genes for PCA")
            adata_use = adata[:, adata.var.highly_variable]
        else:
            # Use all genes
            adata_use = adata
            
        # Run PCA
        sc.tl.pca(
            adata_use,
            n_comps=n_comps,
            svd_solver=svd_solver,
            random_state=random_state,
            return_info=return_info
        )
        
        # Copy PCA results to the original object if using HVGs
        if use_highly_variable and 'highly_variable' in adata.var.columns:
            # Copy the PCA projection (cell coordinates in PC space)
            adata.obsm['X_pca'] = adata_use.obsm['X_pca']
            
            # Copy the PC loadings (gene weights for each PC) if available
            if 'PCs' in adata_use.varm:
                adata.varm['PCs'] = np.zeros((adata.shape[1], n_comps))
                # Only fill in values for highly variable genes
                adata.varm['PCs'][adata.var.highly_variable] = adata_use.varm['PCs']
                
            # Copy the PCA information (variance explained, etc.) if available
            if hasattr(adata_use, 'uns') and 'pca' in adata_use.uns:
                adata.uns['pca'] = adata_use.uns['pca']
        
        # Update the instance
        if inplace:
            self.adata = adata
            if return_info and 'pca' in adata.uns:
                return adata.uns['pca']
        else:
            if return_info and 'pca' in adata.uns:
                return adata, adata.uns['pca']
            return adata
    
    def plot_pca_variance(self, 
                         n_pcs: int = 50,
                         log: bool = False,
                         threshold: Optional[float] = None,
                         figsize: Tuple[float, float] = (10, 4),
                         save_path: Optional[str] = None,
                         return_fig: bool = False) -> Optional[plt.Figure]:
        """
        Plot the explained variance ratio of principal components.
        
        This function visualizes the variance explained by each principal component
        and the cumulative explained variance. It helps determine how many PCs
        to use for downstream analysis.
        
        Args:
            n_pcs: Number of principal components to plot
            log: Whether to use log scale for y-axis
            threshold: If specified, draw a horizontal line at this cumulative explained variance
            figsize: Figure size
            save_path: Path to save the figure (None displays the plot instead)
            return_fig: If True, return the figure object
            
        Returns:
            If return_fig is True, returns the matplotlib figure object
            
        Raises:
            ValueError: If PCA hasn't been performed yet or variance information is missing
        """
        if 'pca' not in self.adata.uns or 'variance_ratio' not in self.adata.uns['pca']:
            raise ValueError("PCA hasn't been performed yet or variance_ratio information is missing.")
            
        # Get variance data
        variance_ratio = self.adata.uns['pca']['variance_ratio']
        variance_cumsum = np.cumsum(variance_ratio)
        
        # Limit to n_pcs (or maximum available)
        n_pcs = min(n_pcs, len(variance_ratio))
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Variance explained by each PC
        ax1.plot(range(1, n_pcs + 1), variance_ratio[:n_pcs] * 100, 'o-')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio (%)')
        if log:
            ax1.set_yscale('log')
        ax1.set_title('PCA Explained Variance')
        ax1.set_xticks(range(1, n_pcs + 1, 5))
        ax1.grid(True)
        
        # Plot 2: Cumulative variance explained
        ax2.plot(range(1, n_pcs + 1), variance_cumsum[:n_pcs] * 100, 'o-')
        ax2.set_xlabel('Number of Principal Components')
        ax2.set_ylabel('Cumulative Explained Variance (%)')
        ax2.set_title('PCA Cumulative Explained Variance')
        ax2.set_xticks(range(1, n_pcs + 1, 5))
        ax2.grid(True)
        
        # Add threshold line if specified
        if threshold is not None:
            ax2.axhline(y=threshold * 100, color='r', linestyle='--', 
                       label=f'{threshold*100:.1f}% Variance')
                       
            # Find the minimum number of PCs required to reach the threshold
            n_pcs_threshold = np.argmax(variance_cumsum >= threshold) + 1
            ax2.axvline(x=n_pcs_threshold, color='r', linestyle='--')
            ax2.text(n_pcs_threshold + 0.1, 10, f'{n_pcs_threshold} PCs', 
                   verticalalignment='bottom', horizontalalignment='left')
            ax2.legend()
            
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
            
        # Return figure if requested
        if return_fig:
            return fig
        
        # Show the plot if not saving
        if not save_path:
            plt.show()
            
        # Close the figure
        plt.close()
            
    def run_umap(self,
                n_components: int = 2,
                min_dist: float = 0.5,
                spread: float = 1.0,
                n_neighbors: int = 15,
                metric: str = 'euclidean',
                random_state: int = 42,
                inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Run UMAP on the data.
        
        Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction
        technique that preserves both local and global structure in the data. It is
        commonly used for visualization and exploration of single-cell data.
        
        Args:
            n_components: Number of dimensions for the embedding
            min_dist: Minimum distance between points in the embedding (lower = tighter clusters)
            spread: Spread of the embedding (affects global structure)
            n_neighbors: Number of neighbors for the KNN graph (higher = more global structure)
            metric: Distance metric to use
            random_state: Random seed for reproducibility
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns AnnData with UMAP embedding
            
        Note:
            This function computes or uses an existing nearest neighbors graph before
            running UMAP. The UMAP embedding is stored in adata.obsm['X_umap'].
        """
        print(f"Running UMAP with {n_components} components and {n_neighbors} neighbors")
        
        # Work with either the original object or a copy
        adata = self.adata if inplace else self.adata.copy()
        
        # Check if PCA has been performed
        if 'X_pca' not in adata.obsm:
            print("Warning: No PCA embedding found. Running PCA first.")
            sc.tl.pca(adata)
            
        # Check if neighbors have been computed
        if 'neighbors' not in adata.uns:
            # Compute neighbors graph (required for UMAP)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric=metric, random_state=random_state)
        else:
            print("Using existing neighbors graph")
            
        # Run UMAP
        sc.tl.umap(
            adata,
            n_components=n_components,
            min_dist=min_dist,
            spread=spread,
            random_state=random_state
        )
        
        # Update the object
        if inplace:
            self.adata = adata
        else:
            return adata
            
    def run_tsne(self,
                n_components: int = 2,
                perplexity: float = 30.0,
                early_exaggeration: float = 12.0,
                learning_rate: float = 200.0,
                random_state: int = 42,
                n_jobs: int = 8,
                inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Run t-SNE on the data.
        
        t-Distributed Stochastic Neighbor Embedding (t-SNE) is a nonlinear dimensionality
        reduction technique well-suited for visualizing high-dimensional data. It excels
        at preserving local structure but may not preserve global structure as well as UMAP.
        
        Args:
            n_components: Number of dimensions for the embedding
            perplexity: Perplexity parameter (roughly, how many neighbors each point considers)
            early_exaggeration: Early exaggeration factor (higher = more space between clusters)
            learning_rate: Learning rate for t-SNE optimization
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs for computation
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns AnnData with t-SNE embedding
            
        Note:
            The t-SNE embedding is stored in adata.obsm['X_tsne']. This implementation
            runs t-SNE on the PCA representation for efficiency.
        """
        print(f"Running t-SNE with {n_components} components and perplexity {perplexity}")
        
        # Work with either the original object or a copy
        adata = self.adata if inplace else self.adata.copy()
        
        # Check if PCA has been performed
        if 'X_pca' not in adata.obsm:
            print("Warning: No PCA embedding found. Running PCA first.")
            sc.tl.pca(adata)
            
        # Run t-SNE
        sc.tl.tsne(
            adata,
            n_components=n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=n_jobs,
            use_rep='X_pca'  # Use PCA representation for faster computation
        )
        
        # Update the object
        if inplace:
            self.adata = adata
        else:
            return adata
            
    def run_diffmap(self,
                  n_comps: int = 15,
                  n_neighbors: int = 15,
                  max_dim: int = 100,
                  random_state: int = 42,
                  inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Run diffusion map dimensionality reduction.
        
        Diffusion maps learn the manifold structure of the data by modeling the
        diffusion process on a graph. They're particularly useful for trajectory
        inference and capturing continuous processes like differentiation.
        
        Args:
            n_comps: Number of diffusion components to compute
            n_neighbors: Number of neighbors for the graph
            max_dim: Maximum number of dimensions to use from the original data
            random_state: Random seed for reproducibility
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns AnnData with diffusion map embedding
            
        Note:
            This function computes or uses an existing nearest neighbors graph before
            running the diffusion map. The result is stored in adata.obsm['X_diffmap'].
        """
        print(f"Running diffusion map with {n_comps} components")
        
        # Work with either the original object or a copy
        adata = self.adata if inplace else self.adata.copy()
        
        # Check if PCA has been performed
        if 'X_pca' not in adata.obsm:
            print("Warning: No PCA embedding found. Running PCA first.")
            sc.tl.pca(adata)
            
        # Check if neighbors have been computed
        if 'neighbors' not in adata.uns:
            # Compute neighbors graph (required for diffusion map)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=random_state)
        else:
            print("Using existing neighbors graph")
            
        # Run diffusion map
        sc.tl.diffmap(
            adata,
            n_comps=n_comps,
            random_state=random_state
        )
        
        # Update the object
        if inplace:
            self.adata = adata
        else:
            return adata
            
    def run_clustering(self,
                      method: str = 'leiden',
                      resolution: float = 1.0,
                      n_neighbors: int = 15,
                      random_state: int = 42,
                      key_added: str = None,
                      inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Run clustering on the dimensionality-reduced data.
        
        This function performs clustering on the neighborhood graph of cells,
        typically after dimensionality reduction with PCA/UMAP. It's useful
        for identifying cell types or states.
        
        Args:
            method: Clustering method ('leiden' or 'louvain')
            resolution: Resolution parameter controlling clustering granularity
            n_neighbors: Number of neighbors for the graph (if not already computed)
            random_state: Random seed for reproducibility
            key_added: Key under which to add the cluster labels (default: method name)
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns AnnData with clustering results
            
        Note:
            This function computes or uses an existing nearest neighbors graph before
            running clustering. The cluster labels are stored in adata.obs[key_added].
        """
        if method not in ['leiden', 'louvain']:
            raise ValueError(f"Unsupported clustering method: {method}. Use 'leiden' or 'louvain'.")
            
        print(f"Running {method} clustering with resolution {resolution}")
        
        # Set default key name based on the method if not provided
        if key_added is None:
            key_added = method
            
        # Work with either the original object or a copy
        adata = self.adata if inplace else self.adata.copy()
        
        # Check if neighbors have been computed
        if 'neighbors' not in adata.uns:
            # Check if PCA has been performed
            if 'X_pca' not in adata.obsm:
                print("Warning: No PCA embedding found. Running PCA first.")
                sc.tl.pca(adata)
                
            # Compute neighbors graph (required for clustering)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=random_state)
            
        # Run the specified clustering method
        if method == 'leiden':
            sc.tl.leiden(
                adata,
                resolution=resolution,
                key_added=key_added,
                random_state=random_state
            )
        else:  # louvain
            sc.tl.louvain(
                adata,
                resolution=resolution,
                key_added=key_added,
                random_state=random_state
            )
            
        print(f"Identified {adata.obs[key_added].nunique()} clusters")
        
        # Update the object
        if inplace:
            self.adata = adata
        else:
            return adata
