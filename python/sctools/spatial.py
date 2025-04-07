#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SpatialAnalysis: Spatial Transcriptomics Analysis

This module provides the SpatialAnalysis class for analyzing spatial transcriptomics data.
It includes functionality for visualizing gene expression in spatial coordinates,
creating spatial bins, calculating spatial statistics like Moran's I,
and analyzing negative control probes.

Spatial transcriptomics combines gene expression measurements with spatial information,
enabling the study of tissue organization and cellular interactions in their native context.

Key features:
- Visualization of gene expression in spatial coordinates
- Creation of spatial bins for aggregating data
- Calculation of spatial autocorrelation (Moran's I)
- Analysis of negative control probes
- Support for various spatial transcriptomics technologies

Upstream dependencies:
- SingleCellQC for quality control and filtering
- Normalization for data normalization
- DimensionalityReduction for dimensional embeddings

Downstream applications:
- Spatial domain identification
- Cellular neighborhood analysis
- Spatial differential expression
- Tissue architecture analysis

Author: Your Name
Date: Current Date
Version: 0.1.0
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from typing import Union, Optional, Tuple, List, Dict
import warnings


class SpatialAnalysis:
    """
    Class for analyzing spatial transcriptomics data.
    
    This class provides tools for analyzing and visualizing spatial transcriptomics data
    from technologies like 10X Visium, Slide-seq, MERFISH, etc. It enables the visualization
    of gene expression in spatial coordinates, creation of spatial bins, calculation of
    spatial statistics, and analysis of negative control probes.
    
    Attributes:
        adata (AnnData): AnnData object containing the spatial transcriptomics data.
        coordinates (np.ndarray): Spatial coordinates of spots or cells.
        spatial_key (str): Key in adata.obsm where spatial coordinates are stored.
        x_coord (str): Name of x-coordinate in obs if not using obsm[spatial_key].
        y_coord (str): Name of y-coordinate in obs if not using obsm[spatial_key].
    """
    
    def __init__(self, adata: ad.AnnData, 
                 spatial_key: str = 'spatial',
                 x_coord: str = 'x', 
                 y_coord: str = 'y'):
        """
        Initialize with AnnData object and spatial coordinates.
        
        Args:
            adata (AnnData): AnnData object with spatial information.
            spatial_key (str): Key in adata.obsm where spatial coordinates are stored.
            x_coord (str): Name of x-coordinate in obs if not using obsm[spatial_key].
            y_coord (str): Name of y-coordinate in obs if not using obsm[spatial_key].
            
        Raises:
            ValueError: If spatial coordinates cannot be found.
        """
        self.adata = adata
        self.spatial_key = spatial_key
        self.x_coord = x_coord
        self.y_coord = y_coord
        
        # Extract spatial coordinates from the appropriate location
        if spatial_key in adata.obsm:
            # Coordinates are in obsm (e.g., 10X Visium data)
            self.coordinates = adata.obsm[spatial_key]
            print(f"Using spatial coordinates from adata.obsm['{spatial_key}']")
        elif x_coord in adata.obs and y_coord in adata.obs:
            # Coordinates are in separate obs columns
            self.coordinates = np.vstack([adata.obs[x_coord].values, adata.obs[y_coord].values]).T
            print(f"Using spatial coordinates from adata.obs['{x_coord}'] and adata.obs['{y_coord}']")
        else:
            # Cannot find coordinates
            raise ValueError("Spatial coordinates not found in adata.obsm or adata.obs")
            
    def plot_spatial_gene_expression(self, genes, 
                                   ncols: int = 4,
                                   figsize: Tuple[int, int] = None,
                                   cmap: str = 'viridis',
                                   size: float = 10.0,
                                   title_fontsize: int = 10,
                                   show_colorbar: bool = True,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot spatial expression of multiple genes.
        
        This function creates scatter plots of cells/spots in their spatial coordinates,
        colored by gene expression or metadata values.
        
        Args:
            genes: Gene or list of genes to plot (can also be metadata columns)
            ncols: Number of columns in the plot grid
            figsize: Figure size (auto-calculated based on genes if None)
            cmap: Colormap for gene expression
            size: Size of the points in the scatter plot
            title_fontsize: Font size for subplot titles
            show_colorbar: Whether to show the colorbar
            save_path: Path to save the figure (None displays the figure instead)
            
        Returns:
            matplotlib.figure.Figure: The matplotlib figure object
            
        Raises:
            ValueError: If none of the specified genes are found in the dataset
        """
        # Convert single gene to list for consistent processing
        if isinstance(genes, str):
            genes = [genes]
            
        # Validate which genes/metrics exist in the dataset
        valid_genes = []
        for gene in genes:
            if gene in self.adata.var_names:
                # Gene exists in var_names (expression data)
                valid_genes.append(gene)
            elif gene in self.adata.obs.columns:
                # Feature exists in obs (metadata)
                valid_genes.append(gene)
            else:
                print(f"Warning: '{gene}' not found in var_names or obs columns")
                
        if len(valid_genes) == 0:
            raise ValueError("None of the specified genes or metrics were found in the dataset")
            
        genes = valid_genes
            
        # Calculate grid dimensions
        n_genes = len(genes)
        nrows = (n_genes + ncols - 1) // ncols  # Ceiling division
        
        # Calculate figure size if not provided
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
            
        # Create figure
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        
        # Extract coordinates
        x, y = self.coordinates[:, 0], self.coordinates[:, 1]
        
        # Plot each gene/feature
        for i, gene in enumerate(genes):
            row, col = i // ncols, i % ncols
            ax = axs[row, col]
            
            # Get gene expression or metadata values
            if gene in self.adata.var_names:
                # Feature is a gene
                gene_expr = self.adata[:, gene].X
                if sparse.issparse(gene_expr):
                    gene_expr = gene_expr.toarray().flatten()
                values = gene_expr
            else:  # Feature is in metadata
                values = self.adata.obs[gene].values
                # Handle categorical data
                if values.dtype == 'object' or pd.api.types.is_categorical_dtype(values):
                    # Convert categorical to numeric for coloring
                    unique_values = sorted(set(values))
                    value_map = {val: i for i, val in enumerate(unique_values)}
                    values = np.array([value_map[v] for v in values])
                    
            # Create the scatter plot
            scatter = ax.scatter(x, y, c=values, cmap=cmap, s=size, alpha=0.8)
            
            # Add title and axis labels
            ax.set_title(gene, fontsize=title_fontsize)
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_aspect('equal')  # Maintain aspect ratio
            
            # Add colorbar if requested
            if show_colorbar:
                plt.colorbar(scatter, ax=ax, shrink=0.7)
                
        # Remove empty subplots
        for i in range(n_genes, nrows * ncols):
            row, col = i // ncols, i % ncols
            axs[row, col].axis('off')
            
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved spatial gene expression plot to {save_path}")
        
        return fig
    
    def create_spatial_grid(self, bin_size: float,
                          aggr_func: str = 'mean',
                          min_cells: int = 1,
                          genes: Optional[List[str]] = None) -> ad.AnnData:
        """
        Create a grid of spatial bins and aggregate gene expression within each bin.
        
        This function divides the spatial area into square bins of the specified size
        and aggregates gene expression values within each bin. This is useful for
        reducing noise, analyzing at different spatial scales, or matching resolution
        across different datasets.
        
        Args:
            bin_size: Size of the square bins
            aggr_func: Aggregation function ('mean', 'sum', 'median', 'max')
            min_cells: Minimum number of cells required in a bin to keep it
            genes: Specific genes to include (if None, all genes are used)
            
        Returns:
            AnnData object containing the binned data
            
        Raises:
            ValueError: If specified genes are not found in the dataset
        """
        print(f"Creating spatial grid with bin_size={bin_size}, aggr_func={aggr_func}")
        
        # Extract coordinates
        x, y = self.coordinates[:, 0], self.coordinates[:, 1]
        
        # Calculate bin indices for each cell
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Calculate number of bins in each dimension
        n_bins_x = int(np.ceil((x_max - x_min) / bin_size))
        n_bins_y = int(np.ceil((y_max - y_min) / bin_size))
        
        # Assign cells to bins
        x_bin = np.floor((x - x_min) / bin_size).astype(int)
        y_bin = np.floor((y - y_min) / bin_size).astype(int)
        bin_indices = y_bin * n_bins_x + x_bin
        
        # Get unique bins
        unique_bins = np.unique(bin_indices)
        
        # Extract gene expression matrix
        if genes is None:
            # Use all genes
            genes = self.adata.var_names.tolist()
        else:
            # Filter to genes that exist in the dataset
            genes = [gene for gene in genes if gene in self.adata.var_names]
            if len(genes) == 0:
                raise ValueError("None of the specified genes were found in the dataset")
                
        # Extract expression for selected genes
        X = self.adata[:, genes].X
        if sparse.issparse(X):
            X = X.toarray()
            
        # Create matrices for binned data
        n_bins = len(unique_bins)
        n_genes = len(genes)
        binned_X = np.zeros((n_bins, n_genes))
        bin_coords = np.zeros((n_bins, 2))
        bin_counts = np.zeros(n_bins, dtype=int)
        
        # Aggregate data within bins
        for i, bin_idx in enumerate(unique_bins):
            # Find cells in this bin
            cells_in_bin = np.where(bin_indices == bin_idx)[0]
            bin_counts[i] = len(cells_in_bin)
            
            # Skip bins with too few cells
            if bin_counts[i] < min_cells:
                continue
                
            # Calculate aggregated expression using the specified function
            if aggr_func == 'mean':
                binned_X[i] = np.mean(X[cells_in_bin], axis=0)
            elif aggr_func == 'sum':
                binned_X[i] = np.sum(X[cells_in_bin], axis=0)
            elif aggr_func == 'median':
                binned_X[i] = np.median(X[cells_in_bin], axis=0)
            elif aggr_func == 'max':
                binned_X[i] = np.max(X[cells_in_bin], axis=0)
            else:
                raise ValueError(f"Unsupported aggregation function: {aggr_func}")
                
            # Calculate bin coordinates (center of the bin)
            bin_x = (bin_idx % n_bins_x) * bin_size + (bin_size / 2) + x_min
            bin_y = (bin_idx // n_bins_x) * bin_size + (bin_size / 2) + y_min
            bin_coords[i] = [bin_x, bin_y]
            
        # Filter out bins with too few cells
        valid_bins = bin_counts >= min_cells
        binned_X = binned_X[valid_bins]
        bin_coords = bin_coords[valid_bins]
        bin_counts = bin_counts[valid_bins]
        
        # Create binned AnnData object
        binned_adata = ad.AnnData(X=binned_X)
        
        # Set var names (genes)
        binned_adata.var_names = genes
        
        # Set obs names and metadata
        binned_adata.obs_names = [f"bin_{i}" for i in range(binned_X.shape[0])]
        binned_adata.obs['bin_size'] = bin_size
        binned_adata.obs['n_cells'] = bin_counts[valid_bins]
        
        # Store bin coordinates
        binned_adata.obs['x'] = bin_coords[:, 0]
        binned_adata.obs['y'] = bin_coords[:, 1]
        binned_adata.obsm[self.spatial_key] = bin_coords
        
        print(f"Created {binned_adata.shape[0]} spatial bins with {binned_adata.shape[1]} genes")
        
        return binned_adata
    
    def calculate_moran_i(self, 
                        genes: Optional[List[str]] = None,
                        max_genes: int = 1000,
                        n_jobs: int = 1) -> pd.DataFrame:
        """
        Calculate Moran's I spatial autocorrelation for each gene.
        
        Moran's I measures the spatial autocorrelation of gene expression,
        identifying genes with significant spatial patterning. Values range from
        -1 (dispersed) to 1 (clustered), with 0 indicating random distribution.
        
        Args:
            genes: Specific genes to analyze (if None, all genes or a random subset are used)
            max_genes: Maximum number of genes to analyze (to limit computation)
            n_jobs: Number of parallel jobs for computation
            
        Returns:
            DataFrame with Moran's I statistics for each gene
            
        Raises:
            ImportError: If pysal and joblib are not installed
            ValueError: If no valid genes are found
            
        Note:
            This function requires the pysal and joblib packages. Install with:
            pip install pysal joblib
        """
        try:
            # Import spatial analysis libraries
            from pysal.explore import esda
            from pysal.lib import weights
            from joblib import Parallel, delayed
        except ImportError:
            raise ImportError("Please install pysal and joblib: pip install pysal joblib")
        
        print("Calculating Moran's I spatial autocorrelation")
        
        # Extract spatial coordinates
        coords = self.coordinates
        
        # Create spatial weights matrix (k-nearest neighbors)
        knn = 8  # Number of nearest neighbors to consider
        w = weights.KNN(coords, k=knn)
        w.transform = 'r'  # Row-standardized weights
        
        # Determine which genes to analyze
        if genes is None:
            # Use all genes (or a random subset if too many)
            genes = self.adata.var_names.tolist()
        else:
            # Filter to genes that exist in the dataset
            genes = [gene for gene in genes if gene in self.adata.var_names]
            if len(genes) == 0:
                raise ValueError("None of the specified genes were found in the dataset")
                
        # Limit number of genes for computational efficiency
        if len(genes) > max_genes:
            print(f"Limiting analysis to {max_genes} randomly selected genes")
            genes = np.random.choice(genes, max_genes, replace=False)
            
        # Define function to calculate Moran's I for a single gene
        def calculate_single_moran(gene):
            """Calculate Moran's I for a single gene"""
            # Get gene expression
            gene_expr = self.adata[:, gene].X
            if sparse.issparse(gene_expr):
                gene_expr = gene_expr.toarray().flatten()
                
            # Calculate Moran's I
            moran = esda.Moran(gene_expr, w)
            
            return {
                'gene': gene,
                'morans_i': moran.I,         # Moran's I statistic
                'p_value': moran.p_sim,      # p-value from simulation
                'z_score': moran.z_sim       # z-score
            }
        
        # Calculate Moran's I for each gene in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_single_moran)(gene) for gene in genes
        )
        
        # Convert results to DataFrame
        moran_df = pd.DataFrame(results)
        
        # Sort by Moran's I value (highest to lowest)
        moran_df = moran_df.sort_values('morans_i', ascending=False).reset_index(drop=True)
        
        print(f"Calculated Moran's I for {len(genes)} genes")
        
        return moran_df
    
    def analyze_negative_probes(self, 
                              prefix: str = 'Negative',
                              bin_size: Optional[float] = None) -> pd.DataFrame:
        """
        Analyze negative probe statistics within spatial bins.
        
        This function calculates statistics for negative control probes (technical controls)
        across spatial locations. It can work with either original cells or binned data.
        
        Args:
            prefix: Prefix for identifying negative probe genes
            bin_size: Size of spatial bins (if None, original cells are used)
            
        Returns:
            DataFrame with negative probe statistics for each cell or bin
            
        Raises:
            ValueError: If no negative probes are found with the specified prefix
            
        Note:
            Negative probes are often included in spatial transcriptomics data as
            technical controls. They help estimate background noise levels.
        """
        # Identify negative probes by prefix
        negative_probes = [gene for gene in self.adata.var_names if gene.startswith(prefix)]
        
        if len(negative_probes) == 0:
            raise ValueError(f"No negative probes found with prefix '{prefix}'")
            
        print(f"Found {len(negative_probes)} negative probes with prefix '{prefix}'")
        
        # Use original cells or create spatial bins
        if bin_size is None:
            # Use original cells/spots
            adata = self.adata
            spatial_unit = "cell"
        else:
            # Create spatial bins and aggregate data
            adata = self.create_spatial_grid(bin_size=bin_size, genes=negative_probes)
            spatial_unit = "bin"
            
        # Calculate statistics for each spatial unit
        results = []
        
        for i in range(adata.n_obs):
            # Extract expression of negative probes
            expr = adata[i, negative_probes].X
            if sparse.issparse(expr):
                expr = expr.toarray().flatten()
                
            # Calculate various statistics
            result = {
                f'{spatial_unit}_id': adata.obs_names[i],
                'x': adata.obsm[self.spatial_key][i, 0],
                'y': adata.obsm[self.spatial_key][i, 1],
                'negative_mean': np.mean(expr),              # Mean expression
                'negative_sum': np.sum(expr),                # Total counts
                'negative_sd': np.std(expr),                 # Standard deviation
                # Coefficient of variation (SD/mean), handling divide-by-zero
                'negative_cv': np.std(expr) / np.mean(expr) if np.mean(expr) > 0 else np.nan
            }
            
            # Add values for individual probes
            for j, probe in enumerate(negative_probes):
                probe_value = expr[j]
                result[f'{probe}'] = probe_value
                
            results.append(result)
            
        # Convert to DataFrame
        stats_df = pd.DataFrame(results)
        
        return stats_df
    
    def find_spatial_domains(self,
                           n_clusters: int = 10,
                           use_genes: Optional[List[str]] = None,
                           method: str = 'leiden',
                           resolution: float = 1.0,
                           n_neighbors: int = 15,
                           random_state: int = 42) -> None:
        """
        Identify spatial domains based on gene expression patterns.
        
        This function clusters cells or spots based on their gene expression profiles,
        taking into account their spatial relationships. It can identify spatially
        coherent regions with similar expression patterns.
        
        Args:
            n_clusters: Target number of clusters (exact for kmeans, approximate for leiden)
            use_genes: Specific genes to use for clustering (if None, all genes are used)
            method: Clustering method ('kmeans', 'leiden', 'louvain')
            resolution: Resolution parameter for community detection methods
            n_neighbors: Number of neighbors for graph construction
            random_state: Random seed for reproducibility
            
        Raises:
            ValueError: If an unsupported clustering method is specified
            
        Note:
            Results are stored in adata.obs['spatial_domains'] and can be visualized
            using plot_spatial_gene_expression().
        """
        print(f"Finding spatial domains using {method} clustering")
        
        # Create a working copy of the AnnData object
        adata = self.adata.copy()
        
        # Subset to selected genes if specified
        if use_genes is not None:
            valid_genes = [gene for gene in use_genes if gene in adata.var_names]
            if len(valid_genes) == 0:
                raise ValueError("None of the specified genes were found in the dataset")
            adata = adata[:, valid_genes]
            
        # Create nearest neighbor graph
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            random_state=random_state,
            use_rep='X'  # Use gene expression, not PCA or other embedding
        )
        
        # Perform clustering based on specified method
        if method == 'kmeans':
            # K-means clustering
            from sklearn.cluster import KMeans
            
            # Extract expression matrix
            X = adata.X
            if sparse.issparse(X):
                X = X.toarray()
                
            # Run k-means
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10
            )
            clusters = kmeans.fit_predict(X)
            
            # Add results to original AnnData
            self.adata.obs['spatial_domains'] = pd.Categorical(
                [str(x) for x in clusters]
            )
            
        elif method == 'leiden':
            # Leiden community detection
            sc.tl.leiden(
                adata,
                resolution=resolution,
                random_state=random_state
            )
            
            # Add results to original AnnData
            self.adata.obs['spatial_domains'] = adata.obs['leiden']
            
        elif method == 'louvain':
            # Louvain community detection
            sc.tl.louvain(
                adata,
                resolution=resolution,
                random_state=random_state
            )
            
            # Add results to original AnnData
            self.adata.obs['spatial_domains'] = adata.obs['louvain']
            
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
            
        n_domains = len(self.adata.obs['spatial_domains'].cat.categories)
        print(f"Identified {n_domains} spatial domains")
        
    def calculate_gene_spatial_autocorrelation(self, gene: str) -> Tuple[float, float, float]:
        """
        Calculate spatial autocorrelation for a single gene.
        
        This function computes Moran's I statistic for a specific gene, including
        the p-value and z-score to assess statistical significance.
        
        Args:
            gene: Name of the gene to analyze
            
        Returns:
            Tuple containing (Moran's I, p-value, z-score)
            
        Raises:
            ValueError: If the gene is not found in the dataset
            
        Note:
            This is a convenience function for analyzing individual genes.
            For batch processing, use calculate_moran_i().
        """
        # Check if gene exists
        if gene not in self.adata.var_names:
            raise ValueError(f"Gene '{gene}' not found in dataset")
            
        # Calculate Moran's I for the single gene
        results = self.calculate_moran_i(genes=[gene])
        
        # Extract results
        morans_i = results.loc[0, 'morans_i']
        p_value = results.loc[0, 'p_value']
        z_score = results.loc[0, 'z_score']
        
        return morans_i, p_value, z_score
    
    def find_spatially_variable_genes(self, n_top_genes: int = 100) -> pd.DataFrame:
        """
        Identify genes with significant spatial variability.
        
        This function calculates Moran's I for all genes and returns those
        with the strongest spatial patterns.
        
        Args:
            n_top_genes: Number of top genes to return
            
        Returns:
            DataFrame with spatially variable genes and their statistics
            
        Note:
            This is a wrapper around calculate_moran_i() that returns the top
            spatially variable genes. It is computationally intensive for
            datasets with many genes.
        """
        print(f"Finding top {n_top_genes} spatially variable genes")
        
        # Calculate Moran's I for all genes
        moran_results = self.calculate_moran_i()
        
        # Get top genes by Moran's I value
        top_genes = moran_results.head(n_top_genes)
        
        return top_genes
