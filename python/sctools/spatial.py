"""
Spatial Transcriptomics Analysis Module

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
    
    The SpatialAnalysis class is typically used after quality control, normalization, and
    other preprocessing steps:
    
    - Upstream dependencies:
      * SingleCellQC for quality control and filtering
      * Normalization for data normalization
      * DimensionalityReduction for embedding information (optional)
    
    - Downstream applications:
      * Spatial domain identification with clustering
      * Cellular neighborhood analysis
      * Spatial differential expression
      * Tissue architecture analysis
    
    Attributes:
        adata (AnnData): AnnData object containing the spatial transcriptomics data.
        coordinates (np.ndarray): Spatial coordinates of spots or cells.
        spatial_key (str): Key in adata.obsm where spatial coordinates are stored.
        x_coord (str): Name of x-coordinate in obs if not using obsm[spatial_key].
        y_coord (str): Name of y-coordinate in obs if not using obsm[spatial_key].
    
    Examples:
        >>> # After QC and normalization
        >>> from sctools.spatial import SpatialAnalysis
        >>> spatial = SpatialAnalysis(adata, spatial_key='spatial')
        >>> 
        >>> # Visualize gene expression in space
        >>> spatial.plot_spatial_gene_expression(['GAPDH', 'CD3E'])
        >>> 
        >>> # Create spatial bins
        >>> binned_adata = spatial.create_spatial_grid(bin_size=100)
        >>> 
        >>> # Calculate spatial autocorrelation
        >>> moran_results = spatial.calculate_moran_i(genes=['GAPDH', 'CD3E'])
    """
    
    def __init__(self, adata: ad.AnnData, 
                 spatial_key: str = 'spatial',
                 x_coord: str = 'x', 
                 y_coord: str = 'y'):
        """
        Initialize with AnnData object and spatial coordinates.
        
        Parameters:
            adata (AnnData): AnnData object with spatial information.
            spatial_key (str): Key in adata.obsm where spatial coordinates are stored.
            x_coord (str): Name of x-coordinate in obs if not using obsm[spatial_key].
            y_coord (str): Name of y-coordinate in obs if not using obsm[spatial_key].
            
        Raises:
            ValueError: If spatial coordinates cannot be found.
            
        Examples:
            >>> # Initialize with 10X Visium data (coordinates in obsm['spatial'])
            >>> spatial = SpatialAnalysis(adata, spatial_key='spatial')
            >>> 
            >>> # Initialize with coordinates in obs columns
            >>> spatial = SpatialAnalysis(adata, x_coord='x_position', y_coord='y_position')
        
        Notes:
            - For 10X Visium data, coordinates are typically in adata.obsm['spatial']
            - For other technologies, coordinates might be in obs columns or other obsm keys
            - The class tries to infer coordinates from either obsm[spatial_key] or obs[x/y_coord]
        """
        self.adata = adata
        self.spatial_key = spatial_key
        self.x_coord = x_coord
        self.y_coord = y_coord
        
        # Extract spatial coordinates
        if spatial_key in adata.obsm:
            self.coordinates = adata.obsm[spatial_key]
            print(f"Using spatial coordinates from adata.obsm['{spatial_key}']")
        elif x_coord in adata.obs and y_coord in adata.obs:
            self.coordinates = np.vstack([adata.obs[x_coord].values, adata.obs[y_coord].values]).T
            print(f"Using spatial coordinates from adata.obs['{x_coord}'] and adata.obs['{y_coord}']")
        else:
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
        
        This method creates scatter plots of cells/spots in their spatial coordinates,
        colored by gene expression or metadata values.
        
        Parameters:
            genes (List[str] or str): Gene or list of genes to plot.
            ncols (int): Number of columns in the plot grid.
            figsize (Optional[Tuple[int, int]]): Figure size. If None, it's calculated based on the number of genes.
            cmap (str): Colormap for gene expression.
            size (float): Size of the points in the scatter plot.
            title_fontsize (int): Font size for subplot titles.
            show_colorbar (bool): Whether to show the colorbar.
            save_path (Optional[str]): Path to save the figure. If None, the figure is displayed instead.
            
        Returns:
            plt.Figure: The matplotlib figure object.
            
        Raises:
            ValueError: If none of the specified genes are found in the dataset.
            
        Examples:
            >>> spatial = SpatialAnalysis(adata)
            >>> 
            >>> # Plot a single gene
            >>> spatial.plot_spatial_gene_expression('GAPDH')
            >>> 
            >>> # Plot multiple genes
            >>> spatial.plot_spatial_gene_expression(
            ...     ['GAPDH', 'CD3E', 'CD8A'], 
            ...     ncols=3, 
            ...     cmap='plasma'
            ... )
            >>> 
            >>> # Plot QC metrics in space
            >>> spatial.plot_spatial_gene_expression(
            ...     ['n_genes_by_counts', 'total_counts', 'percent_mito'],
            ...     save_path='spatial_qc.png'
            ... )
        
        Notes:
            - Can plot both genes (from var_names) and metadata (from obs columns)
            - Useful for visualizing expression patterns, QC metrics, and cluster assignments
            - When plotting multiple genes, a grid of subplots is created
            - Returns the figure object for further customization
        """
        # Convert single gene to list
        if isinstance(genes, str):
            genes = [genes]
            
        # Make sure all genes exist in the dataset
        valid_genes = []
        for gene in genes:
            if gene in self.adata.var_names:
                valid_genes.append(gene)
            elif gene in self.adata.obs.columns:
                valid_genes.append(gene)
            else:
                print(f"Warning: '{gene}' not found in var_names or obs columns")
                
        if len(valid_genes) == 0:
            raise ValueError("None of the specified genes or metrics were found in the dataset")
            
        genes = valid_genes
            
        # Calculate grid dimensions
        n_genes = len(genes)
        nrows = (n_genes + ncols - 1) // ncols
        
        # Calculate figure size if not provided
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
            
        # Create figure
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        
        # Extract coordinates
        x, y = self.coordinates[:, 0], self.coordinates[:, 1]
        
        # Plot each gene
        for i, gene in enumerate(genes):
            row, col = i // ncols, i % ncols
            ax = axs[row, col]
            
            # Get gene expression or metadata
            if gene in self.adata.var_names:
                gene_expr = self.adata[:, gene].X
                if sparse.issparse(gene_expr):
                    gene_expr = gene_expr.toarray().flatten()
                values = gene_expr
            else:  # Assume it's a metadata column
                values = self.adata.obs[gene].values
                if values.dtype == 'object' or pd.api.types.is_categorical_dtype(values):
                    # For categorical data, convert to numeric representation
                    unique_values = sorted(set(values))
                    value_map = {val: i for i, val in enumerate(unique_values)}
                    values = np.array([value_map[v] for v in values])
                    
            # Create the scatter plot
            scatter = ax.scatter(x, y, c=values, cmap=cmap, s=size, alpha=0.8)
            
            # Add title and colorbar
            ax.set_title(gene, fontsize=title_fontsize)
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_aspect('equal')
            
            if show_colorbar:
                plt.colorbar(scatter, ax=ax, shrink=0.7)
                
        # Remove empty subplots
        for i in range(n_genes, nrows * ncols):
            row, col = i // ncols, i % ncols
            axs[row, col].axis('off')
            
        plt.tight_layout()
        
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
        
        This method divides the spatial domain into square bins of specified size
        and aggregates gene expression within each bin. This can be useful for 
        reducing noise, working at different spatial scales, or integrating with
        other spatial data.
        
        Parameters:
            bin_size (float): Size of the square bins.
            aggr_func (str): Aggregation function ('mean', 'sum', 'median', 'max').
            min_cells (int): Minimum number of cells required in a bin to keep it.
            genes (Optional[List[str]]): Specific genes to include. If None, all genes are used.
            
        Returns:
            AnnData: AnnData object containing the binned data.
            
        Examples:
            >>> spatial = SpatialAnalysis(adata)
            >>> 
            >>> # Create bins with default parameters (mean aggregation)
            >>> binned_adata = spatial.create_spatial_grid(bin_size=100)
            >>> 
            >>> # Create bins with sum aggregation and minimum cell requirement
            >>> binned_adata = spatial.create_spatial_grid(
            ...     bin_size=200,
            ...     aggr_func='sum',
            ...     min_cells=5
            ... )
            >>> 
            >>> # Bin only specific genes
            >>> marker_genes = ['CD3E', 'CD8A', 'FOXP3', 'MS4A1']
            >>> binned_adata = spatial.create_spatial_grid(
            ...     bin_size=100,
            ...     genes=marker_genes
            ... )
        
        Notes:
            - Binning reduces spatial resolution but can improve signal-to-noise ratio
            - Useful for matching resolution with other spatial datasets
            - The binned AnnData has one observation per bin with coordinates set to bin centers
            - The number of cells per bin is stored in adata.obs['n_cells']
            - Empty or sparse bins (below min_cells) are filtered out
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
        
        # Extract raw expression matrix
        if genes is None:
            genes = self.adata.var_names.tolist()
        else:
            genes = [gene for gene in genes if gene in self.adata.var_names]
            if len(genes) == 0:
                raise ValueError("None of the specified genes were found in the dataset")
                
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
            # Cells in this bin
            cells_in_bin = np.where(bin_indices == bin_idx)[0]
            bin_counts[i] = len(cells_in_bin)
            
            # Skip bins with too few cells
            if bin_counts[i] < min_cells:
                continue
                
            # Calculate aggregated expression
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
        
        # Set var names
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
        
        Moran's I measures the spatial autocorrelation of gene expression, identifying
        genes with significant spatial patterning. Values range from -1 (dispersed) to 
        1 (clustered), with 0 indicating random distribution.
        
        Parameters:
            genes (Optional[List[str]]): Specific genes to analyze. If None, all genes are analyzed.
            max_genes (int): Maximum number of genes to analyze (to prevent excessive computation).
            n_jobs (int): Number of parallel jobs for computation.
            
        Returns:
            pd.DataFrame: DataFrame with Moran's I statistics for each gene.
            
        Raises:
            ImportError: If pysal and joblib are not installed.
            ValueError: If no valid genes are found.
            
        Examples:
            >>> spatial = SpatialAnalysis(adata)
            >>> 
            >>> # Calculate Moran's I for all genes
            >>> moran_results = spatial.calculate_moran_i()
            >>> 
            >>> # Calculate for specific genes
            >>> marker_genes = ['CD3E', 'CD8A', 'FOXP3', 'MS4A1']
            >>> moran_results = spatial.calculate_moran_i(genes=marker_genes)
            >>> 
            >>> # Top spatially autocorrelated genes
            >>> top_genes = moran_results.sort_values('morans_i', ascending=False).head(10)
            >>> print(top_genes)
        
        Notes:
            - Requires pysal and joblib packages: pip install pysal joblib
            - High positive values indicate spatial clustering (e.g., tissue domains)
            - Values near zero indicate random spatial distribution
            - Negative values indicate spatial dispersion (rare in biology)
            - Computationally intensive for large datasets; max_genes limits computation
            - The p-value indicates statistical significance of the spatial pattern
            - Results are sorted by Moran's I value (highest to lowest)
        """
        try:
            from pysal.explore import esda
            from pysal.lib import weights
            from joblib import Parallel, delayed
        except ImportError:
            raise ImportError("Please install pysal and joblib: pip install pysal joblib")
        
        print("Calculating Moran's I spatial autocorrelation")
        
        # Extract coordinates
        coords = self.coordinates
        
        # Create spatial weights matrix (k-nearest neighbors)
        knn = 8  # Number of nearest neighbors
        w = weights.KNN(coords, k=knn)
        w.transform = 'r'  # Row-standardized weights
        
        # Filter genes to analyze
        if genes is None:
            genes = self.adata.var_names.tolist()
        else:
            genes = [gene for gene in genes if gene in self.adata.var_names]
            if len(genes) == 0:
                raise ValueError("None of the specified genes were found in the dataset")
                
        # Limit number of genes for computational efficiency
        if len(genes) > max_genes:
            print(f"Limiting analysis to {max_genes} randomly selected genes")
            genes = np.random.choice(genes, max_genes, replace=False)
            
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
                'morans_i': moran.I,
                'p_value': moran.p_sim,
                'z_score': moran.z_sim
            }
        
        # Calculate Moran's I for each gene in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_single_moran)(gene) for gene in genes
        )
        
        # Convert results to DataFrame
        moran_df = pd.DataFrame(results)
        
        # Sort by Moran's I
        moran_df = moran_df.sort_values('morans_i', ascending=False).reset_index(drop=True)
        
        print(f"Calculated Moran's I for {len(genes)} genes")
        
        return moran_df
    
    def analyze_negative_probes(self, 
                              prefix: str = 'Negative',
                              bin_size: Optional[float
