"""
Enhanced Visualization Module for Single-Cell Analysis

This module provides the EnhancedVisualization class for creating advanced and customizable
visualizations of single-cell data. It includes functionality for standard scatter plots,
fast plotting of large datasets using datashader, and various helper methods for
extracting and processing data for visualization.

The module is designed to create publication-quality figures while also handling
the challenges of visualizing millions of cells efficiently.

Key features:
    - Standard scatter plots with customizable aesthetics
    - Fast rendering of large datasets using datashader
    - Support for coloring by gene expression or metadata
    - Helper methods for extracting coordinates and color values
    - Support for saving figures in various formats

Upstream dependencies:
    - SingleCellQC for quality control and filtering
    - Normalization for data normalization
    - DimensionalityReduction for embeddings (PCA, UMAP, t-SNE)
    - SpatialAnalysis for spatial data visualization

Downstream applications:
    - Creating figures for publications
    - Interactive data exploration
    - Cluster visualization and annotation
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import plotly.express as px
import plotly.graph_objects as go
from typing import Union, Optional, Tuple, List, Dict
import warnings

# Check for datashader availability
try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.bundling import connect_edges
    from datashader.layout import circular_layout
    from colorcet import fire
    import bokeh.plotting as bpl
    from bokeh.models import HoverTool
    from bokeh.palettes import Spectral11
    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False
    warnings.warn("datashader not installed. For fast rendering of large scatter plots, install datashader: pip install datashader colorcet bokeh")


class EnhancedVisualization:
    """
    Enhanced visualization methods for single-cell data with support for large datasets.
    
    This class provides advanced visualization capabilities for single-cell data,
    including standard scatter plots and fast rendering of large datasets using datashader.
    It supports coloring by gene expression or metadata, and offers customizable aesthetics.
    
    The EnhancedVisualization class is typically used after dimensionality reduction
    to visualize data in reduced dimensional space:
    
    - Upstream dependencies:
      * SingleCellQC for quality control and filtering
      * Normalization for data normalization
      * DimensionalityReduction for producing embeddings (PCA, UMAP, t-SNE)
      * SpatialAnalysis for spatial data processing
    
    - Downstream applications:
      * Creating publication-quality figures
      * Interactive data exploration
      * Cluster visualization and annotation
      * Gene expression pattern visualization
    
    Attributes:
        adata (AnnData): AnnData object containing the single-cell data.
    
    Examples:
        >>> # After dimensionality reduction
        >>> from sctools.visualization import EnhancedVisualization
        >>> viz = EnhancedVisualization(adata)
        >>> 
        >>> # Basic UMAP plot colored by cluster
        >>> viz.scatter(x='X_umap-0', y='X_umap-1', color='leiden')
        >>> 
        >>> # Plot gene expression on UMAP
        >>> viz.scatter(x='X_umap-0', y='X_umap-1', color='CD3E')
        >>> 
        >>> # Fast plotting for large datasets
        >>> viz.scatter_fast(x='X_umap-0', y='X_umap-1', color='total_counts')
    """
    
    def __init__(self, adata: ad.AnnData):
        """
        Initialize with AnnData object.
        
        Parameters:
            adata (AnnData): AnnData object containing gene expression data.
            
        Examples:
            >>> viz = EnhancedVisualization(adata)
        """
        self.adata = adata
        
    def scatter_fast(self,
                   x: str,
                   y: str,
                   color: Optional[str] = None,
                   title: Optional[str] = None,
                   colormap: str = 'viridis',
                   size: float = 1.0,
                   alpha: float = 0.8,
                   width: int = 800,
                   height: int = 600,
                   save_path: Optional[str] = None) -> None:
        """
        Create a fast scatter plot for large datasets using datashader.
        
        This method uses datashader to render scatter plots efficiently for large
        datasets (millions of cells), which would be too slow with standard matplotlib
        or plotly. It creates a density-based visualization that works well even for
        extremely large datasets.
        
        Parameters:
            x (str): Variable for x-axis. Can be a gene name or a key in obsm (e.g., 'X_umap-0').
            y (str): Variable for y-axis. Can be a gene name or a key in obsm (e.g., 'X_umap-1').
            color (Optional[str]): Variable to color points by. Can be a gene name or a key in obs.
            title (Optional[str]): Plot title.
            colormap (str): Colormap for continuous variables.
            size (float): Size of the points.
            alpha (float): Opacity of the points.
            width (int): Width of the plot in pixels.
            height (int): Height of the plot in pixels.
            save_path (Optional[str]): Path to save the figure. If None, the figure is displayed.
            
        Raises:
            ImportError: If datashader is not installed.
            ValueError: If coordinates cannot be extracted.
            
        Examples:
            >>> viz = EnhancedVisualization(adata)
            >>> 
            >>> # Basic fast UMAP plot
            >>> viz.scatter_fast(x='X_umap-0', y='X_umap-1')
            >>> 
            >>> # Color by gene expression
            >>> viz.scatter_fast(
            ...     x='X_umap-0', 
            ...     y='X_umap-1', 
            ...     color='CD3E',
            ...     colormap='plasma'
            ... )
            >>> 
            >>> # Color by metadata
            >>> viz.scatter_fast(
            ...     x='X_umap-0', 
            ...     y='X_umap-1', 
            ...     color='leiden',
            ...     width=1000, 
            ...     height=800
            ... )
            >>> 
            >>> # Save to HTML file
            >>> viz.scatter_fast(
            ...     x='X_umap-0', 
            ...     y='X_umap-1', 
            ...     color='total_counts',
            ...     save_path='umap_plot.html'
            ... )
        
        Notes:
            - Requires datashader and related packages (colorcet, bokeh)
            - Much faster than standard plotting for large datasets (>10,000 cells)
            - Best for interactive exploration of large datasets
            - Saves as interactive HTML files when save_path is provided
            - Falls back to standard scatter() if datashader is not available
        """
        if not HAS_DATASHADER:
            print("Warning: datashader not installed. Using conventional plotting method.")
            return self.scatter(x, y, color=color, save_path=save_path)
            
        print(f"Creating fast scatter plot for {self.adata.n_obs} cells")
        
        # Extract x and y coordinates
        try:
            x_coords, x_name = self._get_coords(x)
            y_coords, y_name = self._get_coords(y)
        except ValueError as e:
            print(f"Error extracting coordinates: {e}")
            return
            
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': x_coords,
            'y': y_coords
        })
        
        # Extract color values if specified
        if color is not None:
            try:
                color_vals, color_name = self._get_color_values(color)
                plot_df['color'] = color_vals
            except ValueError as e:
                print(f"Error extracting color values: {e}")
                color = None
            
        # Create canvas
        cvs = ds.Canvas(plot_width=width, plot_height=height)
        
        # Create aggregation
        if color is None:
            # Simple point count
            agg = cvs.points(plot_df, 'x', 'y')
            img = tf.shade(agg, cmap=fire)
        else:
            # Colored by variable
            if np.issubdtype(plot_df['color'].dtype, np.number):
                # Continuous variable
                agg = cvs.points(plot_df, 'x', 'y', ds.mean('color'))
                img = tf.shade(agg, cmap=colormap)
            else:
                # Categorical variable
                unique_cats = plot_df['color'].unique()
                if len(unique_cats) > 10:
                    print(f"Warning: Too many categories ({len(unique_cats)}). Using first 10.")
                    unique_cats = unique_cats[:10]
                    
                # Create a color key for categories
                color_key = dict(zip(unique_cats, Spectral11[:len(unique_cats)]))
                
                # Create a separate aggregation for each category
                aggc = cvs.points(plot_df, 'x', 'y', ds.count_cat('color'))
                img = tf.shade(aggc, color_key=color_key)
                
        # Add title
        title = title or f'{x_name} vs {y_name}'
        if color is not None:
            title += f' (colored by {color_name})'
            
        # Create Bokeh figure
        p = bpl.figure(width=width, height=height, tools=['pan', 'wheel_zoom', 'box_zoom', 'reset'],
                      x_axis_label=x_name, y_axis_label=y_name, title=title)
        
        # Add the image
        p.image_rgba(image=[img.data], x=np.min(x_coords), y=np.min(y_coords),
                    dw=np.max(x_coords) - np.min(x_coords),
                    dh=np.max(y_coords) - np.min(y_coords))
        
        # Show or save the plot
        if save_path:
            bpl.output_file(save_path)
            bpl.save(p)
            print(f"Saved figure to {save_path}")
        else:
            bpl.show(p)
            
    def _get_coords(self, key: str) -> Tuple[np.ndarray, str]:
        """
        Helper function to extract coordinates for a given key.
        
        This internal method extracts coordinates for an axis from the AnnData object,
        handling various input formats (embedding keys, gene names, obs columns).
        
        Parameters:
            key (str): Key to extract coordinates for.
                
        Returns:
            Tuple[np.ndarray, str]: Coordinates and cleaned key name.
            
        Raises:
            ValueError: If coordinates cannot be found for the key.
            
        Notes:
            This is an internal method called by scatter() and scatter_fast().
        """
        # Check if the key is for a dimension reduction
        if key.startswith('X_'):
            # Handle keys like 'X_umap-0', 'X_pca-1', etc.
            try:
                base_key, dim = key.split('-')
                dim = int(dim)
                if base_key in self.adata.obsm:
                    return self.adata.obsm[base_key][:, dim], key
            except:
                pass
                
            # Handle direct obsm keys
            if key in self.adata.obsm:
                return self.adata.obsm[key][:, 0], key
                
        # Check for standard embeddings
        if key == 'umap1' or key == 'umap_1':
            return self.adata.obsm['X_umap'][:, 0], 'UMAP1'
        elif key == 'umap2' or key == 'umap_2':
            return self.adata.obsm['X_umap'][:, 1], 'UMAP2'
        elif key == 'tsne1' or key == 'tsne_1':
            return self.adata.obsm['X_tsne'][:, 0], 'tSNE1'
        elif key == 'tsne2' or key == 'tsne_2':
            return self.adata.obsm['X_tsne'][:, 1], 'tSNE2'
        elif key == 'pca1' or key == 'pca_1':
            return self.adata.obsm['X_pca'][:, 0], 'PC1'
        elif key == 'pca2' or key == 'pca_2':
            return self.adata.obsm['X_pca'][:, 1], 'PC2'
            
        # Check for obs variables
        if key in self.adata.obs:
            if pd.api.types.is_numeric_dtype(self.adata.obs[key]):
                return self.adata.obs[key].values, key
                
        # Check for genes
        if key in self.adata.var_names:
            gene_expr = self.adata[:, key].X
            if sparse.issparse(gene_expr):
                gene_expr = gene_expr.toarray().flatten()
            return gene_expr, key
            
        raise ValueError(f"Could not find coordinates for key: {key}")
        
    def _get_color_values(self, key: str) -> Tuple[np.ndarray, str]:
        """
        Helper function to extract color values for a given key.
        
        This internal method extracts values for coloring points from the AnnData object,
        handling both gene expression and metadata.
        
        Parameters:
            key (str): Key to extract color values for.
                
        Returns:
            Tuple[np.ndarray, str]: Color values and cleaned key name.
            
        Raises:
            ValueError: If color values cannot be found for the key.
            
        Notes:
            This is an internal method called by scatter() and scatter_fast().
        """
        # Check for obs variables
        if key in self.adata.obs:
            return self.adata.obs[key].values, key
            
        # Check for genes
        if key in self.adata.var_names:
            gene_expr = self.adata[:, key].X
            if sparse.issparse(gene_expr):
                gene_expr = gene_expr.toarray().flatten()
            return gene_expr, key
            
        raise ValueError(f"Could not find color values for key: {key}")
    
    def scatter(self,
              x: str,
              y: str,
              color: Optional[str] = None,
              title: Optional[str] = None,
              figsize: Tuple[float, float] = (10, 8),
              cmap: str = 'viridis',
              size: float = 5.0,
              alpha: float = 0.7,
              use_raw: bool = False,
              save_path: Optional[str] = None,
              return_ax: bool = False,
              ax: Optional[plt.Axes] = None,
              **kwargs) -> Optional[plt.Axes]:
        """
        Create a customizable scatter plot.
        
        This method creates a standard scatter plot using matplotlib, with extensive
        customization options. It can plot gene expression or metadata on any pair of
        coordinates, with customizable aesthetics.
        
        Parameters:
            x (str): Variable for x-axis. Can be obsm key (e.g., 'X_umap-0'), obs key, or gene name.
            y (str): Variable for y-axis. Can be obsm key (e.g., 'X_umap-1'), obs key, or gene name.
            color (Optional[str]): Variable to color points by. Can be obs key or gene name.
            title (Optional[str]): Plot title.
            figsize (Tuple[float, float]): Figure size.
            cmap (str): Colormap for continuous variables.
            size (float): Size of the points.
            alpha (float): Opacity of the points.
            use_raw (bool): Whether to use raw data for gene expression.
            save_path (Optional[str]): Path to save the figure. If None, the figure is displayed.
            return_ax (bool): If True, return the axis object.
            ax (Optional[plt.Axes]): Existing axis to plot on.
            **kwargs (dict): Additional keyword arguments to pass to plt.scatter.
                
        Returns:
            Optional[plt.Axes]: If return_ax is True, returns the axis object.
            
        Examples:
            >>> viz = EnhancedVisualization(adata)
            >>> 
            >>> # Basic UMAP plot
            >>> viz.scatter(x='X_umap-0', y='X_umap-1')
            >>> 
            >>> # Color by gene expression
            >>> viz.scatter(
            ...     x='X_umap-0', 
            ...     y='X_umap-1', 
            ...     color='CD3E',
            ...     cmap='plasma',
            ...     size=10
            ... )
            >>> 
            >>> # Color by metadata and save
            >>> viz.scatter(
            ...     x='X_umap-0', 
            ...     y='X_umap-1', 
            ...     color='leiden',
            ...     save_path='umap_clusters.png'
            ... )
            >>> 
            >>> # Create a custom multi-panel figure
            >>> fig, axs = plt.subplots(1, 2, figsize=(15, 7))
            >>> viz.scatter(
            ...     x='X_umap-0', y='X_umap-1', color='CD3E',
            ...     ax=axs[0], return_ax=True, title='CD3E Expression'
            ... )
            >>> viz.scatter(
            ...     x='X_umap-0', y='X_umap-1', color='leiden',
            ...     ax=axs[1], return_ax=True, title='Clusters'
            ... )
            >>> plt.tight_layout()
            >>> plt.savefig('combined_figure.png')
        
        Notes:
            - More flexible than scanpy's pl.scatter() with additional customization options
            - Works well with gene expression or metadata for coloring
            - Can be integrated into larger figures via the ax parameter
            - For datasets >10,000 cells, consider using scatter_fast() instead
            - Returns the axis object for further customization when return_ax=True
        """
        print(f"Creating scatter plot: {x} vs {y}")
        
        # Extract x and y coordinates
        try:
            x_coords, x_name = self._get_coords(x)
            y_coords, y_name = self._get_coords(y)
        except ValueError as e:
            print(f"Error extracting coordinates: {e}")
            return None
            
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        # Plot without color
        if color is None:
            sc = ax.scatter(x_coords, y_coords, s=size, alpha=alpha, **kwargs)
        else:
            # Extract color values
            try:
                color_vals, color_name = self._get_color_values(color)
            except ValueError as e:
                print(f"Error extracting color values: {e}")
                sc = ax.scatter(x_coords, y_coords, s=size, alpha=alpha, **kwargs)
            else:
                # Check if categorical or continuous
                if pd.api.types.is_categorical_dtype(color_vals) or (hasattr(color_vals, 'dtype') and color_vals.dtype == 'object'):
                    # Categorical coloring
                    categories = pd.Categorical(color_vals).categories
                    if len(categories) > 20:
                        print(f"Warning: Too many categories ({len(categories)}). Consider using a different color variable.")
                        
                    # Create a scatter plot for each category
                    for cat in categories:
                        mask = color_vals == cat
                        ax.scatter(x_coords[mask], y_coords[mask], s=size, alpha=alpha, label=cat, **kwargs)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=color_name)
                else:
                    # Continuous coloring
                    sc = ax.scatter(x_coords, y_coords, c=color_vals, cmap=cmap, s=size, alpha=alpha, **kwargs)
                    plt.colorbar(sc, ax=ax, label=color_name, shrink=0.8)
                    
        # Set labels and title
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(title or f'{x_name} vs {y_name}')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
            
        # Return axis if requested
        if return_ax:
            return ax
            
        # Show the plot if not returning axis and not saving
        if not return_ax and not save_path:
            plt.show()
            
        # Close the figure if we created it and aren't returning the axis
        if ax is not None and not return_ax:
            plt.close()
