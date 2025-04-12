#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SingleCellQC: Quality Control for Single-Cell RNA-seq Data

This module provides the SingleCellQC class for comprehensive quality control
of single-cell RNA-seq data. It handles various input data formats and provides
tools for calculating QC metrics, visualizing QC results, and filtering cells
and genes based on quality thresholds.

The module is designed to be the starting point for single-cell data analysis,
performing the necessary pre-processing steps before downstream analysis.

Key features:
- Loading data from various sources (10X, CSV, H5AD, etc.)
- Calculating standard QC metrics (genes per cell, counts per cell, etc.)
- Identifying mitochondrial and ribosomal genes
- Visualizing QC metrics with customizable plots
- Automated threshold determination for QC filtering
- Flexible cell and gene filtering
- Comprehensive summary statistics

Upstream dependencies:
- S3Utils for loading data from AWS S3 (optional)

Downstream applications:
- Normalization modules for data normalization
- FeatureSelection for finding highly variable genes
- DimensionalityReduction for PCA, UMAP, etc.
- SpatialAnalysis for spatial transcriptomics data

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
import plotly.express as px
import plotly.graph_objects as go
from typing import Union, Optional, Tuple, List, Dict
import os
import gc
import warnings
warnings.filterwarnings('ignore')


class SingleCellQC:
    """
    A comprehensive QC pipeline for single-cell RNA-seq data.
    
    This class provides a complete workflow for quality control of single-cell RNA-seq data,
    from data loading to QC metric calculation, visualization, and filtering. It supports
    various input formats and provides extensive QC metrics and visualizations.
    
    Attributes:
        adata (AnnData): AnnData object containing the single-cell data.
        verbose (bool): Whether to print progress messages.
        qc_metrics (pd.DataFrame): DataFrame of calculated QC metrics.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the QC pipeline.
        
        Args:
            verbose (bool): Whether to print progress messages.
        """
        # Initialize with empty attributes
        self.verbose = verbose  # Control verbosity of log messages
        self.qc_metrics = None  # Will store QC metrics once calculated
        self.adata = None       # Will store AnnData object once loaded
        
    def log(self, message: str) -> None:
        """
        Print log messages if verbose mode is enabled.
        
        Args:
            message (str): The message to log.
        """
        # Only print if verbose mode is enabled
        if self.verbose:
            print(f"[SingleCellQC] {message}")
            
    def load_data(self, data_source: Union[str, pd.DataFrame, np.ndarray, sparse.spmatrix, ad.AnnData],
                 gene_names: Optional[List[str]] = None,
                 cell_names: Optional[List[str]] = None,
                 transpose: bool = False) -> ad.AnnData:
        """
        Load data from various sources and convert to AnnData object.
        
        This function can load data from different formats, including:
        - File paths (csv, tsv, mtx, h5ad, loom, zarr, parquet)
        - pandas DataFrames
        - numpy arrays or sparse matrices
        - AnnData objects
        
        Args:
            data_source: The data to load, can be a path, DataFrame, array, or AnnData
            gene_names: Gene names if loading from array/matrix (optional)
            cell_names: Cell names if loading from array/matrix (optional)
            transpose: Whether to transpose the matrix (cells as columns -> cells as rows)
        
        Returns:
            AnnData object containing the loaded data
            
        Raises:
            ValueError: If an unsupported file format is provided
            TypeError: If an unsupported data type is provided
        """
        self.log(f"Loading data from {type(data_source).__name__} source")
        
        # Handle different input types
        if isinstance(data_source, str):
            # Input is a file path
            file_ext = os.path.splitext(data_source)[1].lower()
            
            if file_ext in ['.csv', '.txt', '.tsv']:
                # CSV or TSV file
                sep = ',' if file_ext == '.csv' else '\t'
                df = pd.read_csv(data_source, sep=sep, index_col=0)
                if transpose:
                    df = df.T  # Transpose if cells are columns
                adata = ad.AnnData(df)
                
            elif file_ext == '.mtx':
                # 10X-style MTX format (assumes genes.tsv and barcodes.tsv in same directory)
                dir_path = os.path.dirname(data_source)
                adata = sc.read_10x_mtx(dir_path)
                
            elif file_ext == '.h5ad':
                # H5AD file (AnnData)
                adata = sc.read_h5ad(data_source)
                
            elif file_ext in ['.h5', '.hdf5']:
                # HDF5 file
                adata = sc.read_hdf(data_source)
                
            elif file_ext == '.loom':
                # Loom file
                adata = sc.read_loom(data_source)
                
            elif file_ext == '.zarr':
                # Zarr store
                adata = sc.read_zarr(data_source)
                
            elif file_ext == '.parquet':
                # Parquet file
                df = pd.read_parquet(data_source)
                if transpose:
                    df = df.T
                adata = ad.AnnData(df)
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        elif isinstance(data_source, pd.DataFrame):
            # Input is a pandas DataFrame
            if transpose:
                data_source = data_source.T
            adata = ad.AnnData(data_source)
            
        elif isinstance(data_source, np.ndarray) or sparse.issparse(data_source):
            # Input is a NumPy array or sparse matrix
            if transpose:
                data_source = data_source.T
                
            # Create default names if not provided
            if gene_names is None:
                gene_names = [f"gene_{i}" for i in range(data_source.shape[1])]
            if cell_names is None:
                cell_names = [f"cell_{i}" for i in range(data_source.shape[0])]
                
            adata = ad.AnnData(
                X=data_source,
                obs=pd.DataFrame(index=cell_names),
                var=pd.DataFrame(index=gene_names)
            )
            
        elif isinstance(data_source, ad.AnnData):
            # Input is already an AnnData object
            adata = data_source
            
        else:
            raise TypeError(f"Unsupported data type: {type(data_source).__name__}")
        
        self.log(f"Loaded data with shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
        self.adata = adata
        return adata
    
    def calculate_qc_metrics(self, 
                           min_genes: int = 200,
                           min_cells: int = 3,
                           percent_mito: Optional[Union[str, List[str]]] = 'MT-',
                           percent_ribo: Optional[Union[str, List[str]]] = 'RP[SL]',
                           inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Calculate quality control metrics for cells and genes.
        
        This function calculates various QC metrics including:
        - Number of genes expressed per cell
        - Total counts per cell
        - Percentage of counts from mitochondrial genes
        - Percentage of counts from ribosomal genes
        
        Args:
            min_genes: Minimum number of genes expressed for a cell to pass filter
            min_cells: Minimum number of cells a gene is expressed in to pass filter
            percent_mito: Pattern to identify mitochondrial genes, or list of genes
            percent_ribo: Pattern to identify ribosomal genes, or list of genes
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns the modified AnnData object
            
        Raises:
            ValueError: If no data has been loaded
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        self.log("Calculating QC metrics")
        
        # Work with a copy if not inplace
        adata = self.adata if inplace else self.adata.copy()
        
        # Calculate basic QC metrics using scanpy
        # This adds n_genes_by_counts, total_counts, and pct_counts_* columns to adata.obs
        sc.pp.calculate_qc_metrics(
            adata, 
            inplace=True, 
            percent_top=[10, 50, 100, 200, 500], 
            log1p=False
        )
        
        # Filter genes by minimum number of cells expressing them
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        # Calculate mitochondrial percentage if specified
        if percent_mito is not None:
            if isinstance(percent_mito, str):
                # Find mitochondrial genes using string pattern
                mito_genes = adata.var_names.str.startswith(percent_mito) if not percent_mito.startswith('^') else adata.var_names.str.contains(percent_mito)
            else:  # List of specific genes
                mito_genes = [gene in percent_mito for gene in adata.var_names]
                
            if sum(mito_genes) > 0:
                # Calculate percentage of mitochondrial gene counts
                adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1) * 100
                self.log(f"Found {sum(mito_genes)} mitochondrial genes")
            else:
                self.log("No mitochondrial genes found")
                # Set to zero if no mitochondrial genes found
                adata.obs['percent_mito'] = 0
        
        # Calculate ribosomal percentage if specified
        if percent_ribo is not None:
            if isinstance(percent_ribo, str):
                # Find ribosomal genes using string pattern
                ribo_genes = adata.var_names.str.startswith(percent_ribo) if not percent_ribo.startswith('^') else adata.var_names.str.contains(percent_ribo)
            else:  # List of specific genes
                ribo_genes = [gene in percent_ribo for gene in adata.var_names]
                
            if sum(ribo_genes) > 0:
                # Calculate percentage of ribosomal gene counts
                adata.obs['percent_ribo'] = np.sum(adata[:, ribo_genes].X, axis=1) / np.sum(adata.X, axis=1) * 100
                self.log(f"Found {sum(ribo_genes)} ribosomal genes")
            else:
                self.log("No ribosomal genes found")
                # Set to zero if no ribosomal genes found
                adata.obs['percent_ribo'] = 0
        
        # Store QC metrics for easier access
        qc_cols = ['n_genes_by_counts', 'total_counts']
        if percent_mito is not None:
            qc_cols.append('percent_mito')
        if percent_ribo is not None:
            qc_cols.append('percent_ribo')
            
        self.qc_metrics = adata.obs[qc_cols].copy()
        
        # Return or update in place
        if inplace:
            self.adata = adata
        else:
            return adata
    
    def get_qc_thresholds(self, 
                        n_mads: float = 5.0,
                        max_mito: Optional[float] = 20.0,
                        min_genes: Optional[int] = None,
                        max_genes: Optional[int] = None,
                        min_counts: Optional[int] = None,
                        max_counts: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate recommended QC thresholds based on data distribution.
        
        This function determines reasonable thresholds for QC filtering using
        median absolute deviations (MADs) or user-provided values.
        
        Args:
            n_mads: Number of median absolute deviations for outlier detection
            max_mito: Maximum percentage of mitochondrial reads allowed
            min_genes: Minimum genes per cell (overrides MAD calculation)
            max_genes: Maximum genes per cell (overrides MAD calculation)
            min_counts: Minimum total counts per cell (overrides MAD calculation)
            max_counts: Maximum total counts per cell (overrides MAD calculation)
            
        Returns:
            Dictionary with recommended thresholds for filtering
            
        Raises:
            ValueError: If QC metrics have not been calculated yet
        """
        if self.qc_metrics is None:
            raise ValueError("QC metrics not calculated. Please call calculate_qc_metrics() first.")
        
        self.log("Calculating recommended QC thresholds")
        
        thresholds = {}
        
        # Helper function to calculate thresholds based on median absolute deviation
        def mad_threshold(values, n_mads):
            """Calculate threshold based on median absolute deviation"""
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            lower = median - n_mads * mad
            upper = median + n_mads * mad
            return max(0, lower), upper
        
        # Calculate thresholds for number of genes
        if min_genes is None or max_genes is None:
            # Use MAD-based calculation if thresholds not explicitly provided
            lower, upper = mad_threshold(self.qc_metrics['n_genes_by_counts'], n_mads)
            thresholds['min_genes'] = min_genes if min_genes is not None else int(lower)
            thresholds['max_genes'] = max_genes if max_genes is not None else int(upper)
        else:
            # Use user-provided thresholds
            thresholds['min_genes'] = min_genes
            thresholds['max_genes'] = max_genes
            
        # Calculate thresholds for total counts
        if min_counts is None or max_counts is None:
            # Use MAD-based calculation if thresholds not explicitly provided
            lower, upper = mad_threshold(self.qc_metrics['total_counts'], n_mads)
            thresholds['min_counts'] = min_counts if min_counts is not None else int(lower)
            thresholds['max_counts'] = max_counts if max_counts is not None else int(upper)
        else:
            # Use user-provided thresholds
            thresholds['min_counts'] = min_counts
            thresholds['max_counts'] = max_counts
            
        # Mitochondrial percentage threshold
        if max_mito is not None and 'percent_mito' in self.qc_metrics.columns:
            # Use user-provided max_mito if given
            thresholds['max_mito'] = max_mito
        elif 'percent_mito' in self.qc_metrics.columns:
            # Calculate based on distribution
            # Use stricter threshold for mito (3.0 MADs instead of n_mads)
            _, upper = mad_threshold(self.qc_metrics['percent_mito'], n_mads=3.0)
            # Cap at 20% by default as an upper bound regardless of distribution
            thresholds['max_mito'] = min(upper, 20.0)
        
        return thresholds
    
    def filter_cells(self, 
                   min_genes: Optional[int] = None,
                   max_genes: Optional[int] = None,
                   min_counts: Optional[int] = None,
                   max_counts: Optional[int] = None,
                   max_mito: Optional[float] = None,
                   inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Filter cells based on QC metrics.
        
        This function removes cells that don't meet quality thresholds for:
        - Number of genes expressed
        - Total UMI counts
        - Mitochondrial content percentage
        
        Args:
            min_genes: Minimum number of genes per cell
            max_genes: Maximum number of genes per cell
            min_counts: Minimum total counts per cell
            max_counts: Maximum total counts per cell
            max_mito: Maximum percentage of mitochondrial reads
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns the filtered AnnData object
            
        Raises:
            ValueError: If no data has been loaded
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        # If no thresholds are provided, calculate recommended ones
        if all(x is None for x in [min_genes, max_genes, min_counts, max_counts, max_mito]):
            thresholds = self.get_qc_thresholds()
            min_genes = thresholds.get('min_genes')
            max_genes = thresholds.get('max_genes')
            min_counts = thresholds.get('min_counts')
            max_counts = thresholds.get('max_counts')
            max_mito = thresholds.get('max_mito')
        
        # Work with a copy if not inplace
        adata = self.adata if inplace else self.adata.copy()
        orig_cells = adata.shape[0]
        
        self.log("Filtering cells based on QC metrics")
        
        # Filter by number of genes
        if min_genes is not None:
            # Use scanpy's filter_cells function to filter by min_genes
            sc.pp.filter_cells(adata, min_genes=min_genes)
            self.log(f"Filtered cells with fewer than {min_genes} genes")
            
        if max_genes is not None:
            # Manual filtering for max_genes
            adata = adata[adata.obs['n_genes_by_counts'] <= max_genes]
            self.log(f"Filtered cells with more than {max_genes} genes")
            
        # Filter by total counts
        if min_counts is not None:
            # Use scanpy's filter_cells function to filter by min_counts
            sc.pp.filter_cells(adata, min_counts=min_counts)
            self.log(f"Filtered cells with fewer than {min_counts} total counts")
            
        if max_counts is not None:
            # Manual filtering for max_counts
            adata = adata[adata.obs['total_counts'] <= max_counts]
            self.log(f"Filtered cells with more than {max_counts} total counts")
            
        # Filter by mitochondrial percentage
        if max_mito is not None and 'percent_mito' in adata.obs.columns:
            # Filter cells with high mitochondrial content
            adata = adata[adata.obs['percent_mito'] <= max_mito]
            self.log(f"Filtered cells with more than {max_mito}% mitochondrial reads")
            
        # Calculate how many cells were removed
        filtered_cells = orig_cells - adata.shape[0]
        self.log(f"Removed {filtered_cells} cells ({filtered_cells/orig_cells:.1%} of total)")
        
        # Update the instance if inplace
        if inplace:
            self.adata = adata
        else:
            return adata
    
    def plot_qc_metrics(self, 
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (20, 12),
                      use_plotly: bool = False) -> None:
        """
        Create a comprehensive set of QC visualizations.
        
        This function generates plots to help analyze QC metrics and determine
        appropriate filtering thresholds.
        
        Args:
            save_path: Path to save the plot (None displays the plot instead)
            figsize: Figure size for matplotlib
            use_plotly: Whether to use Plotly for interactive visualizations
            
        Raises:
            ValueError: If no data has been loaded
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        self.log("Generating QC visualizations")
        
        # Choose between Plotly (interactive) or Matplotlib (static) visualization
        if use_plotly:
            self._plot_qc_metrics_plotly(save_path)
        else:
            self._plot_qc_metrics_matplotlib(save_path, figsize)
    
    def _plot_qc_metrics_matplotlib(self, 
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (20, 12)) -> None:
        """
        Create QC visualizations using matplotlib/seaborn.
        
        This internal method creates static visualizations of QC metrics.
        
        Args:
            save_path: Path to save the plot (None displays the plot instead)
            figsize: Figure size for matplotlib
        """
        # Create the figure with subplots arranged in a 2x3 grid
        fig, axs = plt.subplots(2, 3, figsize=figsize)
        
        # Flatten for easier indexing (convert 2D array of axes to 1D)
        axs = axs.flatten()
        
        # Plot 1: Histogram of genes per cell
        sns.histplot(self.adata.obs['n_genes_by_counts'], bins=100, kde=True, ax=axs[0])
        axs[0].set_title('Genes per Cell')
        axs[0].set_xlabel('Number of Genes')
        axs[0].set_ylabel('Number of Cells')
        
        # Plot 2: Histogram of counts per cell
        sns.histplot(self.adata.obs['total_counts'], bins=100, kde=True, ax=axs[1])
        axs[1].set_title('Counts per Cell')
        axs[1].set_xlabel('Number of Counts')
        axs[1].set_ylabel('Number of Cells')
        
        # Plot 3: Scatter plot of genes vs counts
        sns.scatterplot(
            x='total_counts', 
            y='n_genes_by_counts', 
            data=self.adata.obs, 
            alpha=0.7,  # Transparency
            s=10,       # Point size
            ax=axs[2]
        )
        axs[2].set_title('Genes vs Counts')
        axs[2].set_xlabel('Total Counts')
        axs[2].set_ylabel('Number of Genes')
        
        # Plot 4: Violin plot of genes by counts
        sns.violinplot(y=self.adata.obs['n_genes_by_counts'], ax=axs[3])
        axs[3].set_title('Genes per Cell (Violin)')
        axs[3].set_ylabel('Number of Genes')
        
        # Plot 5: Genes detected vs fraction of mito genes
        if 'percent_mito' in self.adata.obs.columns:
            # Check if mitochondrial percentage was calculated
            sns.scatterplot(
                x='n_genes_by_counts', 
                y='percent_mito', 
                data=self.adata.obs, 
                alpha=0.7, 
                s=10,
                ax=axs[4]
            )
            axs[4].set_title('Genes vs Mitochondrial Content')
            axs[4].set_xlabel('Number of Genes')
            axs[4].set_ylabel('Mitochondrial Content (%)')
        else:
            # Display a message if no mito data
            axs[4].set_title('Mitochondrial Content Not Available')
            axs[4].axis('off')
            
        # Plot 6: Distribution of gene detection frequency
        # Calculate how many cells express each gene
        cells_per_gene = np.sum(self.adata.X > 0, axis=0)
        if sparse.issparse(self.adata.X):
            # Convert to dense array if data is sparse
            cells_per_gene = cells_per_gene.A1
            
        sns.histplot(cells_per_gene, bins=100, kde=True, ax=axs[5])
        axs[5].set_title('Cells per Gene')
        axs[5].set_xlabel('Number of Cells')
        axs[5].set_ylabel('Number of Genes')
        
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.log(f"Saved QC plot to {save_path}")
        else:
            plt.show()
            
    def _plot_qc_metrics_plotly(self, save_path: Optional[str] = None) -> None:
        """
        Create interactive QC visualizations using Plotly.
        
        This internal method creates interactive visualizations of QC metrics.
        
        Args:
            save_path: Path to save the HTML plot (None displays the plot instead)
        """
        # Create subplot layout for interactive plot
        fig = go.Figure()
        
        # Histogram of genes per cell with KDE
        hist_genes = px.histogram(
            self.adata.obs, 
            x='n_genes_by_counts',
            nbins=100,
            title='Genes per Cell',
            labels={'n_genes_by_counts': 'Number of Genes'}
        )
        for trace in hist_genes.data:
            trace.name = 'Genes per Cell'
            trace.showlegend = True
            fig.add_trace(trace)
            
        # Histogram of counts per cell with KDE
        hist_counts = px.histogram(
            self.adata.obs, 
            x='total_counts',
            nbins=100,
            title='Counts per Cell',
            labels={'total_counts': 'Number of Counts'}
        )
        for trace in hist_counts.data:
            trace.name = 'Counts per Cell'
            trace.showlegend = True
            trace.visible = False  # Hide initially
            fig.add_trace(trace)
            
        # Scatter plot of genes vs counts
        scatter_genes_counts = px.scatter(
            self.adata.obs,
            x='total_counts',
            y='n_genes_by_counts',
            opacity=0.7,
            title='Genes vs Counts',
            labels={
                'total_counts': 'Total Counts',
                'n_genes_by_counts': 'Number of Genes'
            }
        )
        for trace in scatter_genes_counts.data:
            trace.name = 'Genes vs Counts'
            trace.showlegend = True
            trace.visible = False  # Hide initially
            fig.add_trace(trace)
            
        # Genes detected vs fraction of mito genes
        if 'percent_mito' in self.adata.obs.columns:
            scatter_genes_mito = px.scatter(
                self.adata.obs,
                x='n_genes_by_counts',
                y='percent_mito',
                opacity=0.7,
                title='Genes vs Mitochondrial Content',
                labels={
                    'n_genes_by_counts': 'Number of Genes',
                    'percent_mito': 'Mitochondrial Content (%)'
                }
            )
            for trace in scatter_genes_mito.data:
                trace.name = 'Genes vs Mito'
                trace.showlegend = True
                trace.visible = False  # Hide initially
                fig.add_trace(trace)
                
        # Distribution of gene detection frequency
        cells_per_gene = np.sum(self.adata.X > 0, axis=0)
        if sparse.issparse(self.adata.X):
            cells_per_gene = cells_per_gene.A1
            
        hist_cells_per_gene = px.histogram(
            x=cells_per_gene,
            nbins=100,
            title='Cells per Gene',
            labels={'x': 'Number of Cells', 'y': 'Number of Genes'}
        )
        for trace in hist_cells_per_gene.data:
            trace.name = 'Cells per Gene'
            trace.showlegend = True
            trace.visible = False  # Hide initially
            fig.add_trace(trace)
            
        # Add buttons for toggling between plots
        fig.update_layout(
            updatemenus=[
                {
                    'active': 0,
                    'buttons': [
                        {'label': 'Genes per Cell',
                         'method': 'update',
                         'args': [{'visible': [True, False, False, False, False]},
                                  {'title': 'Genes per Cell'}]},
                        {'label': 'Counts per Cell',
                         'method': 'update',
                         'args': [{'visible': [False, True, False, False, False]},
                                  {'title': 'Counts per Cell'}]},
                        {'label': 'Genes vs Counts',
                         'method': 'update',
                         'args': [{'visible': [False, False, True, False, False]},
                                  {'title': 'Genes vs Counts'}]},
                        {'label': 'Genes vs Mito',
                         'method': 'update',
                         'args': [{'visible': [False, False, False, True, False]},
                                  {'title': 'Genes vs Mitochondrial Content'}]},
                        {'label': 'Cells per Gene',
                         'method': 'update',
                         'args': [{'visible': [False, False, False, False, True]},
                                  {'title': 'Cells per Gene'}]}
                    ],
                    'direction': 'down',
                    'pad': {'r': 10, 't': 10},
                    'showactive': True,
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 1.15,
                    'yanchor': 'top'
                }
            ],
            height=600,
            width=900,
            title='Single-Cell RNA-seq QC Metrics'
        )
        
        # Save or show the interactive plot
        if save_path:
            fig.write_html(save_path)
            self.log(f"Saved interactive QC plot to {save_path}")
        else:
            fig.show()
    
    def compute_summary_statistics(self) -> pd.DataFrame:
        """
        Compute summary statistics for QC metrics.
        
        This function calculates various statistical measures for QC metrics,
        providing a comprehensive overview of data quality.
        
        Returns:
            DataFrame containing summary statistics
            
        Raises:
            ValueError: If no data has been loaded
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please call
        
        self.log("Computing summary statistics")
        
        # Metrics to summarize
        metrics = ['n_genes_by_counts', 'total_counts']
        if 'percent_mito' in self.adata.obs.columns:
            metrics.append('percent_mito')
        if 'percent_ribo' in self.adata.obs.columns:
            metrics.append('percent_ribo')
            
        # Calculate statistics
        stats = self.adata.obs[metrics].describe().T
        
        # Add median absolute deviation
        stats['mad'] = self.adata.obs[metrics].apply(lambda x: np.median(np.abs(x - np.median(x))))
        
        # Calculate sparsity (fraction of zeros in the matrix)
        sparsity = 1 - (np.count_nonzero(self.adata.X) / self.adata.X.size)
        stats.loc['sparsity', 'mean'] = sparsity
        
        # Add gene stats
        cells_per_gene = np.sum(self.adata.X > 0, axis=0)
        if sparse.issparse(self.adata.X):
            cells_per_gene = cells_per_gene.A1
        
        stats.loc['cells_per_gene', 'mean'] = np.mean(cells_per_gene)
        stats.loc['cells_per_gene', 'std'] = np.std(cells_per_gene)
        stats.loc['cells_per_gene', 'min'] = np.min(cells_per_gene)
        stats.loc['cells_per_gene', '25%'] = np.percentile(cells_per_gene, 25)
        stats.loc['cells_per_gene', '50%'] = np.median(cells_per_gene)
        stats.loc['cells_per_gene', '75%'] = np.percentile(cells_per_gene, 75)
        stats.loc['cells_per_gene', 'max'] = np.max(cells_per_gene)
        stats.loc['cells_per_gene', 'mad'] = np.median(np.abs(cells_per_gene - np.median(cells_per_gene)))
        
        # Add dataset dimensions
        stats.loc['dataset_dimensions', 'count'] = f"{self.adata.shape[0]} cells × {self.adata.shape[1]} genes"
        
        return stats
    
    def run_qc_pipeline(self, 
                      data_source,
                      min_genes: int = 200,
                      min_cells: int = 3,
                      max_mito: float = 20.0,
                      n_mads: float = 5.0,
                      save_path: Optional[str] = None,
                      plotly: bool = False) -> Tuple[ad.AnnData, pd.DataFrame]:
        """
        Run the full QC pipeline from data loading to visualization.
        
        This is a convenience function that chains together the main QC steps:
        1. Loading data
        2. Calculating QC metrics
        3. Getting threshold recommendations
        4. Filtering cells based on QC
        5. Creating QC visualizations
        6. Computing summary statistics
        
        Args:
            data_source: The data to load (same options as load_data)
            min_genes: Minimum number of genes for a cell to pass filter
            min_cells: Minimum number of cells a gene is expressed in to pass filter
            max_mito: Maximum percentage of mitochondrial reads
            n_mads: Number of median absolute deviations for outlier detection
            save_path: Path to save QC plots
            plotly: Whether to use Plotly for interactive visualizations
            
        Returns:
            Tuple containing:
            - Filtered AnnData object
            - DataFrame with summary statistics
        """
        self.log("Running full QC pipeline")
        
        # Load data
        self.load_data(data_source)
        
        # Calculate QC metrics
        self.calculate_qc_metrics(min_genes=min_genes, min_cells=min_cells)
        
        # Get recommended thresholds
        thresholds = self.get_qc_thresholds(n_mads=n_mads, max_mito=max_mito)
        
        # Filter cells
        self.filter_cells(
            min_genes=thresholds['min_genes'],
            max_genes=thresholds['max_genes'],
            min_counts=thresholds['min_counts'],
            max_counts=thresholds['max_counts'],
            max_mito=thresholds['max_mito']
        )
        
        # Create QC plots
        self.plot_qc_metrics(save_path=save_path, use_plotly=plotly)
        
        # Compute summary statistics
        stats = self.compute_summary_statistics()
        
        return self.adata, stats
