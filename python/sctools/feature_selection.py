#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FeatureSelection: Feature Selection for Single-Cell RNA-seq Data

This module provides the FeatureSelection class for identifying highly variable genes
and other feature selection methods for single-cell RNA-seq data. It helps reduce
dimensionality and focus analysis on biologically relevant genes.

Feature selection is an important step to eliminate genes that do not contribute
meaningful signal and to reduce computational requirements for downstream analysis.

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
from typing import Union, Optional, Tuple, List, Dict
import warnings


class FeatureSelection:
    """
    Feature selection methods for single-cell data.
    
    This class provides methods for identifying highly variable genes and other 
    feature selection techniques to reduce dimensionality and focus on 
    biologically relevant signals.
    
    Attributes:
        adata (AnnData): AnnData object containing the single-cell data.
    """
    
    def __init__(self, adata: ad.AnnData):
        """
        Initialize with AnnData object containing normalized data.
        
        Args:
            adata (AnnData): AnnData object with normalized gene expression data.
                            This should be normalized data, typically output from 
                            the Normalization class.
        """
        self.adata = adata
        
    def find_highly_variable_genes(self, 
                                 method: str = 'seurat',
                                 n_top_genes: int = 2000,
                                 min_mean: float = 0.0125,
                                 max_mean: float = 3,
                                 min_disp: float = 0.5,
                                 span: float = 0.3,
                                 batch_key: Optional[str] = None,
                                 subset: bool = False,
                                 inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Identify highly variable genes using various methods.
        
        This function identifies genes with high cell-to-cell variation,
        which are often genes of biological interest.
        
        Args:
            method: Method to use ('seurat', 'cell_ranger', 'seurat_v3', 'dispersion')
            n_top_genes: Number of highly variable genes to keep
            min_mean: Minimum mean expression of genes
            max_mean: Maximum mean expression of genes
            min_disp: Minimum dispersion
            span: Span parameter for trend fitting
            batch_key: If not None, highly-variable genes are selected within batches
            subset: Whether to subset the AnnData object to highly variable genes
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns AnnData with highly-variable genes
            
        Note:
            Results are stored in adata.var columns:
            - highly_variable: boolean indicating highly-variable genes
            - means: mean expression by gene
            - dispersions: dispersion by gene
            - dispersions_norm: normalized dispersion by gene
        """
        print(f"Finding highly variable genes using {method} method")
        
        # Work with either the original object or a copy
        adata = self.adata if inplace else self.adata.copy()
        
        # Find highly-variable genes using the specified method
        if method == 'seurat':
            # Original Seurat method (Macosko et al., 2015)
            sc.pp.highly_variable_genes(
                adata,
                flavor='seurat',
                n_top_genes=n_top_genes,
                min_mean=min_mean,
                max_mean=max_mean,
                min_disp=min_disp,
                batch_key=batch_key
            )
        elif method == 'cell_ranger':
            # Cell Ranger method (Zheng et al., 2017)
            sc.pp.highly_variable_genes(
                adata,
                flavor='cell_ranger',
                n_top_genes=n_top_genes,
                min_mean=min_mean,
                max_mean=max_mean,
                min_disp=min_disp,
                batch_key=batch_key
            )
        elif method == 'seurat_v3':
            # Seurat v3 method using variance stabilizing transformation
            sc.pp.highly_variable_genes(
                adata,
                flavor='seurat_v3',
                n_top_genes=n_top_genes,
                batch_key=batch_key
            )
        elif method == 'dispersion':
            # Simple dispersion-based method
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                min_mean=min_mean,
                max_mean=max_mean,
                min_disp=min_disp,
                batch_key=batch_key
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        # Subset to highly variable genes if requested
        if subset:
            adata = adata[:, adata.var.highly_variable]
            print(f"Subset to {adata.shape[1]} highly variable genes")
            
        # Update the object
        if inplace:
            self.adata = adata
        else:
            return adata
            
    def plot_highly_variable_genes(self, 
                                save_path: Optional[str] = None,
                                figsize: Tuple[float, float] = (8, 6),
                                return_fig: bool = False) -> Optional[plt.Figure]:
        """
        Plot highly variable genes.
        
        This function creates a scatter plot of mean expression vs. dispersion,
        with highly variable genes highlighted.
        
        Args:
            save_path: Path to save the figure (None displays the plot instead)
            figsize: Figure size
            return_fig: If True, return the figure object
            
        Returns:
            If return_fig is True, returns the matplotlib figure object
            
        Raises:
            ValueError: If highly variable genes haven't been identified yet
        """
        if 'highly_variable' not in self.adata.var.columns:
            raise ValueError("No highly variable genes found. Please run find_highly_variable_genes first.")
            
        print("Plotting highly variable genes")
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get the data
        mean = self.adata.var['means']
        disp = self.adata.var['dispersions_norm']
        hv = self.adata.var['highly_variable']
        
        # Plot non-variable genes (in gray)
        ax.scatter(mean[~hv], disp[~hv], s=5, alpha=0.2, label='Not Variable')
        
        # Plot highly variable genes (in red)
        ax.scatter(mean[hv], disp[hv], s=5, color='red', alpha=0.5, label='Highly Variable')
        
        # Add labels and legend
        ax.set_xlabel('Mean Expression')
        ax.set_ylabel('Normalized Dispersion')
        ax.set_xscale('log')  # Log scale for mean expression
        ax.legend()
        ax.set_title(f'Highly Variable Genes (n={sum(hv)})')
        
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
        
    def filter_genes_by_expression(self,
                                  min_cells: int = 5,
                                  min_counts: int = 5,
                                  inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Filter genes based on their expression level.
        
        This function removes genes that are expressed in too few cells or have
        too few counts, which can be noise or unreliable measurements.
        
        Args:
            min_cells: Minimum number of cells in which a gene must be detected
            min_counts: Minimum number of UMI counts required per gene
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns the filtered AnnData object
        """
        print(f"Filtering genes by expression (min_cells={min_cells}, min_counts={min_counts})")
        
        # Work with either the original object or a copy
        adata = self.adata if inplace else self.adata.copy()
        
        # Get initial dimensions
        n_genes_before = adata.shape[1]
        
        # Filter genes by number of cells expressing them
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        # Filter genes by total counts
        if min_counts > 0:
            # Calculate total counts per gene
            gene_counts = np.array(adata.X.sum(axis=0)).flatten()
            # Keep genes with at least min_counts
            adata = adata[:, gene_counts >= min_counts]
        
        # Calculate how many genes were removed
        n_genes_after = adata.shape[1]
        n_genes_removed = n_genes_before - n_genes_after
        print(f"Removed {n_genes_removed} genes ({n_genes_removed/n_genes_before:.1%} of total)")
        
        # Update the object
        if inplace:
            self.adata = adata
        else:
            return adata
            
    def select_marker_genes(self, 
                          markers: Dict[str, List[str]],
                          include_highly_variable: bool = True,
                          inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Select marker genes from predefined lists.
        
        This function selects genes from predefined marker lists and optionally
        includes highly variable genes, creating a combined feature set.
        
        Args:
            markers: Dictionary mapping categories to gene lists
            include_highly_variable: Whether to include highly variable genes
            inplace: Whether to modify self.adata or return a new object
            
        Returns:
            If inplace=False, returns AnnData with selected marker genes
            
        Raises:
            ValueError: If highly_variable is True but no highly variable genes were found
        """
        print("Selecting marker genes")
        
        # Work with either the original object or a copy
        adata = self.adata if inplace else self.adata.copy()
        
        # Get all marker genes
        all_markers = []
        for category, gene_list in markers.items():
            # Find which marker genes are in the data
            present_genes = [gene for gene in gene_list if gene in adata.var_names]
            print(f"Category {category}: {len(present_genes)}/{len(gene_list)} genes found")
            all_markers.extend(present_genes)
            
        # Remove duplicates
        all_markers = list(set(all_markers))
        
        # Include highly variable genes if requested
        if include_highly_variable:
            if 'highly_variable' not in adata.var.columns:
                raise ValueError("No highly variable genes found. Please run find_highly_variable_genes first.")
            
            hvg = adata.var_names[adata.var.highly_variable].tolist()
            all_genes = list(set(all_markers + hvg))
            print(f"Selected {len(all_markers)} marker genes and {len(hvg)} highly variable genes "
                 f"({len(all_genes)} unique genes total)")
        else:
            all_genes = all_markers
            print(f"Selected {len(all_genes)} marker genes")
            
        # Subset to selected genes
        adata = adata[:, all_genes]
        
        # Add marker information to var
        for category, gene_list in markers.items():
            # Create a boolean column indicating which genes are markers for this category
            adata.var[f'marker_{category}'] = [g in gene_list for g in adata.var_names]
            
        # Update the object
        if inplace:
            self.adata = adata
        else:
            return adata
