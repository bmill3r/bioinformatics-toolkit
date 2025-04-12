#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeneSetScoring: Gene Set Scoring module for single-cell RNA-seq data analysis.

This module provides functions and classes to evaluate the activity of gene sets
across single cells using methods including AUCell (Area Under the Curve). It integrates
with the existing single-cell analysis toolkit and works directly with AnnData objects.

The primary use cases include:
- Scoring known gene signatures (e.g., cell type markers, pathways)
- Detecting pathway or gene program activity in cell subpopulations
- Comparing gene set enrichment across different cell groups
- Identifying cells where specific biological processes are active

Key features:
- Support for adding gene sets from various sources
- Implementation of AUCell and Scanpy-based scoring methods
- Visualization of gene set activities
- Analysis of gene set enrichment across cell groups
- Clustering cells based on gene set activities

Upstream dependencies:
- SingleCellQC: For initial data quality control
- Normalization: Normalized data is required for accurate scoring
- DimensionalityReduction: UMAP/t-SNE embeddings for visualization

Downstream applications:
- EnhancedVisualization: For further customized visualizations
- Clustering analysis based on gene set activities
- Cell type annotation using marker gene sets
- Regulatory network inference

Author: Your Name
Date: Current Date
Version: 0.1.0
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from typing import Union, List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc
import warnings


class GeneSetScoring:
    """
    A class for gene set scoring in single-cell data, including AUCell-inspired implementation.
    
    This class provides methods for scoring gene set activity across single cells, helping
    to identify pathway activation, cell types, and other biological signals. The core
    functionality is based on the AUCell approach, which uses the area under the recovery
    curve to quantify gene set enrichment in individual cells.
    
    Attributes:
        adata (AnnData): AnnData object containing gene expression data.
        gene_sets (Dict): Dictionary storing gene sets and their properties.
    """
    
    def __init__(self, adata: ad.AnnData):
        """
        Initialize with AnnData object.
        
        Parameters:
            adata (ad.AnnData): AnnData object with gene expression data
        """
        self.adata = adata
        self.gene_sets = {}
        self._validate_data()
        
    def _validate_data(self):
        """Validate input data format."""
        if not isinstance(self.adata, ad.AnnData):
            raise TypeError("Data must be an AnnData object")
        
        # Check if data is normalized
        if 'log1p' not in self.adata.uns.get('normalization', {}):
            warnings.warn("Data does not appear to be log-normalized, which is recommended for gene set scoring")
    
    def add_gene_set(self, name: str, genes: List[str], 
                    description: Optional[str] = None) -> None:
        """
        Add a gene set to be scored.
        
        Parameters:
            name (str): Name of the gene set
            genes (List[str]): List of gene identifiers in the set
            description (Optional[str]): Optional description of the gene set
        """
        # Check which genes are present in the dataset
        genes_present = [gene for gene in genes if gene in self.adata.var_names]
        genes_missing = [gene for gene in genes if gene not in self.adata.var_names]
        
        if len(genes_present) == 0:
            raise ValueError(f"None of the genes in '{name}' were found in the dataset")
        
        if len(genes_missing) > 0:
            percent_missing = len(genes_missing) / len(genes) * 100
            warnings.warn(f"{len(genes_missing)} genes ({percent_missing:.1f}%) from '{name}' are not in the dataset")
        
        # Store the gene set
        self.gene_sets[name] = {
            'genes': genes_present,
            'original_size': len(genes),
            'description': description
        }
        
        print(f"Added gene set '{name}' with {len(genes_present)} genes (original size: {len(genes)})")
    
    def add_gene_sets_from_dict(self, gene_sets_dict: Dict[str, List[str]],
                              descriptions: Optional[Dict[str, str]] = None) -> None:
        """
        Add multiple gene sets from a dictionary.
        
        Parameters:
            gene_sets_dict (Dict[str, List[str]]): Dictionary mapping gene set names to lists of genes
            descriptions (Optional[Dict[str, str]]): Optional dictionary of descriptions keyed by gene set name
        """
        for name, genes in gene_sets_dict.items():
            description = None if descriptions is None else descriptions.get(name)
            self.add_gene_set(name, genes, description)
    
    def add_gene_sets_from_gmt(self, gmt_file: str, 
                             selected_gene_sets: Optional[List[str]] = None) -> None:
        """
        Add gene sets from a GMT (Gene Matrix Transposed) file.
        
        Parameters:
            gmt_file (str): Path to the GMT file
            selected_gene_sets (Optional[List[str]]): If provided, only load these gene sets from the file
        """
        gene_sets_dict = {}
        descriptions_dict = {}
        
        with open(gmt_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:  # Name, description, and at least one gene
                    continue
                
                name = parts[0]
                if selected_gene_sets is not None and name not in selected_gene_sets:
                    continue
                    
                description = parts[1]
                genes = parts[2:]
                
                gene_sets_dict[name] = genes
                descriptions_dict[name] = description
        
        if not gene_sets_dict:
            raise ValueError(f"No gene sets were loaded from {gmt_file}")
            
        self.add_gene_sets_from_dict(gene_sets_dict, descriptions_dict)
    
    def score_aucell(self, threshold: Optional[float] = None, 
                   n_bins: int = 24, normalize: bool = True,
                   gene_sets: Optional[List[str]] = None) -> None:
        """
        Score gene sets using the AUCell method.
        
        This method implements a simplified version of the AUCell algorithm, which
        quantifies gene set enrichment based on the area under the recovery curve.
        It ranks genes by expression in each cell and calculates the enrichment
        of the gene set among the top-ranked genes.
        
        Parameters:
            threshold (Optional[float]): Threshold for binarizing the ranks (default: top 5% of expressed genes)
            n_bins (int): Number of bins for the AUC calculation
            normalize (bool): Whether to normalize scores between 0 and 1
            gene_sets (Optional[List[str]]): List of gene set names to score. If None, score all added gene sets
        """
        if len(self.gene_sets) == 0:
            raise ValueError("No gene sets have been added")
        
        if gene_sets is None:
            gene_sets = list(self.gene_sets.keys())
        else:
            # Check that all gene sets exist
            missing = [gs for gs in gene_sets if gs not in self.gene_sets]
            if missing:
                raise ValueError(f"The following gene sets have not been added: {missing}")
        
        print(f"Calculating AUCell scores for {len(gene_sets)} gene sets...")
        
        # Extract expression data
        X = self.adata.X
        if sparse.issparse(X):
            X = X.toarray()
        
        # Calculate the default threshold if not provided
        if threshold is None:
            threshold = 0.05  # Top 5% of genes
        
        # Calculate ranks for each cell
        n_cells, n_genes = X.shape
        
        # Dictionary to store the results
        scores = {}
        
        # For each cell, rank genes and calculate AUC
        for gene_set_name in gene_sets:
            gene_set = self.gene_sets[gene_set_name]['genes']
            gene_indices = [i for i, gene in enumerate(self.adata.var_names) if gene in gene_set]
            
            # Calculate scores for each cell
            cell_scores = np.zeros(n_cells)
            
            # Optimized approach using NumPy operations
            # Rank genes in each cell (higher rank for higher expression)
            # We use argsort of argsort to get ranks efficiently
            for i in range(n_cells):
                cell_expr = X[i, :]
                # Skip cells with no expression
                if np.sum(cell_expr) == 0:
                    cell_scores[i] = 0
                    continue
                    
                # Get ranks (1-based, higher expression gets higher rank)
                ranks = n_genes - np.argsort(np.argsort(cell_expr))
                
                # Calculate ranks for genes in the gene set
                set_ranks = ranks[gene_indices]
                
                # Calculate the AUC as the sum of ranks divided by max possible sum
                threshold_rank = int(np.ceil(n_genes * (1 - threshold)))
                
                # Count features above threshold rank
                n_features_above_threshold = np.sum(set_ranks >= threshold_rank)
                
                # Calculate the AUC
                if len(gene_indices) > 0:
                    if normalize:
                        # Normalize by the length of the gene set
                        auc_score = n_features_above_threshold / len(gene_indices)
                    else:
                        # Raw count of genes above threshold
                        auc_score = n_features_above_threshold
                else:
                    auc_score = 0
                
                cell_scores[i] = auc_score
            
            # Store the scores
            scores[gene_set_name] = cell_scores
        
        # Add scores to AnnData object
        for gene_set_name, cell_scores in scores.items():
            score_name = f"AUCell_{gene_set_name}"
            self.adata.obs[score_name] = cell_scores
        
        print(f"Added AUCell scores to adata.obs with keys: {[f'AUCell_{gs}' for gs in gene_sets]}")
    
    def score_scanpy(self, gene_sets: Optional[List[str]] = None, 
                   ctrl_size: int = 50, score_name_prefix: str = "Score_") -> None:
        """
        Score gene sets using Scanpy's implementation.
        
        This method uses Scanpy's built-in score_genes function to score gene sets.
        It calculates the average expression of genes in the gene set, subtracted
        by the average expression of a control gene set with similar expression.
        
        Parameters:
            gene_sets (Optional[List[str]]): List of gene set names to score. If None, score all added gene sets
            ctrl_size (int): Number of control genes for the scanpy score_genes implementation 
            score_name_prefix (str): Prefix for the score names in adata.obs
        """
        if len(self.gene_sets) == 0:
            raise ValueError("No gene sets have been added")
        
        if gene_sets is None:
            gene_sets = list(self.gene_sets.keys())
        else:
            # Check that all gene sets exist
            missing = [gs for gs in gene_sets if gs not in self.gene_sets]
            if missing:
                raise ValueError(f"The following gene sets have not been added: {missing}")
        
        print(f"Calculating Scanpy scores for {len(gene_sets)} gene sets...")
        
        for gene_set_name in gene_sets:
            gene_list = self.gene_sets[gene_set_name]['genes']
            
            # Use scanpy's score_genes function
            score_name = f"{score_name_prefix}{gene_set_name}"
            sc.tl.score_genes(self.adata, gene_list, ctrl_size=ctrl_size, score_name=score_name)
        
        print(f"Added scores to adata.obs with keys: {[f'{score_name_prefix}{gs}' for gs in gene_sets]}")
    
    def plot_gene_set_scores(self, gene_set: str, score_type: str = "AUCell_",
                          groupby: Optional[str] = None, 
                          use_raw: bool = False,
                          cmap: str = 'viridis',
                          size: int = 20,
                          save_path: Optional[str] = None,
                          show: bool = True) -> Optional[plt.Figure]:
        """
        Plot gene set scores on a UMAP or t-SNE embedding.
        
        Parameters:
            gene_set (str): Name of the gene set to plot
            score_type (str): Prefix for the score column (default: "AUCell_")
            groupby (Optional[str]): Column name in adata.obs to group cells by
            use_raw (bool): Whether to use raw data
            cmap (str): Colormap for the plot
            size (int): Size of the points
            save_path (Optional[str]): Path to save the figure
            show (bool): Whether to display the figure
            
        Returns:
            Optional[plt.Figure]: The figure if show=False, otherwise None
        """
        # Check if the gene set has been scored
        score_key = f"{score_type}{gene_set}"
        if score_key not in self.adata.obs.columns:
            raise ValueError(f"Score for gene set '{gene_set}' not found. Run score_aucell or score_scanpy first.")
        
        # Create the plot
        sc.settings.set_figure_params(dpi=100)
        
        fig = plt.figure(figsize=(12, 5))
        
        if groupby is not None:
            # Left subplot: colored by groups
            ax1 = fig.add_subplot(121)
            sc.pl.embedding(self.adata, basis='umap', color=groupby, ax=ax1, show=False, size=size)
            ax1.set_title(f"Grouped by {groupby}")
            
            # Right subplot: colored by gene set score
            ax2 = fig.add_subplot(122)
            sc.pl.embedding(self.adata, basis='umap', color=score_key, ax=ax2, 
                          vmin=0, vmax='p99', cmap=cmap, show=False, size=size)
            ax2.set_title(f"{gene_set} Score")
        else:
            # Single plot with gene set score
            ax = fig.add_subplot(111)
            sc.pl.embedding(self.adata, basis='umap', color=score_key, ax=ax, 
                          vmin=0, vmax='p99', cmap=cmap, show=False, size=size)
            ax.set_title(f"{gene_set} Score")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            return fig
        
    def plot_gene_set_violin(self, gene_sets: List[str], 
                           groupby: str,
                           score_type: str = "AUCell_",
                           figsize: Tuple[int, int] = (12, 6),
                           save_path: Optional[str] = None,
                           show: bool = True) -> Optional[plt.Figure]:
        """
        Plot gene set scores as violin plots, grouped by a categorical variable.
        
        Parameters:
            gene_sets (List[str]): Names of the gene sets to plot
            groupby (str): Column name in adata.obs to group cells by
            score_type (str): Prefix for the score column (default: "AUCell_")
            figsize (Tuple[int, int]): Figure size
            save_path (Optional[str]): Path to save the figure
            show (bool): Whether to display the figure
            
        Returns:
            Optional[plt.Figure]: The figure if show=False, otherwise None
        """
        # Check if the gene sets have been scored
        score_keys = [f"{score_type}{gs}" for gs in gene_sets]
        missing_scores = [key for key in score_keys if key not in self.adata.obs.columns]
        if missing_scores:
            missing_gene_sets = [key.replace(score_type, "") for key in missing_scores]
            raise ValueError(f"Scores for gene sets {missing_gene_sets} not found. Run score_aucell or score_scanpy first.")
        
        if groupby not in self.adata.obs.columns:
            raise ValueError(f"Group variable '{groupby}' not found in adata.obs")
            
        # Create the plot
        fig, axes = plt.subplots(1, len(gene_sets), figsize=figsize, sharey=True)
        
        # Handle the case of a single gene set
        if len(gene_sets) == 1:
            axes = [axes]
        
        for i, (gene_set, score_key) in enumerate(zip(gene_sets, score_keys)):
            # Get the data
            df = pd.DataFrame({
                'score': self.adata.obs[score_key],
                groupby: self.adata.obs[groupby]
            })
            
            # Create the violin plot
            sns.violinplot(x=groupby, y='score', data=df, ax=axes[i], inner='box')
            axes[i].set_title(gene_set)
            axes[i].set_ylabel('AUCell Score' if i == 0 else '')
            axes[i].set_xlabel(groupby)
            axes[i].tick_params(axis='x', rotation=90)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            return fig
            
    def find_cells_with_high_scores(self, gene_set: str, 
                                  score_type: str = "AUCell_",
                                  threshold: Optional[float] = None,
                                  percentile: float = 95) -> np.ndarray:
        """
        Find cells with high gene set scores.
        
        Parameters:
            gene_set (str): Name of the gene set
            score_type (str): Prefix for the score column (default: "AUCell_")
            threshold (Optional[float]): Explicit threshold value. If None, use percentile
            percentile (float): Percentile to use as threshold (default: 95)
            
        Returns:
            np.ndarray: Boolean mask of cells with high scores
        """
        score_key = f"{score_type}{gene_set}"
        if score_key not in self.adata.obs.columns:
            raise ValueError(f"Score for gene set '{gene_set}' not found. Run score_aucell or score_scanpy first.")
        
        scores = self.adata.obs[score_key].values
        
        # Determine threshold
        if threshold is None:
            threshold = np.percentile(scores, percentile)
            print(f"Using {percentile}th percentile as threshold: {threshold:.4f}")
        
        # Find cells above threshold
        high_score_mask = scores >= threshold
        num_cells = np.sum(high_score_mask)
        
        print(f"Found {num_cells} cells ({num_cells/len(scores):.1%}) with {gene_set} score >= {threshold:.4f}")
        
        return high_score_mask
    
    def get_top_scoring_cells(self, gene_set: str, 
                            score_type: str = "AUCell_",
                            n_cells: int = 100) -> pd.DataFrame:
        """
        Get the top scoring cells for a gene set.
        
        Parameters:
            gene_set (str): Name of the gene set
            score_type (str): Prefix for the score column (default: "AUCell_")
            n_cells (int): Number of top cells to return
            
        Returns:
            pd.DataFrame: DataFrame with top cells and their scores
        """
        score_key = f"{score_type}{gene_set}"
        if score_key not in self.adata.obs.columns:
            raise ValueError(f"Score for gene set '{gene_set}' not found. Run score_aucell or score_scanpy first.")
        
        # Get scores and cell names
        scores = self.adata.obs[score_key]
        
        # Sort and get top cells
        top_cells = scores.sort_values(ascending=False).head(n_cells)
        
        # Create DataFrame with results
        result = pd.DataFrame({
            'cell': top_cells.index,
            'score': top_cells.values
        })
        
        return result
    
    def get_gene_set_enrichment_by_group(self, gene_set: str, 
                                        groupby: str,
                                        score_type: str = "AUCell_") -> pd.DataFrame:
        """
        Calculate enrichment statistics for a gene set across different groups.
        
        Parameters:
            gene_set (str): Name of the gene set
            groupby (str): Column name in adata.obs to group cells by
            score_type (str): Prefix for the score column (default: "AUCell_")
            
        Returns:
            pd.DataFrame: DataFrame with enrichment statistics per group
        """
        score_key = f"{score_type}{gene_set}"
        if score_key not in self.adata.obs.columns:
            raise ValueError(f"Score for gene set '{gene_set}' not found. Run score_aucell or score_scanpy first.")
        
        if groupby not in self.adata.obs.columns:
            raise ValueError(f"Group variable '{groupby}' not found in adata.obs")
        
        # Get scores and group assignments
        scores = self.adata.obs[score_key]
        groups = self.adata.obs[groupby]
        
        # Calculate statistics per group
        result = []
        for group in groups.unique():
            group_scores = scores[groups == group]
            result.append({
                'group': group,
                'mean_score': group_scores.mean(),
                'median_score': group_scores.median(),
                'min_score': group_scores.min(),
                'max_score': group_scores.max(),
                'std_score': group_scores.std(),
                'n_cells': len(group_scores)
            })
        
        # Convert to DataFrame and sort by mean score
        result_df = pd.DataFrame(result).sort_values('mean_score', ascending=False)
        
        return result_df
    
    def score_multiple_methods(self, gene_sets: Optional[List[str]] = None) -> None:
        """
        Score gene sets using both AUCell and Scanpy's implementation.
        
        Parameters:
            gene_sets (Optional[List[str]]): List of gene sets to score. If None, score all added gene sets
        """
        self.score_aucell(gene_sets=gene_sets)
        self.score_scanpy(gene_sets=gene_sets)
        
    def compare_scoring_methods(self, gene_set: str, 
                              figsize: Tuple[int, int] = (10, 4),
                              save_path: Optional[str] = None,
                              show: bool = True) -> Optional[plt.Figure]:
        """
        Compare AUCell and Scanpy scoring methods for a gene set.
        
        Parameters:
            gene_set (str): Name of the gene set to compare
            figsize (Tuple[int, int]): Figure size
            save_path (Optional[str]): Path to save the figure
            show (bool): Whether to display the figure
            
        Returns:
            Optional[plt.Figure]: The figure if show=False, otherwise None
        """
        aucell_key = f"AUCell_{gene_set}"
        scanpy_key = f"Score_{gene_set}"
        
        if aucell_key not in self.adata.obs.columns:
            raise ValueError(f"AUCell score for '{gene_set}' not found. Run score_aucell first.")
            
        if scanpy_key not in self.adata.obs.columns:
            raise ValueError(f"Scanpy score for '{gene_set}' not found. Run score_scanpy first.")
        
        # Get scores
        aucell_scores = self.adata.obs[aucell_key]
        scanpy_scores = self.adata.obs[scanpy_key]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot comparing scores
        ax1.scatter(aucell_scores, scanpy_scores, alpha=0.5, s=10)
        ax1.set_xlabel(f"AUCell Score: {gene_set}")
        ax1.set_ylabel(f"Scanpy Score: {gene_set}")
        ax1.set_title("Score Comparison")
        
        # Add correlation coefficient
        corr = np.corrcoef(aucell_scores, scanpy_scores)[0, 1]
        ax1.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Distribution comparison
        ax2.hist(aucell_scores, bins=30, alpha=0.5, label="AUCell")
        ax2.hist(scanpy_scores, bins=30, alpha=0.5, label="Scanpy")
        ax2.set_xlabel("Score")
        ax2.set_ylabel("Number of Cells")
        ax2.set_title("Score Distributions")
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            return fig
    
    def cluster_by_gene_sets(self, gene_sets: List[str], 
                           n_clusters: int = 5,
                           score_type: str = "AUCell_",
                           cluster_key: str = "gene_set_clusters",
                           method: str = 'kmeans') -> None:
        """
        Cluster cells based on their gene set scores.
        
        Parameters:
            gene_sets (List[str]): List of gene sets to use for clustering
            n_clusters (int): Number of clusters to find
            score_type (str): Prefix for the score column (default: "AUCell_")
            cluster_key (str): Key to store cluster assignments in adata.obs
            method (str): Clustering method ('kmeans' or 'leiden')
        """
        from sklearn.cluster import KMeans
        
        # Check if the gene sets have been scored
        score_keys = [f"{score_type}{gs}" for gs in gene_sets]
        missing_scores = [key for key in score_keys if key not in self.adata.obs.columns]
        if missing_scores:
            missing_gene_sets = [key.replace(score_type, "") for key in missing_scores]
            raise ValueError(f"Scores for gene sets {missing_gene_sets} not found. Run score_aucell or score_scanpy first.")
        
        # Extract scores for each gene set
        score_matrix = self.adata.obs[score_keys].values
        
        if method == 'kmeans':
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(score_matrix)
            
            # Store cluster assignments
            self.adata.obs[cluster_key] = clusters.astype(str)
            
        elif method == 'leiden':
            # Create a temporary AnnData object with gene set scores as features
            temp_adata = ad.AnnData(X=score_matrix)
            
            # Compute neighborhood graph
            sc.pp.neighbors(temp_adata)
            
            # Run Leiden clustering
            sc.tl.leiden(temp_adata, resolution=1.0, random_state=42)
            
            # Store cluster assignments
            self.adata.obs[cluster_key] = temp_adata.obs['leiden']
            
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
            
        print(f"Clustered cells into {n_clusters} groups based on {len(gene_sets)} gene sets")
        print(f"Cluster assignments stored in adata.obs['{cluster_key}']")
