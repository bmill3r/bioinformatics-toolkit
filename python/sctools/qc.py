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
    Handles various input formats and provides extensive QC metrics and visualizations.
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize the QC pipeline with configuration options."""
        self.verbose = verbose
        self.qc_metrics = None
        self.adata = None
        
    def log(self, message: str) -> None:
        """Print log messages if verbose mode is enabled."""
        if self.verbose:
            print(f"[SingleCellQC] {message}")
            
    def load_data(self, data_source: Union[str, pd.DataFrame, np.ndarray, sparse.spmatrix, ad.AnnData],
                 gene_names: Optional[List[str]] = None,
                 cell_names: Optional[List[str]] = None,
                 transpose: bool = False) -> ad.AnnData:
        """
        Load data from various sources and convert to AnnData object.
        
        Parameters:
        -----------
        data_source : Union[str, pd.DataFrame, np.ndarray, sparse.spmatrix, ad.AnnData]
            Data source which can be a file path (csv, tsv, mtx, h5ad, loom, zarr),
            pandas DataFrame, numpy array, sparse matrix, or AnnData object.
        gene_names : Optional[List[str]]
            List of gene names if data_source is numpy array or sparse matrix.
        cell_names : Optional[List[str]]
            List of cell names if data_source is numpy array or sparse matrix.
        transpose : bool
            Whether to transpose the matrix (if cells are columns instead of rows).
            
        Returns:
        --------
        adata : AnnData
            AnnData object containing the loaded data.
        """
        self.log(f"Loading data from {type(data_source).__name__} source")
        
        # Handle different input types
        if isinstance(data_source, str):
            # File path
            file_ext = os.path.splitext(data_source)[1].lower()
            
            if file_ext in ['.csv', '.txt', '.tsv']:
                # CSV or TSV file
                sep = ',' if file_ext == '.csv' else '\t'
                df = pd.read_csv(data_source, sep=sep, index_col=0)
                if transpose:
                    df = df.T
                adata = ad.AnnData(df)
                
            elif file_ext == '.mtx':
                # 10X-style MTX format (assumes genes.tsv and barcodes.tsv in same directory)
                dir_path = os.path.dirname(data_source)
                adata = sc.read_10x_mtx(dir_path)
                
            elif file_ext == '.h5ad':
                # H5AD file
                adata = sc.read_h5ad(data_source)
                
            elif file_ext == '.loom':
                # Loom file
                adata = sc.read_loom(data_source)
                
            elif file_ext == '.zarr':
                # Zarr store
                adata = sc.read_zarr(data_source)
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        elif isinstance(data_source, pd.DataFrame):
            # Pandas DataFrame
            if transpose:
                data_source = data_source.T
            adata = ad.AnnData(data_source)
            
        elif isinstance(data_source, np.ndarray) or sparse.issparse(data_source):
            # NumPy array or sparse matrix
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
            # Already an AnnData object
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
        
        Parameters:
        -----------
        min_genes : int
            Minimum number of genes expressed for a cell to pass filter.
        min_cells : int
            Minimum number of cells a gene is expressed in to pass filter.
        percent_mito : Optional[Union[str, List[str]]]
            Prefix or regex pattern for mitochondrial genes, or list of specific genes.
            Set to None to skip mitochondrial calculation.
        percent_ribo : Optional[Union[str, List[str]]]
            Prefix or regex pattern for ribosomal genes, or list of specific genes.
            Set to None to skip ribosomal calculation.
        inplace : bool
            If True, add calculated metrics to self.adata, else return a new AnnData object.
            
        Returns:
        --------
        Optional[AnnData]
            If inplace is False, returns an AnnData object with QC metrics.
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        self.log("Calculating QC metrics")
        
        # Work with a copy if not inplace
        adata = self.adata if inplace else self.adata.copy()
        
        # Calculate basic metrics (n_genes, n_counts per cell)
        sc.pp.calculate_qc_metrics(
            adata, 
            inplace=True, 
            percent_top=[10, 50, 100, 200, 500], 
            log1p=False
        )
        
        # Filter genes by minimum number of cells
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        # Calculate mitochondrial percentage
        if percent_mito is not None:
            if isinstance(percent_mito, str):
                mito_genes = adata.var_names.str.startswith(percent_mito) if not percent_mito.startswith('^') else adata.var_names.str.contains(percent_mito)
            else:  # List of specific genes
                mito_genes = [gene in percent_mito for gene in adata.var_names]
                
            if sum(mito_genes) > 0:
                adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1) * 100
                self.log(f"Found {sum(mito_genes)} mitochondrial genes")
            else:
                self.log("No mitochondrial genes found")
                adata.obs['percent_mito'] = 0
        
        # Calculate ribosomal percentage
        if percent_ribo is not None:
            if isinstance(percent_ribo, str):
                ribo_genes = adata.var_names.str.startswith(percent_ribo) if not percent_ribo.startswith('^') else adata.var_names.str.contains(percent_ribo)
            else:  # List of specific genes
                ribo_genes = [gene in percent_ribo for gene in adata.var_names]
                
            if sum(ribo_genes) > 0:
                adata.obs['percent_ribo'] = np.sum(adata[:, ribo_genes].X, axis=1) / np.sum(adata.X, axis=1) * 100
                self.log(f"Found {sum(ribo_genes)} ribosomal genes")
            else:
                self.log("No ribosomal genes found")
                adata.obs['percent_ribo'] = 0
        
        # Store QC metrics for easier access
        qc_cols = ['n_genes_by_counts', 'total_counts', 'percent_mito', 'percent_ribo'] 
        self.qc_metrics = adata.obs[qc_cols].copy()
        
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
        
        Parameters:
        -----------
        n_mads : float
            Number of median absolute deviations (MADs) for outlier detection.
        max_mito : Optional[float]
            Maximum percentage of mitochondrial reads allowed.
        min_genes : Optional[int]
            Minimum genes per cell (overrides MAD calculation if provided).
        max_genes : Optional[int]
            Maximum genes per cell (overrides MAD calculation if provided).
        min_counts : Optional[int]
            Minimum total counts per cell (overrides MAD calculation if provided).
        max_counts : Optional[int]
            Maximum total counts per cell (overrides MAD calculation if provided).
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with recommended thresholds for filtering.
        """
        if self.qc_metrics is None:
            raise ValueError("QC metrics not calculated. Please call calculate_qc_metrics() first.")
        
        self.log("Calculating recommended QC thresholds")
        
        thresholds = {}
        
        # Function to calculate MAD thresholds
        def mad_threshold(values, n_mads):
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            lower = median - n_mads * mad
            upper = median + n_mads * mad
            return max(0, lower), upper
        
        # Calculate thresholds for number of genes
        if min_genes is None or max_genes is None:
            lower, upper = mad_threshold(self.qc_metrics['n_genes_by_counts'], n_mads)
            thresholds['min_genes'] = min_genes if min_genes is not None else int(lower)
            thresholds['max_genes'] = max_genes if max_genes is not None else int(upper)
        else:
            thresholds['min_genes'] = min_genes
            thresholds['max_genes'] = max_genes
            
        # Calculate thresholds for total counts
        if min_counts is None or max_counts is None:
            lower, upper = mad_threshold(self.qc_metrics['total_counts'], n_mads)
            thresholds['min_counts'] = min_counts if min_counts is not None else int(lower)
            thresholds['max_counts'] = max_counts if max_counts is not None else int(upper)
        else:
            thresholds['min_counts'] = min_counts
            thresholds['max_counts'] = max_counts
            
        # Mitochondrial percentage threshold
        if max_mito is not None and 'percent_mito' in self.qc_metrics.columns:
            thresholds['max_mito'] = max_mito
        elif 'percent_mito' in self.qc_metrics.columns:
            # Calculate based on distribution
            _, upper = mad_threshold(self.qc_metrics['percent_mito'], n_mads=3.0)  # Stricter for mito
            thresholds['max_mito'] = min(upper, 20.0)  # Cap at 20% by default
        
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
        
        Parameters:
        -----------
        min_genes : Optional[int]
            Minimum number of genes per cell.
        max_genes : Optional[int]
            Maximum number of genes per cell.
        min_counts : Optional[int]
            Minimum total counts per cell.
        max_counts : Optional[int]
            Maximum total counts per cell.
        max_mito : Optional[float]
            Maximum percentage of mitochondrial reads.
        inplace : bool
            If True, modify self.adata, else return a filtered copy.
            
        Returns:
        --------
        Optional[AnnData]
            Filtered AnnData object if inplace is False.
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
            sc.pp.filter_cells(adata, min_genes=min_genes)
            self.log(f"Filtered cells with fewer than {min_genes} genes")
            
        if max_genes is not None:
            adata = adata[adata.obs['n_genes_by_counts'] <= max_genes]
            self.log(f"Filtered cells with more than {max_genes} genes")
            
        # Filter by total counts
        if min_counts is not None:
            sc.pp.filter_cells(adata, min_counts=min_counts)
            self.log(f"Filtered cells with fewer than {min_counts} total counts")
            
        if max_counts is not None:
            adata = adata[adata.obs['total_counts'] <= max_counts]
            self.log(f"Filtered cells with more than {max_counts} total counts")
            
        # Filter by mitochondrial percentage
        if max_mito is not None and 'percent_mito' in adata.obs.columns:
            adata = adata[adata.obs['percent_mito'] <= max_mito]
            self.log(f"Filtered cells with more than {max_mito}% mitochondrial reads")
            
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
        
        Parameters:
        -----------
        save_path : Optional[str]
            Path to save the plot. If None, the plot is displayed instead.
        figsize : Tuple[int, int]
            Figure size for matplotlib.
        use_plotly : bool
            Whether to use Plotly for interactive visualizations.
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        self.log("Generating QC visualizations")
        
        if use_plotly:
            self._plot_qc_metrics_plotly(save_path)
        else:
            self._plot_qc_metrics_matplotlib(save_path, figsize)
    
    def _plot_qc_metrics_matplotlib(self, 
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (20, 12)) -> None:
        """
        Create QC visualizations using matplotlib/seaborn.
        
        Parameters:
        -----------
        save_path : Optional[str]
            Path to save the plot. If None, the plot is displayed instead.
        figsize : Tuple[int, int]
            Figure size for matplotlib.
        """
        # Create the figure with subplots
        fig, axs = plt.subplots(2, 3, figsize=figsize)
        
        # Flatten for easier indexing
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
            alpha=0.7, 
            s=10,
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
            axs[4].set_title('Mitochondrial Content Not Available')
            axs[4].axis('off')
            
        # Plot 6: Distribution of gene detection frequency
        cells_per_gene = np.sum(self.adata.X > 0, axis=0)
        if sparse.issparse(self.adata.X):
            cells_per_gene = cells_per_gene.A1
            
        sns.histplot(cells_per_gene, bins=100, kde=True, ax=axs[5])
        axs[5].set_title('Cells per Gene')
        axs[5].set_xlabel('Number of Cells')
        axs[5].set_ylabel('Number of Genes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.log(f"Saved QC plot to {save_path}")
        else:
            plt.show()
            
    def _plot_qc_metrics_plotly(self, save_path: Optional[str] = None) -> None:
        """
        Create interactive QC visualizations using Plotly.
        
        Parameters:
        -----------
        save_path : Optional[str]
            Path to save the HTML plot. If None, the plot is displayed instead.
        """
        # Create subplot layout
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
            trace.visible = False
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
            trace.visible = False
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
                trace.visible = False
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
            trace.visible = False
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
        
        if save_path:
            fig.write_html(save_path)
            self.log(f"Saved interactive QC plot to {save_path}")
        else:
            fig.show()
    
    def compute_summary_statistics(self) -> pd.DataFrame:
        """
        Compute summary statistics for QC metrics.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing summary statistics.
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
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
        
        # Calculate sparsity
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
        
        Parameters:
        -----------
        data_source : Union[str, pd.DataFrame, np.ndarray, sparse.spmatrix, ad.AnnData]
            Data source for loading.
        min_genes : int
            Minimum number of genes for a cell to pass filter.
        min_cells : int
            Minimum number of cells a gene is expressed in to pass filter.
        max_mito : float
            Maximum percentage of mitochondrial reads.
        n_mads : float
            Number of median absolute deviations for outlier detection.
        save_path : Optional[str]
            Path to save QC plots. If None, plots are displayed instead.
        plotly : bool
            Whether to use Plotly for interactive visualizations.
            
        Returns:
        --------
        Tuple[ad.AnnData, pd.DataFrame]
            Filtered AnnData object and summary statistics.
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

class Normalization:
    """
    Class for normalizing single-cell data using various methods.
    """
    
    def __init__(self, adata: ad.AnnData):
        """Initialize with AnnData object."""
        self.adata = adata
        
    def log_norm(self, 
               scale_factor: float = 10000, 
               log_base: float = 2, 
               inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Perform standard log normalization (library size normalization).
        
        Parameters:
        -----------
        scale_factor : float
            Scale factor for library size normalization.
        log_base : float
            Base for the logarithm (2 for log2, 10 for log10, math.e for ln).
        inplace : bool
            If True, modify self.adata, else return a normalized copy.
            
        Returns:
        --------
        Optional[AnnData]
            Normalized AnnData object if inplace is False.
        """
        adata = self.adata if inplace else self.adata.copy()
        
        print(f"Performing log normalization (scale_factor={scale_factor}, log_base={log_base})")
        
        # Library size normalization and log transformation
        sc.pp.normalize_total(adata, target_sum=scale_factor)
        sc.pp.log1p(adata, base=log_base)
        
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
        
        Parameters:
        -----------
        n_pools : int
            Number of pools for scran normalization.
        min_mean : float
            Minimum mean expression for genes to be used in normalization.
        inplace : bool
            If True, modify self.adata, else return a normalized copy.
            
        Returns:
        --------
        Optional[AnnData]
            Normalized AnnData object if inplace is False.
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
            
            # Clean up
            numpy2ri.deactivate()
            pandas2ri.deactivate()
            
        except ImportError:
            print("rpy2 not available. Falling back to scanpy's normalize_total.")
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            
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
        
        Parameters:
        -----------
        n_genes : int
            Maximum number of genes to use (for large datasets).
        n_cells : int
            Maximum number of cells to use (for large datasets).
        inplace : bool
            If True, modify self.adata, else return a normalized copy.
            
        Returns:
        --------
        Optional[AnnData]
            Normalized AnnData object if inplace is False.
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
            
            # Clean up
            numpy2ri.deactivate()
            pandas2ri.deactivate()
            
        except ImportError:
            print("rpy2 or sctransform not available. Falling back to scanpy's normalize_total.")
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            
        if not inplace:
            return adata
        else:
            self.adata = adata
            
    def clr_norm(self, 
               eps: float = 1.0,
               inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Perform centered log-ratio normalization.
        
        Parameters:
        -----------
        eps : float
            Pseudo-count to add to avoid log(0).
        inplace : bool
            If True, modify self.adata, else return a normalized copy.
            
        Returns:
        --------
        Optional[AnnData]
            Normalized AnnData object if inplace is False.
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
        
        if not inplace:
            return adata
        else:
            self.adata = adata
    
    def run_normalization(self, 
                        method: str = 'log', 
                        **kwargs) -> ad.AnnData:
        """
        Run normalization using the specified method.
        
        Parameters:
        -----------
        method : str
            Normalization method ('log', 'scran', 'sctransform', 'clr').
        **kwargs : dict
            Additional parameters for the specific normalization method.
            
        Returns:
        --------
        AnnData
            Normalized AnnData object.
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


class SpatialAnalysis:
    """
    Class for analyzing spatial transcriptomics data.
    """
    
    def __init__(self, adata: ad.AnnData, 
                 spatial_key: str = 'spatial',
                 x_coord: str = 'x', 
                 y_coord: str = 'y'):
        """
        Initialize with AnnData object and spatial coordinates.
        
        Parameters:
        -----------
        adata : AnnData
            AnnData object with spatial information.
        spatial_key : str
            Key in adata.obsm where spatial coordinates are stored.
        x_coord : str
            Name of x-coordinate in obs if not using obsm[spatial_key].
        y_coord : str
            Name of y-coordinate in obs if not using obsm[spatial_key].
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
        
        Parameters:
        -----------
        genes : List[str] or str
            Gene or list of genes to plot.
        ncols : int
            Number of columns in the plot grid.
        figsize : Optional[Tuple[int, int]]
            Figure size. If None, it's calculated based on the number of genes.
        cmap : str
            Colormap for gene expression.
        size : float
            Size of the points in the scatter plot.
        title_fontsize : int
            Font size for subplot titles.
        show_colorbar : bool
            Whether to show the colorbar.
        save_path : Optional[str]
            Path to save the figure. If None, the figure is displayed instead.
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object.
        """
        # Convert single gene to list
        if isinstance(genes, str):
            genes = [genes]
            
        # Make sure all genes exist in the dataset
        valid_genes = [gene for gene in genes if gene in self.adata.var_names]
        if len(valid_genes) == 0:
            raise ValueError("None of the specified genes were found in the dataset")
        elif len(valid_genes) < len(genes):
            missing = set(genes) - set(valid_genes)
            print(f"Warning: The following genes were not found: {', '.join(missing)}")
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
            
            # Get gene expression
            gene_expr = self.adata[:, gene].X
            if sparse.issparse(gene_expr):
                gene_expr = gene_expr.toarray().flatten()
                
            # Create the scatter plot
            scatter = ax.scatter(x, y, c=gene_expr, cmap=cmap, s=size, alpha=0.8)
            
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
        
        Parameters:
        -----------
        bin_size : float
            Size of the square bins.
        aggr_func : str
            Aggregation function ('mean', 'sum', 'median', 'max').
        min_cells : int
            Minimum number of cells required in a bin to keep it.
        genes : Optional[List[str]]
            Specific genes to include. If None, all genes are used.
            
        Returns:
        --------
        AnnData
            AnnData object containing the binned data.
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
        
        Parameters:
        -----------
        genes : Optional[List[str]]
            Specific genes to analyze. If None, all genes are analyzed.
        max_genes : int
            Maximum number of genes to analyze (to prevent excessive computation).
        n_jobs : int
            Number of parallel jobs for computation.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with Moran's I statistics for each gene.
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
                              bin_size: Optional[float] = None) -> pd.DataFrame:
        """
        Analyze negative probe statistics within spatial bins.
        
        Parameters:
        -----------
        prefix : str
            Prefix for identifying negative probe genes.
        bin_size : Optional[float]
            Size of spatial bins. If None, original cells are used.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with negative probe statistics for each bin.
        """
        # Identify negative probes
        negative_probes = [gene for gene in self.adata.var_names if gene.startswith(prefix)]
        
        if len(negative_probes) == 0:
            raise ValueError(f"No negative probes found with prefix '{prefix}'")
            
        print(f"Found {len(negative_probes)} negative probes with prefix '{prefix}'")
        
        # Use original cells or create bins
        if bin_size is None:
            # Use original cells
            adata = self.adata
            spatial_unit = "cell"
        else:
            # Create spatial bins
            adata = self.create_spatial_grid(bin_size=bin_size, genes=negative_probes)
            spatial_unit = "bin"
            
        # Calculate statistics for each spatial unit
        results = []
        
        for i in range(adata.n_obs):
            # Extract expression of negative probes
            expr = adata[i, negative_probes].X
            if sparse.issparse(expr):
                expr = expr.toarray().flatten()
                
            # Calculate statistics
            result = {
                f'{spatial_unit}_id': adata.obs_names[i],
                'x': adata.obsm[self.spatial_key][i, 0],
                'y': adata.obsm[self.spatial_key][i, 1],
                'negative_mean': np.mean(expr),
                'negative_sum': np.sum(expr),
                'negative_sd': np.std(expr),
                'negative_cv': np.std(expr) / np.mean(expr) if np.mean(expr) > 0 else np.nan
            }
            
            # Add statistics for individual probes
            for j, probe in enumerate(negative_probes):
                probe_value = expr[j]
                result[f'{probe}'] = probe_value
                
            results.append(result)
            
        # Convert to DataFrame
        stats_df = pd.DataFrame(results)
        
        return stats_df
