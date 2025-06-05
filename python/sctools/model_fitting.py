"""
Single Cell Model Fitter

Main class for fitting statistical models to single-cell expression data
with comprehensive normalization and batch effect correction capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import warnings
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
import anndata as ad

# Import model classes
from sctools.standard_models import (
    PoissonModel, NegativeBinomialModel, ZeroInflatedNBModel, 
    BernoulliModel, DepthAdjustedNBModel
)
from sctools.batch_aware_models import (
    RegularizedNBModel, HierarchicalNBModel, MixedEffectsNBModel
)


class SingleCellModelFitter:
    """
    A comprehensive class for fitting statistical models to single-cell expression data
    and performing diagnostic tests, visualizations, and batch effect correction.
    """
    
    def __init__(self, adata: ad.AnnData, layer: Optional[str] = None, batch_key: Optional[str] = None):
        """
        Initialize with AnnData object
        
        Parameters:
        ----------
        adata : AnnData
            Annotated data object containing expression data
        layer : str, optional
            Layer to use for analysis. If None, uses adata.X
        batch_key : str, optional
            Column in adata.obs containing batch information for batch-aware normalization
        """
        self.adata = adata
        self.layer = layer
        self.batch_key = batch_key
        self.fitted_models = {}
        self.model_results = {}
        
        # Built-in models
        self.available_models = {
            # Standard models
            'poisson': PoissonModel,
            'negative_binomial': NegativeBinomialModel,
            'zero_inflated_nb': ZeroInflatedNBModel,
            'bernoulli': BernoulliModel,
            'depth_adjusted_nb': DepthAdjustedNBModel,
            # Batch-aware models
            'regularized_nb': RegularizedNBModel,
            'hierarchical_nb': HierarchicalNBModel,
            'mixed_effects_nb': MixedEffectsNBModel
        }
        
        # Calculate size factors (batch-aware if batch_key provided)
        self.size_factors = self._calculate_size_factors()
        
        # Global trend for regularized models (fitted when needed)
        self.global_trend_fitted = False
    
    def get_data_matrix(self):
        """Get the data matrix from specified layer"""
        if self.layer is None:
            return self.adata.X
        else:
            return self.adata.layers[self.layer]
    
    def _is_sparse(self) -> bool:
        """Check if data matrix is sparse"""
        data = self.get_data_matrix()
        return sparse.issparse(data)
    
    def _calculate_size_factors(self, method: str = 'total_umi') -> np.ndarray:
        """
        Calculate size factors for depth adjustment (sparse-aware and batch-aware)
        """
        data_matrix = self.get_data_matrix()
        
        if self.batch_key is None:
            return self._calculate_single_batch_size_factors(data_matrix, method)
        else:
            return self._calculate_batch_aware_size_factors(data_matrix, method)
    
    def _calculate_single_batch_size_factors(self, data_matrix, method: str) -> np.ndarray:
        """Calculate size factors for single batch"""
        if method == 'total_umi':
            if sparse.issparse(data_matrix):
                total_umi = np.array(data_matrix.sum(axis=1)).flatten()
            else:
                total_umi = np.sum(data_matrix, axis=1)
            
            median_umi = np.median(total_umi)
            size_factors = total_umi / median_umi
            
        elif method == 'median_ratio':
            if sparse.issparse(data_matrix):
                n_genes_sample = min(1000, data_matrix.shape[1])
                gene_indices = np.random.choice(data_matrix.shape[1], n_genes_sample, replace=False)
                sample_data = data_matrix[:, gene_indices].toarray()
            else:
                sample_data = data_matrix
            
            # Calculate geometric mean for each gene across cells
            gene_geometric_means = np.exp(np.mean(np.log(sample_data + 1), axis=0)) - 1
            
            # Calculate ratios for each cell
            ratios = np.zeros_like(sample_data)
            for i in range(sample_data.shape[1]):
                if gene_geometric_means[i] > 0:
                    ratios[:, i] = sample_data[:, i] / gene_geometric_means[i]
                else:
                    ratios[:, i] = 0
            
            # Size factor is median ratio for each cell (excluding zeros)
            size_factors = np.zeros(sample_data.shape[0])
            for i in range(sample_data.shape[0]):
                cell_ratios = ratios[i, ratios[i, :] > 0]
                if len(cell_ratios) > 0:
                    size_factors[i] = np.median(cell_ratios)
                else:
                    size_factors[i] = 1.0
                    
        elif method == 'geometric_mean':
            if sparse.issparse(data_matrix):
                total_umi = np.array(data_matrix.sum(axis=1)).flatten()
            else:
                total_umi = np.sum(data_matrix, axis=1)
                
            log_total = np.log(total_umi + 1)
            mean_log_total = np.mean(log_total)
            size_factors = np.exp(log_total - mean_log_total)
            
        else:
            raise ValueError(f"Unknown size factor method: {method}")
        
        # Ensure no zero or negative size factors
        size_factors = np.maximum(size_factors, 0.01)
        return size_factors
    
    def _calculate_batch_aware_size_factors(self, data_matrix, method: str) -> np.ndarray:
        """
        Calculate batch-aware size factors (Tier 1 batch correction)
        
        Calculates size factors within each batch, then normalizes relative to reference batch
        """
        if self.batch_key not in self.adata.obs.columns:
            raise ValueError(f"Batch key '{self.batch_key}' not found in adata.obs")
        
        batches = self.adata.obs[self.batch_key].unique()
        size_factors = np.zeros(self.adata.n_obs)
        
        # Find reference batch (largest batch)
        batch_sizes = self.adata.obs[self.batch_key].value_counts()
        reference_batch = batch_sizes.index[0]
        reference_median = None
        
        print(f"Using batch-aware size factors with {len(batches)} batches")
        print(f"Reference batch: {reference_batch} ({batch_sizes[reference_batch]} cells)")
        
        for batch in batches:
            batch_mask = self.adata.obs[self.batch_key] == batch
            batch_indices = np.where(batch_mask)[0]
            
            if sparse.issparse(data_matrix):
                batch_data = data_matrix[batch_mask, :]
            else:
                batch_data = data_matrix[batch_mask, :]
            
            # Calculate size factors within this batch
            batch_size_factors = self._calculate_single_batch_size_factors(batch_data, method)
            
            # Store reference batch median for normalization
            if batch == reference_batch:
                reference_median = np.median(batch_size_factors)
            
            # Normalize to reference batch
            batch_median = np.median(batch_size_factors)
            if reference_median is not None and batch_median > 0:
                normalized_batch_factors = batch_size_factors / batch_median * reference_median
            else:
                normalized_batch_factors = batch_size_factors
            
            size_factors[batch_indices] = normalized_batch_factors
        
        # Ensure no zero or negative size factors
        size_factors = np.maximum(size_factors, 0.01)
        return size_factors
    
    def register_custom_model(self, name: str, model_class):
        """Register a custom model class"""
        self.available_models[name] = model_class
    
    def fit_models(self, models: List[str], genes: Optional[List[str]] = None, 
                   n_genes: int = 100, chunk_size: int = 100, 
                   max_memory_gb: float = 4.0) -> Dict[str, Dict[str, Any]]:
        """
        Fit multiple models to gene expression data (memory efficient)
        """
        data_matrix = self.get_data_matrix()
        is_sparse = sparse.issparse(data_matrix)
        
        # Select genes to analyze
        if genes is None:
            if is_sparse:
                mean_expression = np.array(data_matrix.mean(axis=0)).flatten()
            else:
                mean_expression = np.mean(data_matrix, axis=0)
            
            top_genes_idx = np.argsort(mean_expression)[-n_genes:]
            gene_names = [self.adata.var_names[i] for i in top_genes_idx]
        else:
            gene_names = genes
            top_genes_idx = [self.adata.var_names.get_loc(gene) for gene in genes]
        
        results = {}
        
        for model_name in models:
            if model_name not in self.available_models:
                warnings.warn(f"Model {model_name} not available. Skipping.")
                continue
            
            print(f"Fitting {model_name} model...")
            model_results = {}
            
            # Process genes in chunks to manage memory
            n_chunks = (len(top_genes_idx) + chunk_size - 1) // chunk_size
            
            for chunk_idx in tqdm(range(n_chunks), desc=f"Processing {model_name}"):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(top_genes_idx))
                
                chunk_gene_indices = top_genes_idx[start_idx:end_idx]
                chunk_gene_names = gene_names[start_idx:end_idx]
                
                # Extract chunk data (convert to dense if needed)
                if is_sparse:
                    chunk_data = data_matrix[:, chunk_gene_indices].toarray()
                else:
                    chunk_data = data_matrix[:, chunk_gene_indices]
                
                for i, gene_idx in enumerate(chunk_gene_indices):
                    gene_name = chunk_gene_names[i]
                    gene_data = chunk_data[:, i]
                    
                    # Fit model
                    model = self.available_models[model_name]()
                    
                    try:
                        # Special handling for different model types
                        if model_name == 'depth_adjusted_nb':
                            model.size_factors = self.size_factors
                            params = model.fit(gene_data, self.size_factors)
                            
                        elif model_name == 'regularized_nb':
                            # Fit global trend first if not done
                            if not self.global_trend_fitted:
                                print("Fitting global trend for regularized model...")
                                dummy_model = RegularizedNBModel()
                                dummy_model.fit_global_trend(data_matrix, self.size_factors)
                                self.global_trend = dummy_model.global_trend
                                self.global_trend_fitted = True
                            
                            model.global_trend = self.global_trend
                            model.size_factors = self.size_factors
                            params = model.fit(gene_data, self.size_factors)
                            
                        elif model_name in ['hierarchical_nb', 'mixed_effects_nb']:
                            # Batch-aware models
                            if self.batch_key is None:
                                warnings.warn(f"{model_name} model requires batch_key. Using regular NB instead.")
                                regular_model = NegativeBinomialModel()
                                params = regular_model.fit(gene_data)
                                model = regular_model
                            else:
                                batch_labels = self.adata.obs[self.batch_key].values
                                model.size_factors = self.size_factors
                                model.batch_labels = batch_labels
                                params = model.fit(gene_data, self.size_factors, batch_labels)
                                
                        else:
                            # Standard models
                            params = model.fit(gene_data)
                        
                        model_results[gene_name] = {
                            'model': model,
                            'params': params,
                            'aic': model.aic(gene_data),
                            'bic': model.bic(gene_data),
                            'data': gene_data,
                            'gene_idx': gene_idx
                        }
                        
                    except Exception as e:
                        warnings.warn(f"Failed to fit {model_name} to {gene_name}: {str(e)}")
                        continue
            
            results[model_name] = model_results
        
        self.model_results = results
        return results
    
    def compare_models(self, criterion: str = 'aic') -> pd.DataFrame:
        """Compare models using information criteria"""
        if not self.model_results:
            raise ValueError("No models fitted yet. Run fit_models() first.")
        
        comparison_data = []
        
        # Get all genes that were fitted for all models
        all_genes = set()
        for model_name in self.model_results:
            all_genes.update(self.model_results[model_name].keys())
        
        common_genes = all_genes.copy()
        for model_name in self.model_results:
            common_genes &= set(self.model_results[model_name].keys())
        
        for gene in common_genes:
            gene_results = {'gene': gene}
            for model_name in self.model_results:
                if gene in self.model_results[model_name]:
                    gene_results[f'{model_name}_{criterion}'] = self.model_results[model_name][gene][criterion]
            comparison_data.append(gene_results)
        
        df = pd.DataFrame(comparison_data)
        
        # Add best model column
        criterion_cols = [col for col in df.columns if col.endswith(f'_{criterion}')]
        df['best_model'] = df[criterion_cols].idxmin(axis=1).str.replace(f'_{criterion}', '')
        
        return df
    
    def normalize_expression(self, model_name: str = 'depth_adjusted_nb', 
                            method: str = 'log_norm',
                            genes: Optional[List[str]] = None,
                            chunk_size: int = 1000,
                            preserve_sparsity: bool = True):
        """
        Normalize expression data using fitted models (memory efficient)
        
        Parameters:
        ----------
        model_name : str
            Model to use for normalization
        method : str
            Normalization method:
            - 'log_norm': log(counts/size_factor + 1) [SPARSE-COMPATIBLE]
            - 'log_cpm': Log counts per million [SPARSE-COMPATIBLE]
            - 'size_factor': Simple size factor normalization [SPARSE-COMPATIBLE]
            - 'regularized_log': Regularized log transformation [SPARSE-COMPATIBLE]
            - 'batch_corrected': Batch-corrected residuals [DESTROYS SPARSITY]
            - 'pearson_residuals': Pearson residuals [DESTROYS SPARSITY]
            - 'deviance_residuals': Deviance residuals [DESTROYS SPARSITY]
        genes : List[str], optional
            Genes to normalize. If None, uses all fitted genes
        chunk_size : int
            Number of genes to process at once
        preserve_sparsity : bool
            Whether to preserve sparse format (recommended for large data)
        
        Returns:
        -------
        sparse.csr_matrix or np.ndarray : Normalized expression matrix
        """
        if model_name not in self.model_results:
            raise ValueError(f"Model {model_name} not fitted. Run fit_models() first.")
        
        data_matrix = self.get_data_matrix()
        is_sparse = sparse.issparse(data_matrix)
        
        if genes is None:
            genes = list(self.model_results[model_name].keys())
        
        # Get gene indices
        gene_indices = []
        valid_genes = []
        for gene in genes:
            try:
                idx = self.adata.var_names.get_loc(gene)
                gene_indices.append(idx)
                valid_genes.append(gene)
            except KeyError:
                warnings.warn(f"Gene {gene} not found in data")
                continue
        
        n_cells = data_matrix.shape[0]
        n_genes = len(gene_indices)
        
        # Check memory requirements for dense methods
        if method in ['pearson_residuals', 'deviance_residuals', 'batch_corrected'] and preserve_sparsity:
            memory_gb = n_cells * n_genes * 8 / (1024**3)
            if memory_gb > 2.0:
                warnings.warn(f"Residual methods will use ~{memory_gb:.1f}GB memory and destroy sparsity. "
                            f"Consider using 'log_norm' method or setting preserve_sparsity=False")
        
        if method in ['log_norm', 'log_cpm', 'size_factor', 'regularized_log'] and preserve_sparsity:
            # Sparse-compatible methods
            return self._normalize_sparse(data_matrix, method, gene_indices, valid_genes)
        else:
            # Dense methods (residuals)
            return self._normalize_dense(data_matrix, model_name, method, gene_indices, valid_genes, chunk_size)
    
    def _normalize_sparse(self, data_matrix, method: str, gene_indices: List[int], 
                         gene_names: List[str]):
        """Sparse-compatible normalization methods"""
        
        if method == 'log_norm':
            # log(counts/size_factor + 1) - preserves sparsity
            if sparse.issparse(data_matrix):
                subset_data = data_matrix[:, gene_indices]
                normalized = subset_data / self.size_factors.reshape(-1, 1)
                return sparse.csr_matrix(np.log1p(normalized))
        
        elif method == 'regularized_log':
            # Regularized log transformation - sparse compatible
            if 'regularized_nb' not in self.model_results:
                raise ValueError("Must fit regularized_nb model first for regularized_log normalization")
            
            if sparse.issparse(data_matrix):
                subset_data = data_matrix[:, gene_indices].toarray()
            else:
                subset_data = data_matrix[:, gene_indices]
            
            # Apply regularized log transformation gene by gene
            normalized_data = np.zeros_like(subset_data, dtype=np.float32)
            
            for i, gene_name in enumerate(gene_names):
                if gene_name in self.model_results['regularized_nb']:
                    model = self.model_results['regularized_nb'][gene_name]['model']
                    gene_data = subset_data[:, i]
                    normalized_data[:, i] = model.regularized_log_transform(gene_data, self.size_factors)
                else:
                    # Fallback to simple log normalization
                    gene_data = subset_data[:, i]
                    normalized_data[:, i] = np.log1p(gene_data / self.size_factors)
            
            # Convert back to sparse (many values close to 0)
            return sparse.csr_matrix(normalized_data)
        
        elif method == 'log_cpm':
            # log(CPM + 1) - sparse compatible
            if sparse.issparse(data_matrix):
                subset_data = data_matrix[:, gene_indices]
                total_counts = np.array(data_matrix.sum(axis=1)).flatten()
                
                # Convert to CPM
                cpm_factors = 1e6 / total_counts
                cpm_factors_broadcast = sparse.diags(cpm_factors)
                normalized = cpm_factors_broadcast @ subset_data
                
                # Apply log
                normalized.data = np.log1p(normalized.data)
                return normalized.tocsr()
            else:
                subset_data = data_matrix[:, gene_indices]
                total_counts = np.sum(data_matrix, axis=1)
                cpm = subset_data / total_counts.reshape(-1, 1) * 1e6
                return sparse.csr_matrix(np.log1p(cpm))
        
        elif method == 'size_factor':
            # Simple size factor normalization
            if sparse.issparse(data_matrix):
                subset_data = data_matrix[:, gene_indices]
                size_factors_broadcast = sparse.diags(1.0 / self.size_factors)
                normalized = size_factors_broadcast @ subset_data
                return normalized.tocsr()
            else:
                subset_data = data_matrix[:, gene_indices]
                normalized = subset_data / self.size_factors.reshape(-1, 1)
                return sparse.csr_matrix(normalized)
    
    def _normalize_dense(self, data_matrix, model_name: str, method: str, 
                        gene_indices: List[int], gene_names: List[str], 
                        chunk_size: int) -> np.ndarray:
        """Dense normalization methods (residuals)"""
        
        n_cells = data_matrix.shape[0]
        n_genes = len(gene_indices)
        normalized_matrix = np.zeros((n_cells, n_genes))
        
        # Process in chunks to manage memory
        n_chunks = (n_genes + chunk_size - 1) // chunk_size
        
        for chunk_idx in tqdm(range(n_chunks), desc="Normalizing"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_genes)
            
            chunk_gene_indices = gene_indices[start_idx:end_idx]
            chunk_gene_names = gene_names[start_idx:end_idx]
            
            # Extract chunk (convert to dense if needed)
            if sparse.issparse(data_matrix):
                chunk_data = data_matrix[:, chunk_gene_indices].toarray()
            else:
                chunk_data = data_matrix[:, chunk_gene_indices]
            
            for i, gene_name in enumerate(chunk_gene_names):
                if gene_name in self.model_results[model_name]:
                    model = self.model_results[model_name][gene_name]['model']
                    gene_data = chunk_data[:, i]
                    
                    if hasattr(model, method):
                        if model_name == 'depth_adjusted_nb':
                            normalized_matrix[:, start_idx + i] = getattr(model, method)(gene_data, self.size_factors)
                        elif model_name in ['hierarchical_nb', 'mixed_effects_nb'] and method == 'batch_corrected':
                            batch_labels = self.adata.obs[self.batch_key].values if self.batch_key else None
                            normalized_matrix[:, start_idx + i] = model.batch_corrected_residuals(gene_data, self.size_factors, batch_labels)
                        else:
                            normalized_matrix[:, start_idx + i] = getattr(model, method)(gene_data)
                    else:
                        # Fallback to log normalization
                        normalized_matrix[:, start_idx + i] = np.log1p(gene_data / self.size_factors * 10000)
                else:
                    # Gene not fitted, use simple normalization
                    gene_data = chunk_data[:, i]
                    normalized_matrix[:, start_idx + i] = np.log1p(gene_data / self.size_factors * 10000)
        
        return normalized_matrix
    
    def add_normalized_layer(self, layer_name: str = 'danb_normalized', 
                           model_name: str = 'depth_adjusted_nb',
                           method: str = 'log_norm',
                           genes: Optional[List[str]] = None,
                           preserve_sparsity: bool = True):
        """Add normalized expression as a new layer to AnnData object (memory efficient)"""
        if genes is None:
            # Normalize all genes that were fitted
            fitted_genes = list(self.model_results[model_name].keys())
            
            if preserve_sparsity and method in ['log_norm', 'log_cpm', 'size_factor', 'regularized_log']:
                # Sparse normalization
                normalized_data = self.normalize_expression(model_name, method, fitted_genes, preserve_sparsity=True)
                
                # Create sparse matrix for all genes
                n_cells, n_genes = self.adata.shape
                full_normalized = sparse.csr_matrix((n_cells, n_genes))
                
                # Insert normalized data for fitted genes
                gene_indices = [self.adata.var_names.get_loc(gene) for gene in fitted_genes]
                full_normalized[:, gene_indices] = normalized_data
                
                self.adata.layers[layer_name] = full_normalized
            else:
                # Dense normalization
                normalized_data = self.normalize_expression(model_name, method, fitted_genes, preserve_sparsity=False)
                
                # Create full matrix with zeros for non-fitted genes
                full_normalized = np.zeros((self.adata.shape[0], self.adata.shape[1]))
                gene_indices = [self.adata.var_names.get_loc(gene) for gene in fitted_genes]
                full_normalized[:, gene_indices] = normalized_data
                
                if preserve_sparsity:
                    self.adata.layers[layer_name] = sparse.csr_matrix(full_normalized)
                else:
                    self.adata.layers[layer_name] = full_normalized
        else:
            # Normalize specific genes
            normalized_data = self.normalize_expression(model_name, method, genes, preserve_sparsity=preserve_sparsity)
            
            # For subset of genes, create new layer or update existing
            if layer_name not in self.adata.layers:
                if preserve_sparsity:
                    self.adata.layers[layer_name] = sparse.csr_matrix(self.adata.shape)
                else:
                    self.adata.layers[layer_name] = np.zeros(self.adata.shape)
            
            gene_indices = [self.adata.var_names.get_loc(gene) for gene in genes]
            self.adata.layers[layer_name][:, gene_indices] = normalized_data
    
    def normalize_hvg_only(self, model_name: str = 'depth_adjusted_nb',
                          method: str = 'log_norm',
                          n_top_genes: int = 2000,
                          layer_name: str = 'hvg_normalized') -> List[str]:
        """Normalize only highly variable genes (memory efficient for large datasets)"""
        # Find highly variable genes
        data_matrix = self.get_data_matrix()
        
        if sparse.issparse(data_matrix):
            # Efficient variance calculation for sparse matrices
            means = np.array(data_matrix.mean(axis=0)).flatten()
            # For variance: E[X^2] - E[X]^2
            data_squared = data_matrix.copy()
            data_squared.data = data_squared.data ** 2
            means_squared = np.array(data_squared.mean(axis=0)).flatten()
            variances = means_squared - means ** 2
        else:
            means = np.mean(data_matrix, axis=0)
            variances = np.var(data_matrix, axis=0)
        
        # Select highly variable genes (exclude very low expression)
        valid_mask = means > 0.01  # Only consider expressed genes
        valid_means = means[valid_mask]
        valid_variances = variances[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # Calculate coefficient of variation
        cv = valid_variances / (valid_means + 1e-8)
        
        # Select top variable genes
        top_var_indices = np.argsort(cv)[-n_top_genes:]
        hvg_indices = valid_indices[top_var_indices]
        hvg_genes = [self.adata.var_names[i] for i in hvg_indices]
        
        print(f"Selected {len(hvg_genes)} highly variable genes")
        
        # Fit models only to HVG
        if model_name not in self.model_results:
            print(f"Fitting {model_name} to HVG...")
            self.fit_models([model_name], genes=hvg_genes)
        
        # Normalize HVG
        self.add_normalized_layer(
            layer_name=layer_name,
            model_name=model_name,
            method=method,
            genes=hvg_genes,
            preserve_sparsity=True
        )
        
        return hvg_genes
    
    def get_sparse_normalized_subset(self, genes: List[str], 
                                   model_name: str = 'depth_adjusted_nb',
                                   method: str = 'log_norm') -> Tuple[sparse.csr_matrix, List[str]]:
        """Get normalized data for a subset of genes in sparse format"""
        # Filter to genes that exist and are fitted
        valid_genes = []
        for gene in genes:
            if gene in self.adata.var_names and gene in self.model_results.get(model_name, {}):
                valid_genes.append(gene)
        
        if not valid_genes:
            raise ValueError("No valid genes found for normalization")
        
        print(f"Normalizing {len(valid_genes)}/{len(genes)} genes")
        
        normalized_data = self.normalize_expression(
            model_name=model_name,
            method=method,
            genes=valid_genes,
            preserve_sparsity=True
        )
        
        return normalized_data, valid_genes
    
    # Tier 4: Post-normalization batch correction methods
    def correct_batch_effects(self, layer_name: str, 
                             corrected_layer_name: str = None,
                             method: str = 'combat',
                             batch_key: Optional[str] = None) -> None:
        """Apply post-normalization batch correction to existing normalized layer"""
        if layer_name not in self.adata.layers:
            raise ValueError(f"Layer '{layer_name}' not found in adata.layers")
        
        if batch_key is None:
            batch_key = self.batch_key
        
        if batch_key is None:
            raise ValueError("No batch_key provided and none set in constructor")
        
        if batch_key not in self.adata.obs.columns:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
        
        if corrected_layer_name is None:
            corrected_layer_name = layer_name
        
        # Get normalized data
        normalized_data = self.adata.layers[layer_name]
        if sparse.issparse(normalized_data):
            normalized_data = normalized_data.toarray()
        
        batch_labels = self.adata.obs[batch_key].values
        
        if method == 'combat':
            corrected_data = self._combat_correction(normalized_data, batch_labels)
        elif method == 'center_scale':
            corrected_data = self._center_scale_correction(normalized_data, batch_labels)
        else:
            raise ValueError(f"Unknown batch correction method: {method}")
        
        # Store corrected data
        self.adata.layers[corrected_layer_name] = corrected_data
        
        print(f"Applied {method} batch correction to {layer_name}")
        if corrected_layer_name != layer_name:
            print(f"Corrected data saved as {corrected_layer_name}")
    
    def _combat_correction(self, data: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
        """ComBat-style batch correction"""
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)
        
        if n_batches == 1:
            print("Only one batch found, no correction needed")
            return data
        
        print(f"Applying ComBat correction to {n_batches} batches")
        
        # Center data
        gene_means = np.mean(data, axis=0)
        centered_data = data - gene_means
        
        # Calculate batch effects for each gene
        corrected_data = centered_data.copy()
        
        for gene_idx in tqdm(range(data.shape[1]), desc="Correcting genes"):
            gene_data = centered_data[:, gene_idx]
            
            # Calculate batch-specific means and variances
            batch_means = []
            batch_vars = []
            batch_sizes = []
            
            for batch in unique_batches:
                batch_mask = batch_labels == batch
                batch_gene_data = gene_data[batch_mask]
                
                if len(batch_gene_data) > 1:
                    batch_means.append(np.mean(batch_gene_data))
                    batch_vars.append(np.var(batch_gene_data))
                    batch_sizes.append(len(batch_gene_data))
                else:
                    batch_means.append(0.0)
                    batch_vars.append(1.0)
                    batch_sizes.append(1)
            
            batch_means = np.array(batch_means)
            batch_vars = np.array(batch_vars)
            batch_sizes = np.array(batch_sizes)
            
            # Estimate global parameters (empirical Bayes)
            global_mean = np.average(batch_means, weights=batch_sizes)
            global_var = np.average(batch_vars, weights=batch_sizes)
            
            # Shrink batch effects towards global estimates
            shrinkage_factor = 0.1
            
            for i, batch in enumerate(unique_batches):
                batch_mask = batch_labels == batch
                
                # Shrink batch mean towards global mean
                shrunk_mean = shrinkage_factor * global_mean + (1 - shrinkage_factor) * batch_means[i]
                
                # Shrink batch variance towards global variance
                shrunk_var = shrinkage_factor * global_var + (1 - shrinkage_factor) * batch_vars[i]
                shrunk_std = np.sqrt(max(shrunk_var, 0.01))
                
                # Apply correction
                if batch_sizes[i] > 1:
                    batch_data = gene_data[batch_mask]
                    # Remove batch mean and scale by batch variance
                    corrected_batch = (batch_data - shrunk_mean) * np.sqrt(global_var) / shrunk_std
                    corrected_data[batch_mask, gene_idx] = corrected_batch
        
        # Add back global gene means
        corrected_data = corrected_data + gene_means
        
        return corrected_data
    
    def _center_scale_correction(self, data: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
        """Simple center-and-scale batch correction"""
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)
        
        if n_batches == 1:
            print("Only one batch found, no correction needed")
            return data
        
        print(f"Applying center-scale correction to {n_batches} batches")
        
        corrected_data = data.copy()
        
        # Calculate global statistics
        global_means = np.mean(data, axis=0)
        global_stds = np.std(data, axis=0)
        global_stds = np.maximum(global_stds, 0.01)  # Avoid division by zero
        
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_data = data[batch_mask, :]
            
            if np.sum(batch_mask) > 1:
                # Calculate batch-specific statistics
                batch_means = np.mean(batch_data, axis=0)
                batch_stds = np.std(batch_data, axis=0)
                batch_stds = np.maximum(batch_stds, 0.01)  # Avoid division by zero
                
                # Center and scale to match global statistics
                corrected_batch = (batch_data - batch_means) / batch_stds * global_stds + global_means
                corrected_data[batch_mask, :] = corrected_batch
        
        return corrected_data
    
    # Visualization and diagnostic methods
    def plot_batch_effects(self, layer_name: str = None, 
                          batch_key: Optional[str] = None,
                          genes: List[str] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> None:
        """Visualize batch effects before and after correction"""
        if batch_key is None:
            batch_key = self.batch_key
        
        if batch_key is None:
            raise ValueError("No batch_key provided")
        
        if layer_name is None:
            data = self.adata.X
        else:
            data = self.adata.layers[layer_name]
        
        if sparse.issparse(data):
            data = data.toarray()
        
        batch_labels = self.adata.obs[batch_key].values
        
        if genes is None:
            # Select top variable genes
            gene_vars = np.var(data, axis=0)
            top_var_indices = np.argsort(gene_vars)[-6:]  # Top 6 genes
            genes = [self.adata.var_names[i] for i in top_var_indices]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, gene in enumerate(genes[:6]):
            if i >= 6:
                break
                
            gene_idx = self.adata.var_names.get_loc(gene)
            gene_data = data[:, gene_idx]
            
            # Create DataFrame for plotting
            plot_df = pd.DataFrame({
                'expression': gene_data,
                'batch': batch_labels
            })
            
            # Box plot
            sns.boxplot(data=plot_df, x='batch', y='expression', ax=axes[i])
            axes[i].set_title(f'{gene}')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print batch effect statistics
        print("Batch Effect Analysis:")
        print(f"Number of batches: {len(np.unique(batch_labels))}")
        
        for gene in genes[:3]:  # Show stats for first 3 genes
            gene_idx = self.adata.var_names.get_loc(gene)
            gene_data = data[:, gene_idx]
            
            batch_means = []
            for batch in np.unique(batch_labels):
                batch_mask = batch_labels == batch
                batch_mean = np.mean(gene_data[batch_mask])
                batch_means.append(batch_mean)
            
            batch_effect_var = np.var(batch_means)
            within_batch_var = np.mean([np.var(gene_data[batch_labels == batch]) 
                                       for batch in np.unique(batch_labels)])
            
            if within_batch_var > 0:
                batch_effect_ratio = batch_effect_var / within_batch_var
                print(f"{gene}: Batch effect ratio = {batch_effect_ratio:.3f}")
    
    def validate_batch_correction(self, original_layer: str, 
                                corrected_layer: str,
                                batch_key: Optional[str] = None) -> Dict[str, float]:
        """Validate batch correction by calculating metrics"""
        if batch_key is None:
            batch_key = self.batch_key
        
        if batch_key is None:
            raise ValueError("No batch_key provided")
        
        original_data = self.adata.layers[original_layer]
        corrected_data = self.adata.layers[corrected_layer]
        
        if sparse.issparse(original_data):
            original_data = original_data.toarray()
        if sparse.issparse(corrected_data):
            corrected_data = corrected_data.toarray()
        
        batch_labels = self.adata.obs[batch_key].values
        
        # Calculate metrics
        metrics = {}
        
        # 1. Batch effect reduction (variance of batch means)
        original_batch_var = self._calculate_batch_variance(original_data, batch_labels)
        corrected_batch_var = self._calculate_batch_variance(corrected_data, batch_labels)
        
        metrics['batch_variance_reduction'] = 1 - (corrected_batch_var / original_batch_var)
        
        # 2. Biological signal preservation (within-batch variance)
        original_within_var = self._calculate_within_batch_variance(original_data, batch_labels)
        corrected_within_var = self._calculate_within_batch_variance(corrected_data, batch_labels)
        
        metrics['biological_signal_preservation'] = corrected_within_var / original_within_var
        
        # 3. Overall data variance preservation
        metrics['total_variance_preservation'] = np.var(corrected_data) / np.var(original_data)
        
        print("Batch Correction Validation:")
        print(f"Batch variance reduction: {metrics['batch_variance_reduction']:.3f} (higher is better)")
        print(f"Biological signal preservation: {metrics['biological_signal_preservation']:.3f} (closer to 1 is better)")
        print(f"Total variance preservation: {metrics['total_variance_preservation']:.3f} (closer to 1 is better)")
        
        return metrics
    
    def _calculate_batch_variance(self, data: np.ndarray, batch_labels: np.ndarray) -> float:
        """Calculate variance of batch means (measures batch effects)"""
        batch_means = []
        for batch in np.unique(batch_labels):
            batch_mask = batch_labels == batch
            batch_mean = np.mean(data[batch_mask], axis=0)
            batch_means.append(batch_mean)
        
        batch_means = np.array(batch_means)
        return np.mean(np.var(batch_means, axis=0))
    
    def _calculate_within_batch_variance(self, data: np.ndarray, batch_labels: np.ndarray) -> float:
        """Calculate average within-batch variance (measures biological signal)"""
        within_batch_vars = []
        for batch in np.unique(batch_labels):
            batch_mask = batch_labels == batch
            batch_data = data[batch_mask]
            if len(batch_data) > 1:
                within_batch_vars.append(np.var(batch_data, axis=0))
        
        if len(within_batch_vars) > 0:
            return np.mean(np.array(within_batch_vars))
        else:
            return 0.0
    
    # Additional diagnostic methods
    def generate_diagnostic_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive diagnostic report"""
        if not self.model_results:
            raise ValueError("No models fitted yet. Run fit_models() first.")
        
        report = []
        report.append("=== Single Cell Expression Model Fitting Report ===\n")
        
        # Dataset summary
        data_matrix = self.get_data_matrix()
        if sparse.issparse(data_matrix):
            total_umi = np.array(data_matrix.sum()).flatten()[0]
            mean_umi_per_cell = np.array(data_matrix.sum(axis=1)).flatten().mean()
            median_genes_per_cell = np.median(np.array(data_matrix.getnnz(axis=1)).flatten())
        else:
            total_umi = np.sum(data_matrix)
            mean_umi_per_cell = np.mean(np.sum(data_matrix, axis=1))
            median_genes_per_cell = np.median(np.sum(data_matrix > 0, axis=1))
        
        report.append(f"Dataset: {data_matrix.shape[0]} cells, {data_matrix.shape[1]} genes")
        report.append(f"Data layer: {'X' if self.layer is None else self.layer}")
        report.append(f"Mean total UMI per cell: {mean_umi_per_cell:.1f}")
        report.append(f"Median genes per cell: {median_genes_per_cell:.0f}")
        
        if self.batch_key:
            n_batches = len(self.adata.obs[self.batch_key].unique())
            report.append(f"Batch-aware analysis: {n_batches} batches")
        
        report.append("")
        
        # Model fitting summary
        report.append("=== Model Fitting Summary ===")
        for model_name, model_data in self.model_results.items():
            report.append(f"{model_name.title()}: {len(model_data)} genes fitted")
        
        # Model comparison
        try:
            comparison_df = self.compare_models('aic')
            report.append("\n=== Model Comparison (AIC) ===")
            best_model_counts = comparison_df['best_model'].value_counts()
            for model, count in best_model_counts.items():
                pct = 100 * count / len(comparison_df)
                report.append(f"{model}: {count} genes ({pct:.1f}%)")
        except:
            report.append("\n=== Model Comparison ===")
            report.append("Could not generate comparison (insufficient data)")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_mean_variance_relationship(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot mean-variance relationship for fitted models"""
        if not self.model_results:
            raise ValueError("No models fitted yet. Run fit_models() first.")
        
        data_matrix = self.get_data_matrix()
        
        # Calculate empirical mean and variance for each gene
        if sparse.issparse(data_matrix):
            gene_means = np.array(data_matrix.mean(axis=0)).flatten()
            gene_vars = np.array(data_matrix.power(2).mean(axis=0)).flatten() - gene_means**2
        else:
            gene_means = np.mean(data_matrix, axis=0)
            gene_vars = np.var(data_matrix, axis=0)
        
        # Filter out genes with zero variance
        valid_mask = (gene_means > 0) & (gene_vars > 0)
        gene_means = gene_means[valid_mask]
        gene_vars = gene_vars[valid_mask]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot empirical relationship
        axes[0].scatter(gene_means, gene_vars, alpha=0.6, s=20)
        axes[0].plot(gene_means, gene_means, 'r--', label='Poisson (var=mean)', alpha=0.7)
        axes[0].plot(gene_means, gene_means + 2*gene_means**2, 'g--', label='NB (overdispersed)', alpha=0.7)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Mean Expression')
        axes[0].set_ylabel('Variance')
        axes[0].set_title('Empirical Mean-Variance Relationship')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot theoretical relationships for fitted models
        model_colors = ['blue', 'orange', 'green', 'red']
        
        for i, (model_name, model_data) in enumerate(self.model_results.items()):
            if i >= 3:  # Limit to first 3 models
                break
                
            theoretical_means = []
            theoretical_vars = []
            
            for gene_name, gene_results in model_data.items():
                mean, var = gene_results['model'].mean_var_relationship()
                if mean > 0 and var > 0:
                    theoretical_means.append(mean)
                    theoretical_vars.append(var)
            
            if theoretical_means:
                axes[i+1].scatter(theoretical_means, theoretical_vars, 
                                alpha=0.6, s=20, color=model_colors[i])
                axes[i+1].set_xscale('log')
                axes[i+1].set_yscale('log')
                axes[i+1].set_xlabel('Theoretical Mean')
                axes[i+1].set_ylabel('Theoretical Variance')
                axes[i+1].set_title(f'{model_name.title()} Model')
                axes[i+1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, criterion: str = 'aic', figsize: Tuple[int, int] = (12, 6)):
        """Plot model comparison results"""
        comparison_df = self.compare_models(criterion)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Distribution of criterion values
        criterion_cols = [col for col in comparison_df.columns if col.endswith(f'_{criterion}')]
        comparison_df[criterion_cols].plot(kind='box', ax=ax1)
        ax1.set_title(f'Distribution of {criterion.upper()} Values')
        ax1.set_ylabel(f'{criterion.upper()}')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Best model frequency
        best_model_counts = comparison_df['best_model'].value_counts()
        best_model_counts.plot(kind='bar', ax=ax2)
        ax2.set_title('Best Model Frequency')
        ax2.set_ylabel('Number of Genes')
        ax2.set_xlabel('Model')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_qq_plots(self, genes: List[str], models: Optional[List[str]] = None, 
                      figsize: Tuple[int, int] = (15, 10)):
        """Generate Q-Q plots for specified genes and models"""
        if not self.model_results:
            raise ValueError("No models fitted yet. Run fit_models() first.")
        
        if models is None:
            models = list(self.model_results.keys())
        
        n_genes = len(genes)
        n_models = len(models)
        
        fig, axes = plt.subplots(n_genes, n_models, figsize=figsize)
        if n_genes == 1:
            axes = axes.reshape(1, -1)
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, gene in enumerate(genes):
            for j, model_name in enumerate(models):
                if gene in self.model_results[model_name]:
                    gene_data = self.model_results[model_name][gene]['data']
                    model = self.model_results[model_name][gene]['model']
                    
                    # Generate theoretical samples
                    n_samples = len(gene_data)
                    theoretical_samples = model.generate_samples(n_samples)
                    
                    # Create Q-Q plot
                    from scipy import stats
                    stats.probplot(gene_data, dist=lambda x: theoretical_samples, 
                                 plot=axes[i, j])
                    axes[i, j].set_title(f'{gene} - {model_name}')
                    axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def test_zero_inflation(self, threshold_ratio: float = 1.5) -> pd.DataFrame:
        """Test for zero-inflation by comparing observed vs expected zeros"""
        if 'negative_binomial' not in self.model_results:
            raise ValueError("Negative binomial model must be fitted first.")
        
        results = []
        
        for gene_name, gene_results in self.model_results['negative_binomial'].items():
            gene_data = gene_results['data']
            model = gene_results['model']
            
            # Observed zeros
            observed_zeros = np.sum(gene_data == 0)
            total_cells = len(gene_data)
            observed_zero_prop = observed_zeros / total_cells
            
            # Expected zeros from NB model
            expected_zero_prob = model.pmf_or_pdf(np.array([0]))[0]
            expected_zeros = expected_zero_prob * total_cells
            
            # Test ratio
            if expected_zeros > 0:
                zero_ratio = observed_zeros / expected_zeros
            else:
                zero_ratio = np.inf if observed_zeros > 0 else 1.0
            
            results.append({
                'gene': gene_name,
                'observed_zeros': observed_zeros,
                'expected_zeros': expected_zeros,
                'observed_zero_prop': observed_zero_prop,
                'expected_zero_prop': expected_zero_prob,
                'zero_inflation_ratio': zero_ratio,
                'zero_inflated': zero_ratio > threshold_ratio
            })
        
        return pd.DataFrame(results)
    
    def plot_normalization_effects(self, genes: List[str], figsize: Tuple[int, int] = (15, 10)):
        """Visualize the effects of normalization on selected genes"""
        if 'depth_adjusted_nb' not in self.model_results:
            raise ValueError("DANB model not fitted. Run fit_models() first.")
        
        data_matrix = self.get_data_matrix()
        if sparse.issparse(data_matrix):
            data_matrix = data_matrix.toarray()
        
        n_genes = len(genes)
        
        fig, axes = plt.subplots(n_genes, 3, figsize=figsize)
        if n_genes == 1:
            axes = axes.reshape(1, -1)
        
        for i, gene in enumerate(genes):
            if gene not in self.model_results['depth_adjusted_nb']:
                continue
                
            gene_idx = self.adata.var_names.get_loc(gene)
            raw_counts = data_matrix[:, gene_idx]
            model = self.model_results['depth_adjusted_nb'][gene]['model']
            
            # Get normalized values
            pearson_res = model.pearson_residuals(raw_counts, self.size_factors)
            deviance_res = model.deviance_residuals(raw_counts, self.size_factors)
            
            # Plot raw counts vs size factors
            axes[i, 0].scatter(self.size_factors, raw_counts, alpha=0.6, s=20)
            axes[i, 0].set_xlabel('Size Factor')
            axes[i, 0].set_ylabel('Raw Counts')
            axes[i, 0].set_title(f'{gene} - Raw Counts')
            axes[i, 0].set_yscale('log')
            
            # Plot Pearson residuals vs size factors
            axes[i, 1].scatter(self.size_factors, pearson_res, alpha=0.6, s=20, color='orange')
            axes[i, 1].set_xlabel('Size Factor')
            axes[i, 1].set_ylabel('Pearson Residuals')
            axes[i, 1].set_title(f'{gene} - Pearson Residuals')
            axes[i, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Plot histogram of residuals
            axes[i, 2].hist(pearson_res, bins=30, alpha=0.7, color='orange', label='Pearson')
            axes[i, 2].hist(deviance_res, bins=30, alpha=0.7, color='green', label='Deviance')
            axes[i, 2].set_xlabel('Residuals')
            axes[i, 2].set_ylabel('Frequency')
            axes[i, 2].set_title(f'{gene} - Residual Distribution')
            axes[i, 2].legend()
            axes[i, 2].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def diagnose_batch_effects(self, layer_name: Optional[str] = None) -> float:
        """Quick batch effect diagnosis"""
        if self.batch_key is None:
            raise ValueError("No batch_key set for batch effect diagnosis")
        
        # Get data
        if layer_name is None:
            data = self.adata.X
            layer_desc = "raw data"
        else:
            data = self.adata.layers[layer_name]
            layer_desc = f"layer '{layer_name}'"
        
        if sparse.issparse(data):
            data = data.toarray()
        
        batch_labels = self.adata.obs[self.batch_key].values
        
        # Calculate batch effect strength
        batch_var = self._calculate_batch_variance(data, batch_labels)
        within_var = self._calculate_within_batch_variance(data, batch_labels)
        
        batch_effect_ratio = batch_var / within_var if within_var > 0 else np.inf
        
        print(f"Batch Effect Analysis for {layer_desc}:")
        print(f"Batch effect ratio: {batch_effect_ratio:.3f}")
        
        if batch_effect_ratio > 0.1:
            print("⚠️  Strong batch effects detected - correction recommended")
        elif batch_effect_ratio > 0.05:
            print("⚡ Moderate batch effects - correction may help")
        else:
            print("✅ Minimal batch effects")
        
        return batch_effect_ratio

    def get_batch_summary(self) -> pd.DataFrame:
        """Get summary statistics for each batch"""
        if self.batch_key is None:
            raise ValueError("No batch_key set")
        
        data_matrix = self.get_data_matrix()
        batch_labels = self.adata.obs[self.batch_key].values
        
        summary_data = []
        
        for batch in np.unique(batch_labels):
            batch_mask = batch_labels == batch
            
            if sparse.issparse(data_matrix):
                batch_data = data_matrix[batch_mask, :]
                n_cells = batch_data.shape[0]
                total_umi = np.array(batch_data.sum(axis=1)).flatten()
                genes_per_cell = np.array(batch_data.getnnz(axis=1)).flatten()
            else:
                batch_data = data_matrix[batch_mask, :]
                n_cells = batch_data.shape[0]
                total_umi = np.sum(batch_data, axis=1)
                genes_per_cell = np.sum(batch_data > 0, axis=1)
            
            summary_data.append({
                'batch': batch,
                'n_cells': n_cells,
                'mean_umi_per_cell': np.mean(total_umi),
                'median_umi_per_cell': np.median(total_umi),
                'mean_genes_per_cell': np.mean(genes_per_cell),
                'median_genes_per_cell': np.median(genes_per_cell),
                'size_factor_mean': np.mean(self.size_factors[batch_mask]),
                'size_factor_std': np.std(self.size_factors[batch_mask])
            })
        
        return pd.DataFrame(summary_data)

    def export_normalized_data(self, layer_name: str, output_file: str, 
                              format: str = 'h5ad', genes: Optional[List[str]] = None):
        """
        Export normalized data to file
        
        Parameters:
        ----------
        layer_name : str
            Name of the layer to export
        output_file : str
            Output file path
        format : str
            Output format ('h5ad', 'csv', 'tsv', 'mtx')
        genes : List[str], optional
            Specific genes to export. If None, exports all genes
        """
        if layer_name not in self.adata.layers:
            raise ValueError(f"Layer '{layer_name}' not found in adata.layers")
        
        data = self.adata.layers[layer_name]
        
        if genes is not None:
            gene_indices = [self.adata.var_names.get_loc(gene) for gene in genes 
                           if gene in self.adata.var_names]
            if sparse.issparse(data):
                data = data[:, gene_indices]
            else:
                data = data[:, gene_indices]
            
            # Create subset AnnData for export
            adata_subset = self.adata[:, gene_indices].copy()
            adata_subset.X = data
        else:
            # Export full dataset
            adata_subset = self.adata.copy()
            adata_subset.X = data
        
        if format == 'h5ad':
            adata_subset.write(output_file)
        elif format == 'csv':
            if sparse.issparse(data):
                data = data.toarray()
            df = pd.DataFrame(data.T, 
                            index=adata_subset.var_names, 
                            columns=adata_subset.obs_names)
            df.to_csv(output_file)
        elif format == 'tsv':
            if sparse.issparse(data):
                data = data.toarray()
            df = pd.DataFrame(data.T, 
                            index=adata_subset.var_names, 
                            columns=adata_subset.obs_names)
            df.to_csv(output_file, sep='\t')
        elif format == 'mtx':
            from scipy.io import mmwrite
            if not sparse.issparse(data):
                data = sparse.csr_matrix(data)
            mmwrite(output_file, data.T)  # Transpose for gene x cell format
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Exported {data.shape[0]} cells x {data.shape[1]} genes to {output_file}")

    def save_model_results(self, output_file: str):
        """Save fitted model results to file for later use"""
        import pickle
        
        # Prepare data for saving (exclude large arrays to save space)
        save_data = {
            'model_results': {},
            'size_factors': self.size_factors,
            'batch_key': self.batch_key,
            'layer': self.layer,
            'global_trend': getattr(self, 'global_trend', None)
        }
        
        # Save model parameters and metadata (but not large data arrays)
        for model_name, model_data in self.model_results.items():
            save_data['model_results'][model_name] = {}
            for gene_name, gene_results in model_data.items():
                save_data['model_results'][model_name][gene_name] = {
                    'params': gene_results['params'],
                    'aic': gene_results['aic'],
                    'bic': gene_results['bic'],
                    'gene_idx': gene_results['gene_idx']
                    # Note: excluding 'model' and 'data' to save space
                }
        
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Model results saved to {output_file}")
        print("Note: Model objects not saved to reduce file size. Re-fit models if needed.")

    def load_model_results(self, input_file: str):
        """Load previously saved model results"""
        import pickle
        
        with open(input_file, 'rb') as f:
            save_data = pickle.load(f)
        
        self.size_factors = save_data['size_factors']
        self.batch_key = save_data['batch_key']
        self.layer = save_data['layer']
        
        if save_data['global_trend'] is not None:
            self.global_trend = save_data['global_trend']
            self.global_trend_fitted = True
        
        # Note: Model objects need to be re-fitted as they weren't saved
        print(f"Model results loaded from {input_file}")
        print("Note: You'll need to re-fit models to use normalization methods that require model objects.")

    def get_gene_statistics(self, genes: Optional[List[str]] = None) -> pd.DataFrame:
        """Get comprehensive statistics for genes"""
        data_matrix = self.get_data_matrix()
        
        if genes is None:
            genes = list(self.adata.var_names)
        
        gene_indices = [self.adata.var_names.get_loc(gene) for gene in genes 
                       if gene in self.adata.var_names]
        valid_genes = [gene for gene in genes if gene in self.adata.var_names]
        
        stats_data = []
        
        for i, gene in enumerate(valid_genes):
            gene_idx = gene_indices[i]
            
            if sparse.issparse(data_matrix):
                gene_data = data_matrix[:, gene_idx].toarray().flatten()
            else:
                gene_data = data_matrix[:, gene_idx]
            
            # Basic statistics
            stats_dict = {
                'gene': gene,
                'mean_expression': np.mean(gene_data),
                'median_expression': np.median(gene_data),
                'std_expression': np.std(gene_data),
                'var_expression': np.var(gene_data),
                'detection_rate': np.mean(gene_data > 0),
                'n_cells_expressing': np.sum(gene_data > 0),
                'max_expression': np.max(gene_data),
                'cv': np.std(gene_data) / (np.mean(gene_data) + 1e-8)
            }
            
            # Add model fitting results if available
            for model_name in self.model_results:
                if gene in self.model_results[model_name]:
                    model_params = self.model_results[model_name][gene]['params']
                    for param_name, param_value in model_params.items():
                        stats_dict[f'{model_name}_{param_name}'] = param_value
                    
                    stats_dict[f'{model_name}_aic'] = self.model_results[model_name][gene]['aic']
                    stats_dict[f'{model_name}_bic'] = self.model_results[model_name][gene]['bic']
            
            stats_data.append(stats_dict)
        
        return pd.DataFrame(stats_data)

    def create_summary_plots(self, output_dir: str = './model_summary_plots'):
        """Create a comprehensive set of summary plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating summary plots in {output_dir}")
        
        # 1. Mean-variance relationship
        try:
            self.plot_mean_variance_relationship()
            plt.savefig(f"{output_dir}/mean_variance_relationship.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not create mean-variance plot: {e}")
        
        # 2. Model comparison
        try:
            self.plot_model_comparison()
            plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not create model comparison plot: {e}")
        
        # 3. Batch effects (if applicable)
        if self.batch_key is not None:
            try:
                self.plot_batch_effects()
                plt.savefig(f"{output_dir}/batch_effects_raw.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # Plot batch effects for normalized layers
                for layer_name in self.adata.layers.keys():
                    if 'normalized' in layer_name or 'corrected' in layer_name:
                        try:
                            self.plot_batch_effects(layer_name=layer_name)
                            plt.savefig(f"{output_dir}/batch_effects_{layer_name}.png", 
                                      dpi=300, bbox_inches='tight')
                            plt.close()
                        except Exception as e:
                            print(f"Could not create batch effects plot for {layer_name}: {e}")
                            
            except Exception as e:
                print(f"Could not create batch effects plots: {e}")
        
        # 4. Size factor distribution
        try:
            plt.figure(figsize=(10, 6))
            if self.batch_key is not None:
                batch_labels = self.adata.obs[self.batch_key]
                for batch in batch_labels.unique():
                    batch_mask = batch_labels == batch
                    batch_sf = self.size_factors[batch_mask]
                    plt.hist(batch_sf, alpha=0.7, label=f'Batch {batch}', bins=50)
                plt.legend()
            else:
                plt.hist(self.size_factors, bins=50)
            
            plt.xlabel('Size Factor')
            plt.ylabel('Number of Cells')
            plt.title('Distribution of Size Factors')
            plt.savefig(f"{output_dir}/size_factor_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not create size factor plot: {e}")
        
        print("Summary plots created successfully!")

    def __repr__(self) -> str:
        """String representation of the fitter"""
        n_cells, n_genes = self.adata.shape
        n_models = len(self.model_results)
        n_layers = len(self.adata.layers)
        
        repr_str = f"SingleCellModelFitter:\n"
        repr_str += f"  Data: {n_cells} cells × {n_genes} genes\n"
        repr_str += f"  Layer: {'X' if self.layer is None else self.layer}\n"
        repr_str += f"  Batch key: {self.batch_key}\n"
        repr_str += f"  Models fitted: {n_models}\n"
        repr_str += f"  Available layers: {n_layers}\n"
        
        if self.model_results:
            repr_str += f"  Fitted models: {list(self.model_results.keys())}\n"
        
        return repr_str

    def __str__(self) -> str:
        """User-friendly string representation"""
        return self.__repr__()

    def get_available_methods(self) -> Dict[str, List[str]]:
        """Get available normalization methods for each model type"""
        methods = {
            'sparse_compatible': [
                'log_norm', 'log_cpm', 'size_factor', 'regularized_log'
            ],
            'dense_only': [
                'pearson_residuals', 'deviance_residuals', 'batch_corrected'
            ],
            'batch_correction': [
                'combat', 'center_scale'
            ]
        }
        
        return methods

    def get_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage of current data and results"""
        import sys
        
        usage = {}
        
        # Data matrix memory
        data_matrix = self.get_data_matrix()
        if sparse.issparse(data_matrix):
            usage['data_matrix_mb'] = data_matrix.data.nbytes / (1024**2)
        else:
            usage['data_matrix_mb'] = data_matrix.nbytes / (1024**2)
        
        # Size factors
        usage['size_factors_mb'] = self.size_factors.nbytes / (1024**2)
        
        # Model results (approximate)
        model_results_size = 0
        for model_name, model_data in self.model_results.items():
            for gene_name, gene_results in model_data.items():
                # Approximate size of stored data
                model_results_size += sys.getsizeof(gene_results['data'])
                model_results_size += sys.getsizeof(gene_results['params'])
        
        usage['model_results_mb'] = model_results_size / (1024**2)
        
        # Layers
        layers_size = 0
        for layer_name, layer_data in self.adata.layers.items():
            if sparse.issparse(layer_data):
                layers_size += layer_data.data.nbytes
            else:
                layers_size += layer_data.nbytes
        
        usage['layers_mb'] = layers_size / (1024**2)
        
        usage['total_estimated_mb'] = sum(usage.values())
        
        return usage

    def cleanup(self, keep_normalized_layers: bool = True):
        """
        Clean up memory by removing large stored objects
        
        Parameters:
        ----------
        keep_normalized_layers : bool
            Whether to keep normalized layers in adata.layers
        """
        # Clear model results (keep parameters but remove data and model objects)
        for model_name in self.model_results:
            for gene_name in self.model_results[model_name]:
                # Remove large data arrays and model objects
                if 'data' in self.model_results[model_name][gene_name]:
                    del self.model_results[model_name][gene_name]['data']
                if 'model' in self.model_results[model_name][gene_name]:
                    del self.model_results[model_name][gene_name]['model']
        
        # Optionally clear normalized layers
        if not keep_normalized_layers:
            layers_to_remove = []
            for layer_name in self.adata.layers:
                if any(keyword in layer_name.lower() 
                      for keyword in ['normalized', 'corrected', 'residuals']):
                    layers_to_remove.append(layer_name)
            
            for layer_name in layers_to_remove:
                del self.adata.layers[layer_name]
        
        # Reset global trend
        self.global_trend_fitted = False
        if hasattr(self, 'global_trend'):
            del self.global_trend
        
        print("Memory cleanup completed.")
        print("Note: You'll need to re-fit models to use methods that require model objects.")

    def reset(self):
        """Reset the fitter to initial state"""
        self.model_results = {}
        self.fitted_models = {}
        self.global_trend_fitted = False
        
        if hasattr(self, 'global_trend'):
            del self.global_trend
        
        # Recalculate size factors
        self.size_factors = self._calculate_size_factors()
        
        print("Fitter reset to initial state.")

    def get_help(self, method_name: Optional[str] = None):
        """Get help for specific methods"""
        if method_name is None:
            help_text = """
SingleCellModelFitter - Main Methods:

Model Fitting:
  - fit_models(models, genes=None, n_genes=100)
  - compare_models(criterion='aic')
  - register_custom_model(name, model_class)

Normalization:
  - normalize_expression(model_name, method='log_norm', genes=None)
  - add_normalized_layer(layer_name, model_name, method='log_norm')
  - normalize_hvg_only(model_name, method='log_norm', n_top_genes=2000)

Batch Effects:
  - correct_batch_effects(layer_name, method='combat')
  - plot_batch_effects(layer_name=None, genes=None)
  - validate_batch_correction(original_layer, corrected_layer)
  - diagnose_batch_effects(layer_name=None)

Visualization:
  - plot_mean_variance_relationship()
  - plot_model_comparison()
  - plot_normalization_effects(genes)
  - create_summary_plots(output_dir='./plots')

Analysis:
  - get_gene_statistics(genes=None)
  - get_batch_summary()
  - test_zero_inflation()
  - generate_diagnostic_report()

Export/IO:
  - export_normalized_data(layer_name, output_file, format='h5ad')
  - save_model_results(output_file)
  - load_model_results(input_file)

Use get_help('method_name') for specific method help.
            """
            print(help_text)
        else:
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                if hasattr(method, '__doc__') and method.__doc__:
                    print(f"Help for {method_name}:")
                    print(method.__doc__)
                else:
                    print(f"No documentation available for {method_name}")
            else:
                print(f"Method '{method_name}' not found")
                print("Use get_help() to see available methods")
