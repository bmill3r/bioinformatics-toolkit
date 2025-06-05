"""
Batch-Aware Statistical Models for Single-Cell RNA-seq Data

This module contains advanced statistical models that explicitly account for batch effects
through hierarchical modeling and regularization techniques.
"""

import numpy as np
from scipy import stats, sparse
from typing import Optional, Dict, Tuple
from sctools.standard_models import BaseModel, DepthAdjustedNBModel


class RegularizedNBModel(BaseModel):
    """
    Regularized Negative Binomial Model (sctransform-style)
    
    Uses global mean-variance trend to regularize gene-specific parameters.
    Handles Tier 1 batch effects through batch-aware size factors.
    """
    
    def __init__(self, regularization_alpha: float = 0.5):
        super().__init__()
        self.alpha = regularization_alpha
        self.global_trend = None
        self.size_factors = None
    
    def fit_global_trend(self, data_matrix, size_factors: np.ndarray, 
                        n_genes_sample: int = 1000):
        """Fit global mean-variance trend across genes"""
        self.size_factors = size_factors
        
        # Sample genes for trend fitting
        if sparse.issparse(data_matrix):
            n_genes = data_matrix.shape[1]
            gene_indices = np.random.choice(n_genes, min(n_genes_sample, n_genes), replace=False)
            sample_data = data_matrix[:, gene_indices].toarray()
        else:
            gene_indices = np.random.choice(data_matrix.shape[1], 
                                          min(n_genes_sample, data_matrix.shape[1]), replace=False)
            sample_data = data_matrix[:, gene_indices]
        
        # Calculate normalized mean and variance for each sampled gene
        gene_means = []
        gene_vars = []
        
        for i in range(sample_data.shape[1]):
            gene_data = sample_data[:, i]
            normalized_data = gene_data / size_factors
            
            mean_expr = np.mean(normalized_data)
            var_expr = np.var(normalized_data)
            
            if mean_expr > 0 and var_expr > 0:
                gene_means.append(mean_expr)
                gene_vars.append(var_expr)
        
        gene_means = np.array(gene_means)
        gene_vars = np.array(gene_vars)
        
        # Fit log(variance) ~ log(mean) relationship
        valid_mask = (gene_means > 0) & (gene_vars > gene_means)
        
        if np.sum(valid_mask) > 10:
            log_means = np.log(gene_means[valid_mask])
            log_vars = np.log(gene_vars[valid_mask])
            
            # Fit linear trend
            coeffs = np.polyfit(log_means, log_vars, 1)
            self.global_trend = {
                'slope': coeffs[0],
                'intercept': coeffs[1]
            }
        else:
            # Fallback to simple relationship
            self.global_trend = {
                'slope': 2.0,
                'intercept': 0.0
            }
    
    def fit(self, data: np.ndarray, size_factors: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Fit regularized NB model to a single gene"""
        if size_factors is not None:
            self.size_factors = size_factors
        elif self.size_factors is None:
            self.size_factors = np.ones(len(data))
        
        if self.global_trend is None:
            raise ValueError("Must fit global trend first using fit_global_trend()")
        
        # Calculate gene-specific estimates
        valid_mask = (self.size_factors > 0) & (data >= 0)
        data_valid = data[valid_mask]
        size_factors_valid = self.size_factors[valid_mask]
        
        if len(data_valid) == 0:
            self.params = {'mu': 0.01, 'theta': 1.0, 'theta_regularized': 1.0}
            self.fitted = True
            return self.params
        
        # Gene-specific estimates
        normalized_data = data_valid / size_factors_valid
        gene_mean = np.mean(normalized_data)
        gene_var = np.var(normalized_data)
        
        # Gene-specific theta estimate
        if gene_var > gene_mean and gene_mean > 0:
            theta_gene = (gene_mean ** 2) / max(gene_var - gene_mean, 0.01)
        else:
            theta_gene = 100.0
        
        # Global trend prediction
        if gene_mean > 0:
            log_mean = np.log(gene_mean)
            predicted_log_var = (self.global_trend['slope'] * log_mean + 
                                self.global_trend['intercept'])
            predicted_var = np.exp(predicted_log_var)
            
            if predicted_var > gene_mean:
                theta_global = (gene_mean ** 2) / max(predicted_var - gene_mean, 0.01)
            else:
                theta_global = 100.0
        else:
            theta_global = 1.0
        
        # Regularized theta (weighted average)
        theta_regularized = (self.alpha * theta_global + 
                           (1 - self.alpha) * theta_gene)
        
        self.params = {
            'mu': max(gene_mean, 0.001),
            'theta': max(theta_gene, 0.01),
            'theta_regularized': max(theta_regularized, 0.01)
        }
        self.fitted = True
        return self.params
    
    def pmf_or_pdf(self, x: np.ndarray, size_factors: Optional[np.ndarray] = None, **params) -> np.ndarray:
        """Calculate probability using regularized parameters"""
        mu = params.get('mu', self.params['mu'])
        theta = params.get('theta_regularized', self.params['theta_regularized'])
        
        if size_factors is None:
            size_factors = self.size_factors
        if size_factors is None:
            size_factors = np.ones(len(x))
        
        expected_counts = mu * size_factors
        r = theta
        p = r / (r + expected_counts)
        
        return stats.nbinom.pmf(x, r, p)
    
    def mean_var_relationship(self, size_factors: Optional[np.ndarray] = None, **params) -> Tuple[float, float]:
        """Return mean and variance using regularized parameters"""
        mu = params.get('mu', self.params['mu'])
        theta = params.get('theta_regularized', self.params['theta_regularized'])
        
        if size_factors is None:
            size_factors = self.size_factors
        if size_factors is None:
            size_factors = np.array([1.0])
        
        mean_size_factor = np.mean(size_factors)
        mean_count = mu * mean_size_factor
        var_count = mean_count + (mean_count ** 2) / theta
        
        return mean_count, var_count
    
    def generate_samples(self, n_samples: int, size_factors: Optional[np.ndarray] = None, **params) -> np.ndarray:
        """Generate samples using regularized parameters"""
        mu = params.get('mu', self.params['mu'])
        theta = params.get('theta_regularized', self.params['theta_regularized'])
        
        if size_factors is None:
            if self.size_factors is not None and len(self.size_factors) >= n_samples:
                size_factors = self.size_factors[:n_samples]
            else:
                size_factors = np.ones(n_samples)
        
        samples = np.zeros(n_samples)
        for i in range(n_samples):
            expected_count = mu * size_factors[i]
            r = theta
            p = r / (r + expected_count)
            samples[i] = np.random.negative_binomial(r, p)
        
        return samples
    
    def regularized_log_transform(self, data: np.ndarray, 
                                 size_factors: Optional[np.ndarray] = None,
                                 clip_value: float = 30.0) -> np.ndarray:
        """
        Calculate regularized log-transformed values (sctransform-style normalization)
        
        This is the key normalization output that preserves sparsity while stabilizing variance.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if size_factors is None:
            size_factors = self.size_factors
        if size_factors is None:
            size_factors = np.ones(len(data))
        
        mu = self.params['mu']
        theta = self.params['theta_regularized']
        
        # Expected counts for each cell
        expected_counts = mu * size_factors
        
        # Regularization parameter (inversely related to theta)
        c = 1.0 / theta
        
        # Regularized log transformation: log((observed + c) / (expected + c))
        numerator = data + c
        denominator = expected_counts + c
        denominator = np.maximum(denominator, 1e-8)
        
        regularized_values = np.log(numerator / denominator)
        
        # Clip extreme values
        regularized_values = np.clip(regularized_values, -clip_value, clip_value)
        
        return regularized_values


class HierarchicalNBModel(BaseModel):
    """
    Hierarchical Negative Binomial Model with Batch Effects (Tier 3)
    
    Models: Y_ij ~ NB(μ_ij * s_i, θ_j)
    where: log(μ_ij) = β_j + γ_b[i]
    - β_j = gene-specific baseline expression
    - γ_b[i] = batch effect for cell i's batch
    - s_i = cell-specific size factor
    - θ_j = gene-specific dispersion
    
    This model explicitly accounts for batch effects through random effects.
    """
    
    def __init__(self, batch_labels: Optional[np.ndarray] = None, size_factors: Optional[np.ndarray] = None):
        super().__init__()
        self.batch_labels = batch_labels
        self.size_factors = size_factors
        self.batch_effects = {}
        
    def fit(self, data: np.ndarray, size_factors: Optional[np.ndarray] = None, 
            batch_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Fit hierarchical NB model with batch effects"""
        if size_factors is not None:
            self.size_factors = size_factors
        elif self.size_factors is None:
            self.size_factors = np.ones(len(data))
            
        if batch_labels is not None:
            self.batch_labels = batch_labels
        elif self.batch_labels is None:
            self.batch_labels = np.zeros(len(data))
        
        # Get unique batches
        unique_batches = np.unique(self.batch_labels)
        
        if len(unique_batches) == 1:
            # Only one batch - use regular DANB fitting
            regular_model = DepthAdjustedNBModel(self.size_factors)
            params = regular_model.fit(data, self.size_factors)
            self.params = params
            self.params['batch_effects'] = {unique_batches[0]: 0.0}
            self.fitted = True
            return self.params
        
        # Fit batch-specific models
        batch_params = {}
        batch_sizes = []
        
        for batch in unique_batches:
            batch_mask = self.batch_labels == batch
            batch_counts = data[batch_mask]
            batch_size_factors = self.size_factors[batch_mask]
            
            if len(batch_counts) < 3:  # Need minimum cells
                continue
                
            # Fit DANB model to this batch
            batch_model = DepthAdjustedNBModel(batch_size_factors)
            batch_model.fit(batch_counts, batch_size_factors)
            
            batch_params[batch] = batch_model.params.copy()
            batch_sizes.append(len(batch_counts))
        
        if len(batch_params) == 0:
            # Fallback if no batches have enough cells
            self.params = {'mu': np.mean(data / self.size_factors), 'theta': 1.0, 'batch_effects': {}}
            self.fitted = True
            return self.params
        
        # Estimate global parameters using empirical Bayes approach
        batch_mus = [params['mu'] for params in batch_params.values()]
        batch_thetas = [params['theta'] for params in batch_params.values()]
        
        # Global estimates (weighted by batch size)
        batch_weights = np.array(batch_sizes)
        batch_weights = batch_weights / np.sum(batch_weights)
        
        global_mu = np.average(batch_mus, weights=batch_weights)
        global_theta = np.average(batch_thetas, weights=batch_weights)
        
        # Calculate batch effects relative to global mean
        batch_effects = {}
        for batch, params in batch_params.items():
            # Batch effect in log scale
            batch_effects[batch] = np.log(params['mu'] / global_mu) if global_mu > 0 else 0.0
        
        # Regularize batch effects (shrink towards zero)
        shrinkage_factor = 0.1  # Can be tuned
        for batch in batch_effects:
            batch_effects[batch] *= (1 - shrinkage_factor)
        
        self.params = {
            'mu': global_mu,
            'theta': global_theta,
            'batch_effects': batch_effects
        }
        self.batch_effects = batch_effects
        self.fitted = True
        
        return self.params
    
    def pmf_or_pdf(self, x: np.ndarray, size_factors: Optional[np.ndarray] = None, 
                   batch_labels: Optional[np.ndarray] = None, **params) -> np.ndarray:
        """Calculate probability accounting for batch effects"""
        mu = params.get('mu', self.params['mu'])
        theta = params.get('theta', self.params['theta'])
        batch_effects = params.get('batch_effects', self.params['batch_effects'])
        
        if size_factors is None:
            size_factors = self.size_factors
        if size_factors is None:
            size_factors = np.ones(len(x))
            
        if batch_labels is None:
            batch_labels = self.batch_labels
        if batch_labels is None:
            batch_labels = np.zeros(len(x))
        
        probs = np.zeros(len(x))
        
        for i in range(len(x)):
            batch = batch_labels[i]
            batch_effect = batch_effects.get(batch, 0.0)
            
            # Adjust mean for batch effect
            adjusted_mu = mu * np.exp(batch_effect)
            expected_count = adjusted_mu * size_factors[i]
            
            # NB parameters
            r = theta
            p = r / (r + expected_count) if expected_count > 0 else 1.0
            
            probs[i] = stats.nbinom.pmf(x[i], r, p)
        
        return probs
    
    def mean_var_relationship(self, size_factors: Optional[np.ndarray] = None,
                            batch_labels: Optional[np.ndarray] = None, **params) -> Tuple[float, float]:
        """Return mean and variance accounting for batch effects"""
        mu = params.get('mu', self.params['mu'])
        theta = params.get('theta', self.params['theta'])
        
        if size_factors is None:
            size_factors = self.size_factors
        if size_factors is None:
            size_factors = np.array([1.0])
        
        # Average across size factors and batches
        mean_size_factor = np.mean(size_factors)
        mean_count = mu * mean_size_factor
        var_count = mean_count + (mean_count ** 2) / theta
        
        return mean_count, var_count
    
    def generate_samples(self, n_samples: int, size_factors: Optional[np.ndarray] = None,
                        batch_labels: Optional[np.ndarray] = None, **params) -> np.ndarray:
        """Generate samples accounting for batch effects"""
        mu = params.get('mu', self.params['mu'])
        theta = params.get('theta', self.params['theta'])
        batch_effects = params.get('batch_effects', self.params['batch_effects'])
        
        if size_factors is None:
            if self.size_factors is not None and len(self.size_factors) >= n_samples:
                size_factors = self.size_factors[:n_samples]
            else:
                size_factors = np.ones(n_samples)
                
        if batch_labels is None:
            if self.batch_labels is not None and len(self.batch_labels) >= n_samples:
                batch_labels = self.batch_labels[:n_samples]
            else:
                batch_labels = np.zeros(n_samples)
        
        samples = np.zeros(n_samples)
        
        for i in range(n_samples):
            batch = batch_labels[i]
            batch_effect = batch_effects.get(batch, 0.0)
            
            # Adjust mean for batch effect
            adjusted_mu = mu * np.exp(batch_effect)
            expected_count = adjusted_mu * size_factors[i]
            
            # Generate sample
            r = theta
            p = r / (r + expected_count) if expected_count > 0 else 1.0
            samples[i] = np.random.negative_binomial(r, p)
        
        return samples
    
    def batch_corrected_residuals(self, data: np.ndarray, 
                                 size_factors: Optional[np.ndarray] = None,
                                 batch_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate batch-corrected residuals
        
        These residuals have batch effects removed while preserving biological variation.
        This is the key normalization method for hierarchical models.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if size_factors is None:
            size_factors = self.size_factors
        if size_factors is None:
            size_factors = np.ones(len(data))
            
        if batch_labels is None:
            batch_labels = self.batch_labels
        if batch_labels is None:
            batch_labels = np.zeros(len(data))
        
        mu = self.params['mu']
        theta = self.params['theta']
        batch_effects = self.params['batch_effects']
        
        residuals = np.zeros(len(data))
        
        for i in range(len(data)):
            batch = batch_labels[i]
            batch_effect = batch_effects.get(batch, 0.0)
            
            # Expected count with batch effect (for variance calculation)
            expected_with_batch = mu * np.exp(batch_effect) * size_factors[i]
            
            # Expected count without batch effect (for residual calculation)
            expected_no_batch = mu * size_factors[i]
            
            # Variance (using batch-adjusted mean)
            variance = expected_with_batch + (expected_with_batch ** 2) / theta
            
            # Residual removes batch effect
            residuals[i] = (data[i] - expected_no_batch) / np.sqrt(variance + 1e-8)
        
        return residuals
    
    def get_batch_effects_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of fitted batch effects
        
        Returns:
        -------
        Dict with batch effect statistics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        summary = {}
        batch_effects = self.params['batch_effects']
        
        for batch, effect in batch_effects.items():
            # Convert log effect back to multiplicative scale
            multiplicative_effect = np.exp(effect)
            summary[batch] = {
                'log_effect': effect,
                'multiplicative_effect': multiplicative_effect,
                'percent_change': (multiplicative_effect - 1) * 100
            }
        
        return summary


class MixedEffectsNBModel(BaseModel):
    """
    Mixed Effects Negative Binomial Model (Advanced Tier 3)
    
    More sophisticated hierarchical model with:
    - Fixed effects for covariates
    - Random effects for batches
    - Shared dispersion estimation
    
    This is for advanced users who need more control over batch effect modeling.
    """
    
    def __init__(self, batch_labels: Optional[np.ndarray] = None, 
                 covariates: Optional[np.ndarray] = None,
                 size_factors: Optional[np.ndarray] = None):
        super().__init__()
        self.batch_labels = batch_labels
        self.covariates = covariates  # Additional fixed effects
        self.size_factors = size_factors
        self.fixed_effects = {}
        self.random_effects = {}
        
    def fit(self, data: np.ndarray, size_factors: Optional[np.ndarray] = None, 
            batch_labels: Optional[np.ndarray] = None,
            covariates: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Fit mixed effects model
        
        This is a simplified implementation. In practice, you'd use specialized
        libraries like statsmodels or R's lme4.
        """
        if size_factors is not None:
            self.size_factors = size_factors
        elif self.size_factors is None:
            self.size_factors = np.ones(len(data))
            
        if batch_labels is not None:
            self.batch_labels = batch_labels
        elif self.batch_labels is None:
            self.batch_labels = np.zeros(len(data))
            
        if covariates is not None:
            self.covariates = covariates
        elif self.covariates is None:
            self.covariates = np.ones((len(data), 1))  # Intercept only
        
        # Simplified fitting: use hierarchical approach as approximation
        hierarchical_model = HierarchicalNBModel(self.batch_labels, self.size_factors)
        hierarchical_params = hierarchical_model.fit(data, self.size_factors, self.batch_labels)
        
        # Copy parameters
        self.params = hierarchical_params.copy()
        self.random_effects = hierarchical_params['batch_effects']
        
        # Add fixed effects (simplified - just intercept)
        self.fixed_effects = {'intercept': np.log(hierarchical_params['mu'])}
        
        self.fitted = True
        return self.params
    
    def pmf_or_pdf(self, x: np.ndarray, **params) -> np.ndarray:
        """Calculate probability (delegates to hierarchical model for now)"""
        # This is a placeholder - full implementation would use the mixed effects structure
        hierarchical_model = HierarchicalNBModel(self.batch_labels, self.size_factors)
        hierarchical_model.params = self.params
        hierarchical_model.fitted = True
        
        return hierarchical_model.pmf_or_pdf(x, self.size_factors, self.batch_labels, **params)
    
    def mean_var_relationship(self, **params) -> Tuple[float, float]:
        """Return mean and variance (delegates to hierarchical model)"""
        hierarchical_model = HierarchicalNBModel(self.batch_labels, self.size_factors)
        hierarchical_model.params = self.params
        hierarchical_model.fitted = True
        
        return hierarchical_model.mean_var_relationship(self.size_factors, self.batch_labels, **params)
    
    def generate_samples(self, n_samples: int, **params) -> np.ndarray:
        """Generate samples (delegates to hierarchical model)"""
        hierarchical_model = HierarchicalNBModel(self.batch_labels, self.size_factors)
        hierarchical_model.params = self.params
        hierarchical_model.fitted = True
        
        return hierarchical_model.generate_samples(n_samples, self.size_factors, self.batch_labels, **params)