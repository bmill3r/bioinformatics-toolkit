"""
Standard Statistical Models for Single-Cell RNA-seq Data

This module contains basic statistical model implementations that do not account for batch effects.
These models are suitable for single-batch datasets or when batch effects are handled separately.
"""

import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple


class BaseModel(ABC):
    """Abstract base class for all statistical models"""
    
    def __init__(self):
        self.params = {}
        self.fitted = False
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> Dict[str, float]:
        """Fit model to data and return parameters"""
        pass
    
    @abstractmethod
    def pmf_or_pdf(self, x: np.ndarray, **params) -> np.ndarray:
        """Probability mass/density function"""
        pass
    
    @abstractmethod
    def mean_var_relationship(self, **params) -> Tuple[float, float]:
        """Return theoretical mean and variance given parameters"""
        pass
    
    @abstractmethod
    def generate_samples(self, n_samples: int, **params) -> np.ndarray:
        """Generate samples from the fitted distribution"""
        pass
    
    def aic(self, data: np.ndarray) -> float:
        """Calculate Akaike Information Criterion"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        log_likelihood = np.sum(np.log(self.pmf_or_pdf(data, **self.params) + 1e-10))
        k = len(self.params)
        return 2 * k - 2 * log_likelihood
    
    def bic(self, data: np.ndarray) -> float:
        """Calculate Bayesian Information Criterion"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        log_likelihood = np.sum(np.log(self.pmf_or_pdf(data, **self.params) + 1e-10))
        k = len(self.params)
        n = len(data)
        return k * np.log(n) - 2 * log_likelihood


class PoissonModel(BaseModel):
    """Poisson distribution model for count data"""
    
    def fit(self, data: np.ndarray) -> Dict[str, float]:
        self.params = {'lambda': np.mean(data)}
        self.fitted = True
        return self.params
    
    def pmf_or_pdf(self, x: np.ndarray, **params) -> np.ndarray:
        lam = params.get('lambda', self.params['lambda'])
        return stats.poisson.pmf(x, lam)
    
    def mean_var_relationship(self, **params) -> Tuple[float, float]:
        lam = params.get('lambda', self.params['lambda'])
        return lam, lam  # For Poisson: mean = variance = lambda
    
    def generate_samples(self, n_samples: int, **params) -> np.ndarray:
        lam = params.get('lambda', self.params['lambda'])
        return np.random.poisson(lam, n_samples)


class NegativeBinomialModel(BaseModel):
    """Negative Binomial distribution model for overdispersed count data"""
    
    def fit(self, data: np.ndarray) -> Dict[str, float]:
        data = data[data > 0]  # Remove zeros for initial estimation
        if len(data) == 0:
            self.params = {'mu': 0.1, 'alpha': 1.0}
        else:
            sample_mean = np.mean(data)
            sample_var = np.var(data)
            
            # Method of moments estimation
            if sample_var > sample_mean:
                alpha = (sample_var - sample_mean) / (sample_mean ** 2)
                mu = sample_mean
            else:
                alpha = 1.0
                mu = sample_mean
            
            self.params = {'mu': max(mu, 0.01), 'alpha': max(alpha, 0.01)}
        
        self.fitted = True
        return self.params
    
    def pmf_or_pdf(self, x: np.ndarray, **params) -> np.ndarray:
        mu = params.get('mu', self.params['mu'])
        alpha = params.get('alpha', self.params['alpha'])
        
        # Parameterization: r = 1/alpha, p = r/(r+mu)
        r = 1 / alpha
        p = r / (r + mu)
        
        return stats.nbinom.pmf(x, r, p)
    
    def mean_var_relationship(self, **params) -> Tuple[float, float]:
        mu = params.get('mu', self.params['mu'])
        alpha = params.get('alpha', self.params['alpha'])
        
        variance = mu + alpha * (mu ** 2)
        return mu, variance
    
    def generate_samples(self, n_samples: int, **params) -> np.ndarray:
        mu = params.get('mu', self.params['mu'])
        alpha = params.get('alpha', self.params['alpha'])
        
        r = 1 / alpha
        p = r / (r + mu)
        
        return np.random.negative_binomial(r, p, n_samples)


class ZeroInflatedNBModel(BaseModel):
    """Zero-Inflated Negative Binomial model for data with excess zeros"""
    
    def fit(self, data: np.ndarray) -> Dict[str, float]:
        # Simple estimation - in practice, you'd use EM algorithm
        zero_prop = np.mean(data == 0)
        non_zero_data = data[data > 0]
        
        if len(non_zero_data) == 0:
            self.params = {'mu': 0.1, 'alpha': 1.0, 'pi': 0.9}
        else:
            sample_mean = np.mean(non_zero_data)
            sample_var = np.var(non_zero_data)
            
            alpha = max((sample_var - sample_mean) / (sample_mean ** 2), 0.01)
            mu = sample_mean
            pi = min(zero_prop, 0.95)  # Cap at 95%
            
            self.params = {'mu': mu, 'alpha': alpha, 'pi': pi}
        
        self.fitted = True
        return self.params
    
    def pmf_or_pdf(self, x: np.ndarray, **params) -> np.ndarray:
        mu = params.get('mu', self.params['mu'])
        alpha = params.get('alpha', self.params['alpha'])
        pi = params.get('pi', self.params['pi'])
        
        # Zero-inflation component
        prob_zero = pi + (1 - pi) * self._nb_pmf(0, mu, alpha)
        prob_nonzero = (1 - pi) * self._nb_pmf(x, mu, alpha)
        
        result = np.where(x == 0, prob_zero, prob_nonzero)
        return result
    
    def _nb_pmf(self, x: np.ndarray, mu: float, alpha: float) -> np.ndarray:
        r = 1 / alpha
        p = r / (r + mu)
        return stats.nbinom.pmf(x, r, p)
    
    def mean_var_relationship(self, **params) -> Tuple[float, float]:
        mu = params.get('mu', self.params['mu'])
        alpha = params.get('alpha', self.params['alpha'])
        pi = params.get('pi', self.params['pi'])
        
        mean = (1 - pi) * mu
        variance = (1 - pi) * (mu + alpha * mu**2) + pi * (1 - pi) * mu**2
        return mean, variance
    
    def generate_samples(self, n_samples: int, **params) -> np.ndarray:
        mu = params.get('mu', self.params['mu'])
        alpha = params.get('alpha', self.params['alpha'])
        pi = params.get('pi', self.params['pi'])
        
        # Generate zero-inflation mask
        zero_mask = np.random.binomial(1, pi, n_samples)
        
        # Generate NB samples
        r = 1 / alpha
        p = r / (r + mu)
        nb_samples = np.random.negative_binomial(r, p, n_samples)
        
        # Apply zero-inflation
        return np.where(zero_mask, 0, nb_samples)


class BernoulliModel(BaseModel):
    """Bernoulli model for presence/absence analysis"""
    
    def fit(self, data: np.ndarray) -> Dict[str, float]:
        # Convert to binary (detected/not detected)
        binary_data = (data > 0).astype(int)
        self.params = {'p': np.mean(binary_data)}
        self.fitted = True
        return self.params
    
    def pmf_or_pdf(self, x: np.ndarray, **params) -> np.ndarray:
        p = params.get('p', self.params['p'])
        binary_x = (x > 0).astype(int)
        return stats.bernoulli.pmf(binary_x, p)
    
    def mean_var_relationship(self, **params) -> Tuple[float, float]:
        p = params.get('p', self.params['p'])
        return p, p * (1 - p)
    
    def generate_samples(self, n_samples: int, **params) -> np.ndarray:
        p = params.get('p', self.params['p'])
        return np.random.binomial(1, p, n_samples)


class DepthAdjustedNBModel(BaseModel):
    """
    Depth-Adjusted Negative Binomial Model
    
    Models counts as: Y_ij ~ NB(μ_j * s_i, θ_j)
    where:
    - μ_j = gene-specific mean expression
    - s_i = cell-specific size factor (depth)
    - θ_j = gene-specific dispersion parameter
    
    This handles Tier 1 batch effects through batch-aware size factors.
    """
    
    def __init__(self, size_factors: Optional[np.ndarray] = None):
        super().__init__()
        self.size_factors = size_factors
    
    def fit(self, data: np.ndarray, size_factors: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Fit DANB model to a single gene across all cells"""
        if size_factors is not None:
            self.size_factors = size_factors
        elif self.size_factors is None:
            self.size_factors = np.ones(len(data))
        
        # Remove cells with zero size factors
        valid_mask = (self.size_factors > 0) & (data >= 0)
        data_valid = data[valid_mask]
        size_factors_valid = self.size_factors[valid_mask]
        
        if len(data_valid) == 0:
            self.params = {'mu': 0.01, 'theta': 1.0}
            self.fitted = True
            return self.params
        
        # Normalize by size factors for initial estimation
        normalized_data = data_valid / size_factors_valid
        
        # Method of moments estimation
        sample_mean = np.mean(normalized_data)
        sample_var = np.var(normalized_data)
        
        if sample_var > sample_mean and sample_mean > 0:
            theta = (sample_mean ** 2) / max(sample_var - sample_mean, 0.01)
        else:
            theta = 100.0  # High theta = low dispersion
        
        mu = max(sample_mean, 0.001)
        
        # Refine estimates using simplified MLE
        try:
            mu, theta = self._mle_estimation(data_valid, size_factors_valid, mu, theta)
        except:
            pass  # Fall back to method of moments
        
        self.params = {'mu': mu, 'theta': max(theta, 0.01)}
        self.fitted = True
        return self.params
    
    def _mle_estimation(self, data: np.ndarray, size_factors: np.ndarray, 
                       mu_init: float, theta_init: float, max_iter: int = 20) -> Tuple[float, float]:
        """Simplified maximum likelihood estimation"""
        mu, theta = mu_init, theta_init
        
        for _ in range(max_iter):
            # Update mu
            weighted_mean = np.mean(data / size_factors)
            mu_new = max(weighted_mean, 0.001)
            
            # Update theta
            expected_counts = mu * size_factors
            residuals = (data - expected_counts) ** 2
            expected_var = expected_counts + (expected_counts ** 2) / theta
            theta_new = np.sum(expected_counts ** 2) / np.sum(residuals - expected_counts)
            theta_new = max(theta_new, 0.01)
            
            # Check convergence
            if abs(mu_new - mu) < 1e-6 and abs(theta_new - theta) < 1e-6:
                break
                
            mu, theta = mu_new, theta_new
        
        return mu, theta
    
    def pmf_or_pdf(self, x: np.ndarray, size_factors: Optional[np.ndarray] = None, **params) -> np.ndarray:
        """Calculate probability mass function"""
        mu = params.get('mu', self.params['mu'])
        theta = params.get('theta', self.params['theta'])
        
        if size_factors is None:
            size_factors = self.size_factors
        if size_factors is None:
            size_factors = np.ones(len(x))
        
        expected_counts = mu * size_factors
        r = theta
        p = r / (r + expected_counts)
        
        return stats.nbinom.pmf(x, r, p)
    
    def mean_var_relationship(self, size_factors: Optional[np.ndarray] = None, **params) -> Tuple[float, float]:
        """Return mean and variance accounting for size factors"""
        mu = params.get('mu', self.params['mu'])
        theta = params.get('theta', self.params['theta'])
        
        if size_factors is None:
            size_factors = self.size_factors
        if size_factors is None:
            size_factors = np.array([1.0])
        
        mean_size_factor = np.mean(size_factors)
        mean_count = mu * mean_size_factor
        var_count = mean_count + (mean_count ** 2) / theta
        
        return mean_count, var_count
    
    def generate_samples(self, n_samples: int, size_factors: Optional[np.ndarray] = None, **params) -> np.ndarray:
        """Generate samples from the DANB distribution"""
        mu = params.get('mu', self.params['mu'])
        theta = params.get('theta', self.params['theta'])
        
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
    
    def pearson_residuals(self, data: np.ndarray, size_factors: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate Pearson residuals for normalization"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if size_factors is None:
            size_factors = self.size_factors
        if size_factors is None:
            size_factors = np.ones(len(data))
        
        mu = self.params['mu']
        theta = self.params['theta']
        
        expected = mu * size_factors
        variance = expected + (expected ** 2) / theta
        residuals = (data - expected) / np.sqrt(variance + 1e-8)
        
        return residuals
    
    def deviance_residuals(self, data: np.ndarray, size_factors: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate deviance residuals for normalization"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if size_factors is None:
            size_factors = self.size_factors
        if size_factors is None:
            size_factors = np.ones(len(data))
        
        mu = self.params['mu']
        theta = self.params['theta']
        expected = mu * size_factors
        
        residuals = np.zeros_like(data, dtype=float)
        
        for i in range(len(data)):
            y = data[i]
            mu_i = expected[i]
            
            if y == 0:
                residuals[i] = -np.sqrt(2 * theta * np.log(1 + mu_i / theta))
            else:
                term1 = y * np.log(y / mu_i) if mu_i > 0 else 0
                term2 = (y + theta) * np.log((y + theta) / (mu_i + theta))
                deviance = 2 * (term1 - term2)
                residuals[i] = np.sign(y - mu_i) * np.sqrt(abs(deviance))
        
        return residuals
