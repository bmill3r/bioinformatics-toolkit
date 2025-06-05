# Single Cell Model Fitting & Normalization - Comprehensive Guide

A comprehensive Python framework for fitting statistical models to single-cell RNA-seq data with memory-efficient handling of large sparse matrices and multi-tier batch effect correction.

## 📑 Table of Contents

1. [Quick Start](#-quick-start)
2. [SingleCellModelFitter Class](#-singlecellmodelfitter-class)
   - [Basic Usage](#basic-usage)
   - [Memory Management](#memory-management)
   - [Normalization Methods](#normalization-methods)
   - [Batch Effect Handling](#batch-effect-handling)
3. [Standard Statistical Models](#-standard-statistical-models)
   - [Available Models](#available-standard-models)
   - [Model Selection](#model-selection)
   - [Usage Examples](#standard-model-examples)
4. [Batch-Aware Models](#-batch-aware-models)
   - [Available Models](#available-batch-aware-models)
   - [Tier System Overview](#tier-system-overview)
   - [Usage Examples](#batch-aware-examples)
5. [Adding Custom Models](#-adding-custom-models)
   - [Creating Standard Models](#creating-standard-models)
   - [Creating Batch-Aware Models](#creating-batch-aware-models)
   - [Registration and Usage](#registration-and-usage)
6. [Advanced Workflows](#-advanced-workflows)
7. [Memory Optimization](#-memory-optimization)
8. [Troubleshooting](#-troubleshooting)
9. [API Reference](#-api-reference)

---

## 🚀 Quick Start

```python
import scanpy as sc
from sc_model_fitter import SingleCellModelFitter

# Load your data
adata = sc.datasets.pbmc3k()

# Initialize the fitter
fitter = SingleCellModelFitter(adata, layer=None)  # Use adata.X

# Fit models and normalize
fitter.fit_models(['depth_adjusted_nb'], n_genes=100)
fitter.add_normalized_layer('danb_normalized', method='log_norm')

# Your adata now has normalized data in adata.layers['danb_normalized']
print(f"Available layers: {list(adata.layers.keys())}")
```

---

## 🔧 SingleCellModelFitter Class

The main interface for all single-cell modeling and normalization tasks.

### Basic Usage

```python
# Initialize with basic settings
fitter = SingleCellModelFitter(adata)

# Initialize with batch awareness
fitter = SingleCellModelFitter(adata, batch_key='sample_id')

# Initialize with specific layer
fitter = SingleCellModelFitter(adata, layer='raw', batch_key='batch')

# Check fitter status
print(fitter)
```

### Memory Management

The fitter automatically handles sparse matrices and provides memory-efficient processing:

```python
# Check memory usage
memory_info = fitter.get_memory_usage()
print(f"Total memory usage: {memory_info['total_estimated_mb']:.1f} MB")

# For large datasets (1M+ cells)
fitter = SingleCellModelFitter(adata, batch_key='sample')

# Process only highly variable genes
hvg_genes = fitter.normalize_hvg_only(
    model_name='depth_adjusted_nb',
    method='log_norm',
    n_top_genes=2000
)
```

### Normalization Methods

| Method | Preserves Sparsity | Memory Usage | Use Case |
|--------|-------------------|--------------|----------|
| `log_norm` | ✅ | ~Original | General purpose |
| `log_cpm` | ✅ | ~Original | Simple normalization |
| `size_factor` | ✅ | ~Original | Depth adjustment only |
| `regularized_log` | ✅ | ~Original | sctransform-style |
| `pearson_residuals` | ❌ | ~100x larger | Small datasets only |
| `deviance_residuals` | ❌ | ~100x larger | Small datasets only |
| `batch_corrected` | ❌ | ~100x larger | Hierarchical models only |

```python
# Sparse-compatible normalization (recommended)
normalized = fitter.normalize_expression(
    model_name='depth_adjusted_nb',
    method='log_norm',
    preserve_sparsity=True
)

# Add to AnnData layers
fitter.add_normalized_layer(
    layer_name='log_normalized',
    method='log_norm',
    preserve_sparsity=True
)
```

### Batch Effect Handling

The fitter provides three tiers of batch effect correction:

#### Tier 1: Batch-Aware Size Factors (Automatic)
```python
# Automatically enabled when batch_key is provided
fitter = SingleCellModelFitter(adata, batch_key='sample')
# Size factors are calculated within each batch and normalized
```

#### Tier 3: Hierarchical Modeling
```python
# Use hierarchical models that explicitly model batch effects
fitter.fit_models(['hierarchical_nb'], n_genes=500)
fitter.add_normalized_layer('tier3_corrected', 
                           model_name='hierarchical_nb',
                           method='batch_corrected')
```

#### Tier 4: Post-Normalization Correction
```python
# Apply batch correction after normalization
fitter.add_normalized_layer('base_normalized', method='log_norm')
fitter.correct_batch_effects(
    layer_name='base_normalized',
    corrected_layer_name='combat_corrected',
    method='combat'
)
```

---

## 📊 Standard Statistical Models

Located in `standard_models.py`, these models form the foundation of the analysis framework.

### Available Standard Models

| Model | Description | Best For | Parameters |
|-------|-------------|----------|------------|
| `PoissonModel` | Simple Poisson distribution | Very low-noise data | λ (lambda) |
| `NegativeBinomialModel` | Overdispersed count data | Most scRNA-seq datasets | μ (mu), α (alpha) |
| `ZeroInflatedNBModel` | NB with excess zeros | High dropout datasets | μ, α, π (pi) |
| `BernoulliModel` | Presence/absence | Binary analysis | p |
| `DepthAdjustedNBModel` | NB with size factors | UMI count data | μ, θ (theta) |

### Model Selection

```python
# Fit multiple models for comparison
fitter.fit_models([
    'poisson',
    'negative_binomial', 
    'depth_adjusted_nb'
], n_genes=200)

# Compare using information criteria
comparison = fitter.compare_models('aic')
print(comparison['best_model'].value_counts())

# Visualize comparison
fitter.plot_model_comparison()
```

### Standard Model Examples

#### Basic Model Fitting
```python
# Fit a simple negative binomial model
fitter.fit_models(['negative_binomial'], n_genes=100)

# Access fitted parameters
model_results = fitter.model_results['negative_binomial']
for gene, results in list(model_results.items())[:3]:
    print(f"{gene}: μ={results['params']['mu']:.3f}, α={results['params']['alpha']:.3f}")
```

#### Depth-Adjusted Model (Recommended)
```python
# Fit depth-adjusted negative binomial (handles varying sequencing depth)
fitter.fit_models(['depth_adjusted_nb'], n_genes=500)

# Normalize using Pearson residuals (for small datasets)
fitter.add_normalized_layer('danb_pearson', 
                           model_name='depth_adjusted_nb',
                           method='pearson_residuals',
                           preserve_sparsity=False)  # Residuals destroy sparsity

# Normalize using log transformation (for large datasets)
fitter.add_normalized_layer('danb_log', 
                           model_name='depth_adjusted_nb',
                           method='log_norm',
                           preserve_sparsity=True)  # Maintains sparsity
```

#### Zero-Inflation Testing
```python
# Test for zero-inflation
fitter.fit_models(['negative_binomial'], n_genes=1000)
zi_results = fitter.test_zero_inflation(threshold_ratio=1.5)

# Genes with significant zero-inflation
zi_genes = zi_results[zi_results['zero_inflated'] == True]
print(f"Zero-inflated genes: {len(zi_genes)}")

# Fit zero-inflated model to these genes
fitter.fit_models(['zero_inflated_nb'], genes=zi_genes['gene'].tolist())
```

---

## 🔄 Batch-Aware Models

Located in `batch_aware_models.py`, these advanced models explicitly handle batch effects through sophisticated statistical approaches.

### Available Batch-Aware Models

| Model | Description | Tier | Computational Cost | Best For |
|-------|-------------|------|-------------------|----------|
| `RegularizedNBModel` | sctransform-style regularization | 1 | Medium | Variance stabilization |
| `HierarchicalNBModel` | Mixed effects with batch terms | 3 | High | Multi-batch datasets |
| `MixedEffectsNBModel` | Advanced mixed effects | 3 | Very High | Specialized analysis |

### Tier System Overview

**Tier 1 (Batch-Aware Size Factors):**
- Automatic when `batch_key` is provided
- Addresses sequencing depth differences between batches
- Compatible with all models and normalization methods
- Minimal computational overhead

**Tier 3 (Hierarchical Modeling):**
- Explicitly models batch effects as random effects
- Provides batch-corrected residuals
- Higher computational cost but principled approach
- Best for complex experimental designs

**Tier 4 (Post-Normalization Correction):**
- Applied after standard normalization
- ComBat-style or simple center-scale correction
- Flexible but two-step process

### Batch-Aware Examples

#### sctransform-Style Analysis
```python
# Regularized negative binomial with global trend fitting
fitter = SingleCellModelFitter(adata, batch_key='sample')

# Fit regularized model (automatically includes Tier 1)
fitter.fit_models(['regularized_nb'], n_genes=2000)

# Normalize using regularized log transformation
fitter.add_normalized_layer('sct_normalized', 
                           model_name='regularized_nb',
                           method='regularized_log')

# Use for downstream analysis
adata.X = adata.layers['sct_normalized']
```

#### Hierarchical Batch Correction (Tier 3)
```python
# For complex batch effects (different protocols, platforms, etc.)
fitter = SingleCellModelFitter(adata, batch_key='protocol')

# Fit hierarchical model
fitter.fit_models(['hierarchical_nb'], n_genes=1000)

# Get batch effect summary
if 'hierarchical_nb' in fitter.model_results:
    example_gene = list(fitter.model_results['hierarchical_nb'].keys())[0]
    model = fitter.model_results['hierarchical_nb'][example_gene]['model']
    batch_effects = model.get_batch_effects_summary()
    print("Batch effects summary:")
    for batch, effects in batch_effects.items():
        print(f"  {batch}: {effects['percent_change']:.1f}% change")

# Apply batch correction
fitter.add_normalized_layer('hierarchical_corrected',
                           model_name='hierarchical_nb', 
                           method='batch_corrected')
```

#### Combined Approach
```python
# Use multiple tiers for maximum batch effect removal
fitter = SingleCellModelFitter(adata, batch_key='batch')

# Tier 1 + 3: Hierarchical modeling with batch-aware size factors
fitter.fit_models(['hierarchical_nb'], n_genes=1500)
fitter.add_normalized_layer('tier1_3_corrected', 
                           method='batch_corrected')

# Tier 4: Additional post-normalization correction if needed
fitter.correct_batch_effects('tier1_3_corrected',
                             'fully_corrected', 
                             method='center_scale')

# Validate correction
metrics = fitter.validate_batch_correction('log_norm', 'fully_corrected')
print(f"Batch variance reduction: {metrics['batch_variance_reduction']:.1%}")
```

#### Batch Effect Diagnosis
```python
# Diagnose batch effects before and after correction
print("=== Before Correction ===")
fitter.diagnose_batch_effects()  # Raw data

print("=== After Correction ===")
fitter.diagnose_batch_effects('fully_corrected')

# Visualize batch effects
fitter.plot_batch_effects(genes=['CD3D', 'CD79A'])  # Before
fitter.plot_batch_effects('fully_corrected', genes=['CD3D', 'CD79A'])  # After

# Get detailed batch summary
batch_summary = fitter.get_batch_summary()
print(batch_summary)
```

---

## 🔌 Adding Custom Models

The framework is designed to be extensible. You can add custom models by inheriting from `BaseModel`.

### Creating Standard Models

#### Step 1: Define Your Model Class

```python
from standard_models import BaseModel
import numpy as np
from scipy import stats
from typing import Dict, Tuple

class GammaModel(BaseModel):
    """Custom Gamma distribution model for continuous-like expression data"""
    
    def fit(self, data: np.ndarray) -> Dict[str, float]:
        """Fit gamma distribution parameters"""
        # Remove zeros (gamma doesn't support zero values)
        nonzero_data = data[data > 0]
        
        if len(nonzero_data) == 0:
            self.params = {'shape': 1.0, 'scale': 1.0}
        else:
            # Method of moments estimation
            sample_mean = np.mean(nonzero_data)
            sample_var = np.var(nonzero_data)
            
            # Gamma: mean = shape * scale, var = shape * scale^2
            if sample_var > 0 and sample_mean > 0:
                scale = sample_var / sample_mean
                shape = sample_mean / scale
            else:
                scale = 1.0
                shape = 1.0
            
            self.params = {
                'shape': max(shape, 0.1), 
                'scale': max(scale, 0.1)
            }
        
        self.fitted = True
        return self.params
    
    def pmf_or_pdf(self, x: np.ndarray, **params) -> np.ndarray:
        """Probability density function"""
        shape = params.get('shape', self.params['shape'])
        scale = params.get('scale', self.params['scale'])
        
        return stats.gamma.pdf(x, a=shape, scale=scale)
    
    def mean_var_relationship(self, **params) -> Tuple[float, float]:
        """Return theoretical mean and variance"""
        shape = params.get('shape', self.params['shape'])
        scale = params.get('scale', self.params['scale'])
        
        mean = shape * scale
        variance = shape * scale * scale
        
        return mean, variance
    
    def generate_samples(self, n_samples: int, **params) -> np.ndarray:
        """Generate random samples"""
        shape = params.get('shape', self.params['shape'])
        scale = params.get('scale', self.params['scale'])
        
        return np.random.gamma(shape, scale, n_samples)
    
    def custom_transform(self, data: np.ndarray) -> np.ndarray:
        """Custom normalization method specific to this model"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        shape = self.params['shape']
        scale = self.params['scale']
        expected = shape * scale
        
        # Custom square-root transformation
        return np.sqrt(data) - np.sqrt(expected)
```

#### Step 2: Register and Use Your Model

```python
# Register the custom model
fitter.register_custom_model('gamma', GammaModel)

# Verify registration
available_models = fitter.available_models
print("Available models:", list(available_models.keys()))

# Use your custom model
fitter.fit_models(['gamma'], n_genes=100)

# Compare with standard models
comparison_results = fitter.fit_models(['negative_binomial', 'gamma'], n_genes=200)
comparison_df = fitter.compare_models('aic')
print("Best model frequency:")
print(comparison_df['best_model'].value_counts())
```

### Creating Batch-Aware Models

For more advanced batch-aware models, inherit from `BaseModel` and implement batch handling:

```python
from batch_aware_models import BaseModel  # Import from appropriate module

class BatchAwareGammaModel(BaseModel):
    """Gamma model with batch-specific parameters"""
    
    def __init__(self, batch_labels=None):
        super().__init__()
        self.batch_labels = batch_labels
        self.batch_params = {}
    
    def fit(self, data: np.ndarray, size_factors=None, batch_labels=None) -> Dict[str, float]:
        """Fit batch-specific gamma models"""
        if batch_labels is not None:
            self.batch_labels = batch_labels
        elif self.batch_labels is None:
            # No batch information - use regular fitting
            self.batch_labels = np.zeros(len(data))
        
        unique_batches = np.unique(self.batch_labels)
        
        # Fit separate gamma model for each batch
        for batch in unique_batches:
            batch_mask = self.batch_labels == batch
            batch_data = data[batch_mask]
            
            if len(batch_data) > 5:  # Minimum data points
                # Fit gamma to this batch
                gamma_model = GammaModel()
                batch_params = gamma_model.fit(batch_data)
                self.batch_params[batch] = batch_params
        
        # Calculate global parameters (weighted average)
        if self.batch_params:
            all_shapes = [params['shape'] for params in self.batch_params.values()]
            all_scales = [params['scale'] for params in self.batch_params.values()]
            
            self.params = {
                'global_shape': np.mean(all_shapes),
                'global_scale': np.mean(all_scales),
                'batch_params': self.batch_params
            }
        else:
            self.params = {'global_shape': 1.0, 'global_scale': 1.0, 'batch_params': {}}
        
        self.fitted = True
        return self.params
    
    def pmf_or_pdf(self, x: np.ndarray, **params) -> np.ndarray:
        """Calculate probability accounting for batch-specific parameters"""
        # Implementation similar to HierarchicalNBModel
        # Use batch-specific parameters when available
        probs = np.zeros(len(x))
        
        for i in range(len(x)):
            if self.batch_labels is not None and len(self.batch_labels) > i:
                batch = self.batch_labels[i]
                if batch in self.batch_params:
                    batch_shape = self.batch_params[batch]['shape']
                    batch_scale = self.batch_params[batch]['scale']
                else:
                    batch_shape = self.params['global_shape']
                    batch_scale = self.params['global_scale']
            else:
                batch_shape = self.params['global_shape']
                batch_scale = self.params['global_scale']
            
            probs[i] = stats.gamma.pdf(x[i], a=batch_shape, scale=batch_scale)
        
        return probs
    
    # Implement other required methods...
    def mean_var_relationship(self, **params) -> Tuple[float, float]:
        shape = self.params['global_shape']
        scale = self.params['global_scale']
        return shape * scale, shape * scale * scale
    
    def generate_samples(self, n_samples: int, **params) -> np.ndarray:
        shape = self.params['global_shape']
        scale = self.params['global_scale']
        return np.random.gamma(shape, scale, n_samples)
```

### Registration and Usage

```python
# Register batch-aware custom model
fitter.register_custom_model('batch_aware_gamma', BatchAwareGammaModel)

# Use with batch data
fitter = SingleCellModelFitter(adata, batch_key='sample')
fitter.fit_models(['batch_aware_gamma'], n_genes=100)

# Access batch-specific results
if 'batch_aware_gamma' in fitter.model_results:
    example_gene = list(fitter.model_results['batch_aware_gamma'].keys())[0]
    model = fitter.model_results['batch_aware_gamma'][example_gene]['model']
    
    print("Batch-specific parameters:")
    for batch, params in model.batch_params.items():
        print(f"  Batch {batch}: shape={params['shape']:.3f}, scale={params['scale']:.3f}")
```

---

## 🧪 Advanced Workflows

### Complete sctransform-Style Pipeline
```python
def sctransform_pipeline(adata, batch_key=None, n_hvg=3000):
    """Complete sctransform-style analysis pipeline"""
    # Initialize with batch awareness
    fitter = SingleCellModelFitter(adata, batch_key=batch_key)
    
    # Fit regularized model to highly variable genes
    hvg_genes = fitter.normalize_hvg_only(
        model_name='regularized_nb',
        method='regularized_log',
        n_top_genes=n_hvg,
        layer_name='sct_normalized'
    )
    
    # Use normalized data as main expression matrix
    adata.X = adata.layers['sct_normalized']
    
    # Continue with standard scanpy workflow
    import scanpy as sc
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    return adata, hvg_genes, fitter

# Usage
adata_sct, hvg_genes, fitter = sctransform_pipeline(adata, batch_key='sample')
```

### Multi-Tier Batch Correction Pipeline
```python
def comprehensive_batch_correction(adata, batch_key, validation_genes=None):
    """Apply and validate multi-tier batch correction"""
    fitter = SingleCellModelFitter(adata, batch_key=batch_key)
    
    if validation_genes is None:
        validation_genes = ['CD3D', 'CD79A', 'NKG7']  # Common markers
    
    # Step 1: Baseline normalization
    fitter.fit_models(['depth_adjusted_nb'], n_genes=2000)
    fitter.add_normalized_layer('baseline', method='log_norm')
    
    # Step 2: Hierarchical correction (Tier 1 + 3)
    fitter.fit_models(['hierarchical_nb'], n_genes=1000)
    fitter.add_normalized_layer('hierarchical', 
                               model_name='hierarchical_nb',
                               method='batch_corrected')
    
    # Step 3: Additional ComBat correction (Tier 4)
    fitter.correct_batch_effects('hierarchical', 'final_corrected', method='combat')
    
    # Validation
    results = {}
    for layer in ['baseline', 'hierarchical', 'final_corrected']:
        batch_ratio = fitter.diagnose_batch_effects(layer)
        metrics = fitter.validate_batch_correction('baseline', layer) if layer != 'baseline' else None
        
        results[layer] = {
            'batch_effect_ratio': batch_ratio,
            'metrics': metrics
        }
    
    # Visualization
    fitter.create_summary_plots(f'./batch_correction_analysis')
    
    return fitter, results

# Usage
fitter, correction_results = comprehensive_batch_correction(adata, 'sample_id')
```

---

## 💾 Memory Optimization

### Large Dataset Strategies

#### Strategy 1: HVG-Only Processing
```python
# For datasets with 1M+ cells
fitter = SingleCellModelFitter(adata, batch_key='sample')

# Process only top variable genes
hvg_genes = fitter.normalize_hvg_only(
    model_name='depth_adjusted_nb',
    method='log_norm',
    n_top_genes=2000,
    layer_name='hvg_normalized'
)

print(f"Processed {len(hvg_genes)} HVG instead of {adata.n_vars} total genes")
```

#### Strategy 2: Gene Subset Analysis
```python
# Focus on specific pathways or gene sets
immune_genes = ['CD3D', 'CD4', 'CD8A', 'CD19', 'CD79A', 'NKG7']

# Fit models only to genes of interest
fitter.fit_models(['hierarchical_nb'], genes=immune_genes)

# Get normalized subset in sparse format
normalized_immune, valid_genes = fitter.get_sparse_normalized_subset(
    genes=immune_genes,
    method='log_norm'
)

print(f"Memory usage: {normalized_immune.data.nbytes / 1024**2:.1f} MB")
print(f"Sparsity: {normalized_immune.nnz / normalized_immune.size:.3f}")
```

#### Strategy 3: Chunked Processing
```python
# Process large gene sets in chunks
all_genes = adata.var_names.tolist()
chunk_size = 500

for i in range(0, len(all_genes), chunk_size):
    chunk_genes = all_genes[i:i+chunk_size]
    print(f"Processing chunk {i//chunk_size + 1}: genes {i}-{i+len(chunk_genes)}")
    
    # Fit models to chunk
    fitter.fit_models(['depth_adjusted_nb'], 
                     genes=chunk_genes,
                     chunk_size=100)
    
    # Normalize chunk
    fitter.add_normalized_layer(f'chunk_{i//chunk_size}', 
                               genes=chunk_genes,
                               method='log_norm')

# Combine chunks if needed
print("All chunks processed successfully")
```

### Memory Monitoring
```python
# Monitor memory usage throughout analysis
def print_memory_usage(fitter, step_name):
    memory_info = fitter.get_memory_usage()
    print(f"{step_name}: {memory_info['total_estimated_mb']:.1f} MB")

# Usage
fitter = SingleCellModelFitter(adata)
print_memory_usage(fitter, "Initial")

fitter.fit_models(['depth_adjusted_nb'], n_genes=1000)
print_memory_usage(fitter, "After model fitting")

fitter.add_normalized_layer('normalized', method='log_norm')
print_memory_usage(fitter, "After normalization")

# Clean up when done
fitter.cleanup(keep_normalized_layers=True)
print_memory_usage(fitter, "After cleanup")
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Memory Errors
```python
# Issue: Memory error when using residual methods
try:
    fitter.normalize_expression(method='pearson_residuals')
except MemoryError:
    print("Switching to sparse-compatible method")
    normalized = fitter.normalize_expression(method='log_norm', preserve_sparsity=True)
```

#### Model Fitting Failures
```python
# Check which genes failed to fit
gene_list = ['CD3D', 'CD79A', 'INVALID_GENE']
fitter.fit_models(['depth_adjusted_nb'], genes=gene_list)

fitted_genes = list(fitter.model_results['depth_adjusted_nb'].keys())
failed_genes = [g for g in gene_list if g not in fitted_genes]

print(f"Successfully fitted: {len(fitted_genes)} genes")
print(f"Failed to fit: {failed_genes}")
```

#### Batch Effect Issues
```python
# Diagnose if batch correction is working
def validate_batch_correction_pipeline(fitter, layers_to_check):
    """Comprehensive batch effect validation"""
    results = {}
    
    for layer in layers_to_check:
        try:
            ratio = fitter.diagnose_batch_effects(layer)
            results[layer] = {
                'batch_effect_ratio': ratio,
                'status': 'Strong' if ratio > 0.1 else 'Moderate' if ratio > 0.05 else 'Minimal'
            }
        except Exception as e:
            results[layer] = {'error': str(e)}
    
    return results

# Usage
validation_results = validate_batch_correction_pipeline(
    fitter, 
    ['baseline', 'batch_corrected', 'combat_corrected']
)

for layer, result in validation_results.items():
    if 'error' in result:
        print(f"{layer}: Error - {result['error']}")
    else:
        print(f"{layer}: {result['status']} batch effects (ratio: {result['batch_effect_ratio']:.3f})")
```

#### Sparse Matrix Issues
```python
# Ensure data is properly formatted
def check_data_format(adata):
    """Validate AnnData format for the fitter"""
    issues = []
    
    # Check if data is sparse
    if not sparse.issparse(adata.X):
        issues.append("Data is not sparse - consider converting: adata.X = sparse.csr_matrix(adata.X)")
    
    # Check for negative values
    if hasattr(adata.X, 'data'):
        if (adata.X.data < 0).any():
            issues.append("Data contains negative values")
    else:
        if (adata.X < 0).any():
            issues.append("Data contains negative values")
    
    # Check data type
    if adata.X.dtype not in [np.int32, np.int64, np.float32, np.float64]:
        issues.append(f"Unusual data type: {adata.X.dtype}")
    
    if issues:
        print("Data format issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Data format looks good")
    
    return len(issues) == 0

# Usage
check_data_format(adata)
```

---

## 📖 API Reference

### SingleCellModelFitter Class Methods

#### Initialization
- `__init__(adata, layer=None, batch_key=None)`

#### Model Fitting
- `fit_models(models, genes=None, n_genes=100, chunk_size=100)`
- `compare_models(criterion='aic')`
- `register_custom_model(name, model_class)`

#### Normalization
- `normalize_expression(model_name, method='log_norm', genes=None)`
- `add_normalized_layer(layer_name, model_name, method='log_norm')`
- `normalize_hvg_only(model_name, method='log_norm', n_top_genes=2000)`
- `get_sparse_normalized_subset(genes, model_name, method='log_norm')`

#### Batch Effects
- `correct_batch_effects(layer_name, method='combat', batch_key=None)`
- `plot_batch_effects(layer_name=None, genes=None)`
- `validate_batch_correction(original_layer, corrected_layer)`
- `diagnose_batch_effects(layer_name=None)`
- `get_batch_summary()`

#### Visualization
- `plot_mean_variance_relationship()`
- `plot_model_comparison(criterion='aic')`
- `plot_qq_plots(genes, models=None)`
- `plot_normalization_effects(genes)`
- `create_summary_plots(output_dir='./plots')`

#### Analysis & Diagnostics
- `get_gene_statistics(genes=None)`
- `test_zero_inflation(threshold_ratio=1.5)`
- `generate_diagnostic_report(output_file=None)`

#### Export & I/O
- `export_normalized_data(layer_name, output_file, format='h5ad', genes=None)`
- `save_model_results(output_file)`
- `load_model_results(input_file)`

#### Utilities
- `get_memory_usage()`
- `get_available_methods()`
- `cleanup(keep_normalized_layers=True)`
- `reset()`
- `get_help(method_name=None)`

### Standard Models

All models inherit from `BaseModel` and implement:
- `fit(data)` - Fit model parameters
- `pmf_or_pdf(x, **params)` - Probability function
- `mean_var_relationship(**params)` - Theoretical moments
- `generate_samples(n_samples, **params)` - Sample generation
- `aic(data)` - Akaike Information Criterion
- `bic(data)` - Bayesian Information Criterion

Additional methods for `DepthAdjustedNBModel`:
- `pearson_residuals(data, size_factors=None)`
- `deviance_residuals(data, size_factors=None)`

### Batch-Aware Models

Additional methods for `RegularizedNBModel`:
- `fit_global_trend(data_matrix, size_factors, n_genes_sample=1000)`
- `regularized_log_transform(data, size_factors=None, clip_value=30.0)`

Additional methods for `HierarchicalNBModel`:
- `batch_corrected_residuals(data, size_factors=None, batch_labels=None)`
- `get_batch_effects_summary()`

---

## 🎯 Best Practices Summary

### Model Selection Guidelines
1. **Start with `depth_adjusted_nb`** for most scRNA-seq datasets
2. **Use `regularized_nb`** for sctransform-style variance stabilization  
3. **Use `hierarchical_nb`** when you have multiple batches with strong effects
4. **Compare models** with `compare_models()` before proceeding
5. **Test for zero-inflation** if you suspect high dropout rates

### Normalization Strategy
1. **Prefer sparse-compatible methods** (`log_norm`, `regularized_log`) for large datasets
2. **Use residual methods** (`pearson_residuals`, `batch_corrected`) only for small datasets or HVG subsets
3. **Always validate** batch correction with diagnostic plots and metrics
4. **Consider the downstream analysis** when choosing normalization methods

### Batch Effect Handling
1. **Always use Tier 1** (batch-aware size factors) when you have multiple samples
2. **Use Tier 3** (hierarchical models) for complex experimental designs
3. **Apply Tier 4** (post-correction) if batch effects remain after other methods
4. **Validate thoroughly** - batch correction can over-correct and remove biological signal

### Memory Management
1. **Use `preserve_sparsity=True`** whenever possible
2. **Process HVG only** for initial exploration of large datasets
3. **Use chunked processing** for memory-constrained environments
4. **Monitor memory usage** with `get_memory_usage()`
5. **Clean up** with `cleanup()` when analysis is complete

### Quality Control Workflow
```python
def quality_control_workflow(fitter):
    """Comprehensive QC for model fitting results"""
    print("=== Quality Control Report ===")
    
    # 1. Check data format
    data_matrix = fitter.get_data_matrix()
    sparsity = data_matrix.nnz / data_matrix.size if sparse.issparse(data_matrix) else 0
    print(f"Data sparsity: {sparsity:.3f}")
    
    # 2. Model fitting success rate
    total_genes_attempted = 0
    total_genes_fitted = 0
    
    for model_name, model_data in fitter.model_results.items():
        genes_fitted = len(model_data)
        total_genes_fitted += genes_fitted
        print(f"{model_name}: {genes_fitted} genes fitted")
    
    # 3. Batch effect analysis (if applicable)
    if fitter.batch_key:
        try:
            batch_ratio = fitter.diagnose_batch_effects()
            batch_summary = fitter.get_batch_summary()
            print(f"Batch effect strength: {batch_ratio:.3f}")
            print(f"Number of batches: {len(batch_summary)}")
        except:
            print("Could not analyze batch effects")
    
    # 4. Memory usage
    memory_info = fitter.get_memory_usage()
    print(f"Current memory usage: {memory_info['total_estimated_mb']:.1f} MB")
    
    # 5. Generate comprehensive report
    try:
        report = fitter.generate_diagnostic_report()
        print("\n" + "="*50)
        print(report)
    except Exception as e:
        print(f"Could not generate full report: {e}")

# Usage
quality_control_workflow(fitter)
```

---

## 📚 Example Notebooks & Workflows

### Complete Analysis Example
```python
import scanpy as sc
import pandas as pd
from sc_model_fitter import SingleCellModelFitter

# 1. Load and prepare data
adata = sc.datasets.pbmc3k()
adata.obs['batch'] = ['batch_A'] * 1000 + ['batch_B'] * 1000 + ['batch_C'] * (len(adata) - 2000)

print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"Batches: {adata.obs['batch'].value_counts()}")

# 2. Initialize fitter with batch awareness
fitter = SingleCellModelFitter(adata, batch_key='batch')
print(fitter)

# 3. Model comparison
print("\n=== Model Comparison ===")
fitter.fit_models(['negative_binomial', 'depth_adjusted_nb', 'hierarchical_nb'], n_genes=500)
comparison = fitter.compare_models('aic')
print("Best model frequency:")
print(comparison['best_model'].value_counts())

# 4. Normalization with batch correction
print("\n=== Normalization ===")
fitter.add_normalized_layer('log_normalized', 
                           model_name='depth_adjusted_nb', 
                           method='log_norm')

fitter.add_normalized_layer('batch_corrected',
                           model_name='hierarchical_nb',
                           method='batch_corrected')

# 5. Batch effect validation
print("\n=== Batch Effect Analysis ===")
print("Before correction:")
ratio_before = fitter.diagnose_batch_effects('log_normalized')

print("After correction:")
ratio_after = fitter.diagnose_batch_effects('batch_corrected')

metrics = fitter.validate_batch_correction('log_normalized', 'batch_corrected')

# 6. Export results
print("\n=== Export ===")
fitter.export_normalized_data('batch_corrected', 'normalized_data.h5ad')
fitter.save_model_results('model_results.pkl')
fitter.create_summary_plots('./analysis_plots')

# 7. Generate comprehensive report
report = fitter.generate_diagnostic_report('analysis_report.txt')
print("Analysis complete! Check 'analysis_report.txt' for full results.")
```

This comprehensive guide provides everything needed to effectively use the Single Cell Model Fitting framework, from basic usage to advanced custom model development. The modular design allows users to start simple and gradually incorporate more sophisticated analyses as needed.

---

## 🔗 Quick Reference Links

- **Getting Started**: [Quick Start](#-quick-start)
- **Main Interface**: [SingleCellModelFitter Class](#-singlecellmodelfitter-class)  
- **Basic Models**: [Standard Statistical Models](#-standard-statistical-models)
- **Advanced Models**: [Batch-Aware Models](#-batch-aware-models)
- **Customization**: [Adding Custom Models](#-adding-custom-models)
- **Large Data**: [Memory Optimization](#-memory-optimization)
- **Help**: [Troubleshooting](#-troubleshooting)

For additional help, use the built-in help system:

```python
fitter.get_help()  # Overview of all methods
fitter.get_help('fit_models')  # Specific method help
```

---

## 📄 Citation

If you use this framework in your research, please cite:

```
Single Cell Model Fitting Framework
A comprehensive Python toolkit for statistical modeling and normalization 
of single-cell RNA-seq data with multi-tier batch effect correction.
```

---

## 🤝 Contributing

This framework is designed to be extensible. Contributions are welcome:

1. **Bug Reports**: Report issues with specific error messages and minimal reproducible examples
2. **Feature Requests**: Suggest new models or normalization methods
3. **Custom Models**: Share useful custom model implementations
4. **Documentation**: Improve examples and explanations

---

## 📜 License

This framework is provided for research and educational purposes. Please respect the licenses of underlying dependencies (scipy, numpy, pandas, scanpy, etc.).

---

## 🔧 Requirements

```python
# Core dependencies
numpy >= 1.19.0
scipy >= 1.7.0
pandas >= 1.3.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
tqdm >= 4.60.0

# Single-cell specific
scanpy >= 1.8.0
anndata >= 0.8.0

# Optional for advanced features
numba >= 0.54.0  # For faster computation
```

Install with:
```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn tqdm scanpy anndata
```
