# Comprehensive Python Tutorial for Single-Cell and Spatial Data Analysis

This tutorial provides a step-by-step guide for analyzing single-cell and spatial transcriptomics data, from raw data to advanced visualizations using our custom toolkit.

## Table of Contents
1. [Setting Up the Environment](#1-setting-up-the-environment)
2. [Loading Different Data Types](#2-loading-different-data-types)
3. [Quality Control](#3-quality-control)
4. [Filtering and Data Preprocessing](#4-filtering-and-data-preprocessing)
5. [Normalization](#5-normalization)
6. [Feature Selection](#6-feature-selection)
7. [Dimensionality Reduction](#7-dimensionality-reduction)
8. [Visualization](#8-visualization)
9. [Spatial Data Analysis](#9-spatial-data-analysis)
10. [Gene Set Scoring](#10-gene-set-scoring)
11. [Advanced Analyses](#11-advanced-analyses)
12. [Saving and Loading Results](#12-saving-and-loading-results)

## 1. Setting Up the Environment

First, let's import all necessary modules from our toolkit:

```python
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from sctools.utils.s3_utils import S3Utils
from sctools.qc import SingleCellQC
from sctools.normalization import Normalization
from sctools.feature_selection import FeatureSelection
from sctools.dim_reduction import DimensionalityReduction
from sctools.visualization import EnhancedVisualization
from sctools.spatial import SpatialAnalysis
from sctools.geneset import GeneSetScoring

# Set random seed for reproducibility
np.random.seed(42)
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=100, frameon=False)
```

## 2. Loading Different Data Types

Our toolkit supports various input data types, including direct loading from AWS S3 buckets. Here are examples for different data formats:

### Loading from S3

```python
# Initialize S3 utilities with a specific profile
s3 = S3Utils(profile_name="my-profile")

# List available buckets
buckets = s3.list_buckets()
print(f"Available S3 buckets: {buckets}")

# Initialize QC object
qc = SingleCellQC(verbose=True)

# Load a 10X dataset from S3
adata = s3.read_10x_mtx("my-bucket", "path/to/filtered_feature_bc_matrix/")
qc.load_data(adata)

# Alternatively, load an H5AD file from S3
adata = s3.read_h5ad("my-bucket", "path/to/data.h5ad")
qc.load_data(adata)

print(f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")
```

### Loading Local Files

```python
# Initialize QC object
qc = SingleCellQC(verbose=True)

# 10X dataset (filtered_feature_bc_matrix directory)
adata = qc.load_data("path/to/filtered_feature_bc_matrix/")

# H5AD file
adata = qc.load_data("path/to/data.h5ad")

# CSV file (cells as rows, genes as columns)
adata = qc.load_data("path/to/expression_matrix.csv")

# CSV file with transposition (if genes are rows)
adata = qc.load_data("path/to/expression_matrix.csv", transpose=True)

# AnnData object directly
adata = sc.read_h5ad("path/to/data.h5ad")
qc.load_data(adata)

print(f"Loaded dataset with {qc.adata.n_obs} cells and {qc.adata.n_vars} genes")
```

### Loading Spatial Data

```python
# Load 10X Visium data
adata = qc.load_data("path/to/visium_data/")

# Verify spatial coordinates are present
if 'spatial' in adata.obsm:
    print("Spatial coordinates found")
else:
    print("No spatial coordinates found")
```

## 3. Quality Control

Quality control is a critical first step in any single-cell analysis. Our `SingleCellQC` module provides comprehensive QC metrics and visualizations to help identify low-quality cells and genes.

### Calculate Basic QC Metrics

```python
# Calculate QC metrics with default parameters
qc.calculate_qc_metrics(
    min_genes=200,         # Min genes per cell
    min_cells=3,           # Min cells per gene
    percent_mito='MT-',    # Mitochondrial gene prefix (human)
    # percent_mito='mt-'   # For mouse data
    percent_ribo='RP[SL]'  # Ribosomal gene pattern
)

# View QC metrics summary
print(qc.adata.obs[['n_genes_by_counts', 'total_counts', 'percent_mito']].describe())
```

### Visualize QC Metrics

```python
# Generate publication-quality static plots
qc.plot_qc_metrics(figsize=(20, 12), save_path="qc_metrics.png")

# Or generate interactive plots with plotly
qc.plot_qc_metrics(use_plotly=True, save_path="interactive_qc.html")
```

### Expert Tips: Interpreting QC Metrics

When analyzing QC metrics as an experienced bioinformatician, you should look for:

1. **Distribution of genes per cell**: A bimodal distribution often indicates two populations of cells - one high quality and one low quality or empty droplets. You want a relatively normal distribution with a clear peak.

2. **UMI counts per cell**: Similar to genes per cell, should show a single clear peak. Very low counts indicate degraded RNA or empty droplets.

3. **Mitochondrial content**: High mitochondrial percentage (>20-25%) typically indicates stressed or dying cells. However, be cautious as some cell types naturally have higher mitochondrial content.

4. **Relationship between genes and UMIs**: There should be a strong positive correlation. Outliers may indicate technical artifacts.

5. **Genes expressed across cells**: Genes detected in very few cells may be noise or rare transcripts. Be careful about filtering them out if you're looking for rare cell types.

Adjust thresholds based on experimental context - thresholds for a FACS-sorted sample will differ from those for a droplet-based system.

## 4. Filtering and Data Preprocessing

Based on the QC metrics, we can filter out low-quality cells and genes.

### Determine Filtering Thresholds

```python
# Get recommended thresholds based on distribution
thresholds = qc.get_qc_thresholds(
    n_mads=5.0,        # Number of MADs for outlier detection
    max_mito=20.0      # Max mitochondrial percentage
)

print("Recommended filtering thresholds:")
for key, value in thresholds.items():
    print(f"{key}: {value}")
```

### Filter Cells

```python
# Filter cells with automatic thresholds
qc.filter_cells()

# OR filter with custom thresholds
qc.filter_cells(
    min_genes=500,     # Minimum genes per cell
    max_genes=6000,    # Maximum genes per cell
    min_counts=1000,   # Minimum counts per cell
    max_counts=30000,  # Maximum counts per cell
    max_mito=15.0      # Maximum mitochondrial percentage
)

print(f"After filtering: {qc.adata.n_obs} cells and {qc.adata.n_vars} genes")

# Compute QC stats after filtering
filtered_stats = qc.compute_summary_statistics()
print(filtered_stats)
```

### Expert Tips: Cell Filtering Decisions

Experienced bioinformaticians approach filtering with these considerations:

1. **Balance stringency with data preservation**: Overly aggressive filtering can remove valid rare cell populations. Start conservative and iterate if needed.

2. **Consider dataset-specific factors**: Fresh tissue samples can tolerate stricter thresholds than frozen samples, which naturally have more degradation.

3. **Project goals matter**: If you're looking for well-characterized cell types, aggressive filtering is acceptable. For discovery of rare populations, be more permissive.

4. **Technical factors**: Consider sequencing depth when setting count thresholds - deeply sequenced libraries require different thresholds than shallowly sequenced ones.

5. **Adaptive thresholds**: For datasets with multiple samples, consider sample-specific thresholds rather than global ones to account for technical variation between samples.

## 5. Normalization

Normalization corrects for technical variations in sequencing depth between cells.

```python
# Initialize normalization with filtered AnnData
norm = Normalization(qc.adata)

# Standard log normalization
norm.log_norm(
    scale_factor=10000,  # Scale factor for normalization
    log_base=2,          # Base for logarithm
    inplace=True         # Modify data in place
)

# Alternative: SCTransform (variance stabilizing transformation)
try:
    norm.sctransform(inplace=True)
    print("SCTransform normalization completed")
except Exception as e:
    print(f"SCTransform failed: {e}. Falling back to log normalization.")
    norm.log_norm(inplace=True)

# Alternative: Centered log-ratio normalization (for compositional data)
# norm.clr_norm(inplace=True)
```

### Expert Tips: Choosing Normalization Methods

As an experienced bioinformatician, consider these factors when choosing a normalization method:

1. **Log normalization**: Simple, fast, and robust. Good default choice for most datasets, especially when computational resources are limited.

2. **SCTransform**: Better at handling technical noise and variations in sequencing depth. Preferable for datasets with significant batch effects or varying capture efficiencies.

3. **CLR normalization**: Best for compositional data when the relative abundances between genes are more important than absolute values.

4. **Different cell types may require different strategies**: Some cell types have dramatically different RNA content - consider whether global normalization is appropriate.

5. **Normalization influences all downstream analyses**: Different methods can yield different results in clustering and differential expression. When in doubt, try multiple methods and compare results.

## 6. Feature Selection

Identifying highly variable genes focuses the analysis on biologically informative features.

```python
# Initialize feature selection with normalized AnnData
fs = FeatureSelection(norm.adata)

# Find highly variable genes
fs.find_highly_variable_genes(
    method='seurat',    # Method: 'seurat', 'cell_ranger', 'seurat_v3', 'dispersion'
    n_top_genes=2000,   # Number of HVGs to select
    min_mean=0.0125,    # Minimum mean expression
    max_mean=3,         # Maximum mean expression
    min_disp=0.5,       # Minimum dispersion
    batch_key=None,     # If you have batch information
    subset=False        # Whether to subset to HVGs immediately
)

# Plot distribution of highly variable genes
fs.plot_highly_variable_genes(save_path="hvg_plot.png")

# Get number of highly variable genes
n_hvgs = sum(fs.adata.var.highly_variable)
print(f"Selected {n_hvgs} highly variable genes")

# Optional: Rank genes to find markers for cell types if we have annotations
if 'cell_type' in fs.adata.obs:
    fs.rank_genes_groups(groupby='cell_type', method='wilcoxon')
    fs.plot_ranked_genes(n_genes=5, save_path="top_markers.png")
    top_markers = fs.get_top_ranked_genes(n_genes=10)
    print("Top marker genes per cell type:")
    print(top_markers)
```

### Expert Tips: Feature Selection Strategies

Considerations when selecting features:

1. **Number of HVGs matters**: 1,000-3,000 genes is typically a good balance. Too few may miss biological signals; too many include noise.

2. **Method selection**: 'seurat_v3' typically performs best for droplet-based data; 'dispersion' for well-based methods.

3. **Expression thresholds**: Check the mean-variance plot and adjust min_mean/max_mean to capture the most informative part of the gene expression distribution.

4. **Incorporate prior knowledge**: Consider adding known cell type markers even if they're not highly variable to improve cell type identification.

5. **Balance between variance and mean expression**: Very highly expressed genes often show high variance due to technical factors rather than biological interest.

## 7. Dimensionality Reduction

Dimensionality reduction techniques help visualize and analyze high-dimensional gene expression data.

```python
# Initialize dimensionality reduction with feature-selected AnnData
dr = DimensionalityReduction(fs.adata)

# Run PCA
dr.run_pca(
    n_comps=50,                  # Number of PCs to compute
    use_highly_variable=True,    # Use only HVGs
    random_state=42              # Random seed
)

# Plot PCA variance explained
dr.plot_pca_variance(
    n_pcs=50,                    # Number of PCs to plot
    threshold=0.9,               # Mark 90% variance threshold
    save_path="pca_variance.png" # Save path
)

# Compute neighbors graph (needed for UMAP and clustering)
sc.pp.neighbors(
    dr.adata, 
    n_neighbors=15,    # Number of neighbors
    n_pcs=30,          # Number of PCs to use
    metric='euclidean' # Distance metric
)

# Run UMAP
dr.run_umap(
    n_components=2,    # Output dimensions
    min_dist=0.3,      # Minimum distance between points
    spread=1.0,        # Spread of the embedding
    random_state=42    # Random seed
)

# Optionally run t-SNE
dr.run_tsne(
    n_components=2,    # Output dimensions
    perplexity=30.0,   # Perplexity parameter
    random_state=42    # Random seed
)

# Run clustering (Leiden algorithm)
dr.run_clustering(
    method='leiden',
    resolution=1.0,
    random_state=42
)

print(f"Identified {dr.adata.obs['leiden'].nunique()} clusters")
```

### Expert Tips: Dimensionality Reduction Best Practices

From an expert's perspective:

1. **PCA components**: Examine the elbow plot to determine how many PCs to retain. Typically 15-50 PCs capture the majority of biological variation while removing technical noise.

2. **UMAP parameters**: `min_dist` controls cluster tightness (lower values = tighter clusters). Start with default (0.5) and adjust based on your data structure. Increase if clusters appear artificially separated; decrease if there's too much mixing.

3. **Balance local and global structure**: UMAP preserves local structure better than t-SNE but can sometimes exaggerate separation. Consider showing both visualizations for important findings.

4. **Neighbors choice**: The `n_neighbors` parameter influences how much global structure is preserved. Higher values (30-50) preserve more global structure but may blur local patterns.

5. **Resolution parameter**: For clustering, start with resolution=1.0 and adjust based on biological knowledge. Higher values give more clusters; compare with known markers to validate cluster granularity.

## 8. Visualization

Visualizing your data is crucial for interpretation and communication of results.

```python
# Initialize visualization with processed AnnData
viz = EnhancedVisualization(dr.adata)

# Create UMAP visualization colored by clusters
viz.scatter(
    x='X_umap-0',              # UMAP first dimension
    y='X_umap-1',              # UMAP second dimension  
    color='leiden',            # Color by cluster
    title='UMAP - Cell Clusters',
    size=5.0,                  # Point size
    save_path="umap_clusters.png" # Save path
)

# Plot UMAP colored by QC metrics
viz.scatter(
    x='X_umap-0', 
    y='X_umap-1',
    color='n_genes_by_counts',
    title='UMAP - Genes per Cell',
    cmap='viridis',
    size=5.0,
    save_path="umap_genes.png"
)

viz.scatter(
    x='X_umap-0', 
    y='X_umap-1',
    color='percent_mito',
    title='UMAP - Mitochondrial Percentage',
    cmap='Reds',
    size=5.0,
    save_path="umap_mito.png"
)

# Plot gene expression for multiple genes
marker_genes = ['CD3E', 'CD4', 'CD8A', 'MS4A1', 'CD19', 'FCGR3A', 'CD14']
viz.plot_gene_expression(
    genes=marker_genes,
    basis='umap',
    ncols=4,
    color_map='viridis',
    save_path="marker_genes_umap.png"
)

# For large datasets, use fast plotting with datashader
if dr.adata.n_obs > 10000:
    viz.scatter_fast(
        x='X_umap-0',
        y='X_umap-1',
        color='leiden',
        title='UMAP - Cell Clusters (Fast Rendering)',
        save_path="umap_fast.html"
    )
```

### Expert Tips: Creating Effective Visualizations

Guidelines for creating informative visualizations:

1. **Color schemes**: Use colorblind-friendly palettes (viridis, plasma) for continuous variables. For categorical data (clusters), ensure adequate color separation.

2. **Data exploration sequence**: First visualize QC metrics on your embedding to ensure technical artifacts aren't driving clustering. Then examine known marker genes to validate cell type expectations.

3. **Multi-panel figures**: Combine related plots (different genes or metrics) in a single figure for easier comparison. Use a consistent layout and embedding across panels.

4. **Point size and transparency**: For large datasets, reduce point size and increase transparency to reveal density patterns.

5. **Publication-ready figures**: For publications, use consistent fonts, appropriate axis labels, and maintain aspect ratios. Add scale bars for spatial plots.

## 9. Spatial Data Analysis

For spatial transcriptomics data, we can analyze gene expression in the context of spatial coordinates.

```python
# Check if spatial coordinates exist
if 'spatial' in dr.adata.obsm:
    # Initialize spatial analysis
    spatial = SpatialAnalysis(
        dr.adata,
        spatial_key='spatial'  # Key in obsm with spatial coordinates
    )
    print("Spatial analysis initialized")
    
    # Visualize spatial distribution of clusters
    spatial.plot_spatial_gene_expression(
        genes=['leiden'],
        cmap='tab20',
        size=10.0,
        show_colorbar=True,
        save_path="spatial_clusters.png"
    )
    
    # Visualize expression of marker genes in space
    spatial.plot_spatial_gene_expression(
        genes=marker_genes[:4],  # First 4 marker genes
        ncols=2,                 # Number of columns in grid
        figsize=(12, 10),
        cmap='viridis',
        size=8.0,
        save_path="spatial_markers.png"
    )
    
    # Create spatial bins to reduce noise
    binned_adata = spatial.create_spatial_grid(
        bin_size=100.0,         # Size of square bins
        aggr_func='mean',       # Aggregation function
        min_cells=5             # Minimum cells per bin
    )
    
    print(f"Created {binned_adata.n_obs} spatial bins")
    
    # Initialize spatial analysis for binned data
    binned_spatial = SpatialAnalysis(binned_adata)
    
    # Visualize binned data
    binned_spatial.plot_spatial_gene_expression(
        genes=['n_cells'] + marker_genes[:3],  # Number of cells per bin + markers
        ncols=2,
        figsize=(12, 10),
        cmap='viridis',
        size=25.0,  # Larger points for bins
        save_path="spatial_binned.png"
    )
    
    # Calculate spatial statistics (Moran's I) - identifies genes with spatial patterns
    try:
        moran_results = spatial.calculate_moran_i(
            max_genes=500,        # Limit for computational efficiency
            n_jobs=4              # Parallel jobs
        )
        
        # Save results
        moran_results.to_csv("moran_i_results.csv")
        
        # Show top spatially patterned genes
        print("Top spatially patterned genes:")
        print(moran_results.head(10))
        
        # Plot top spatially patterned genes
        top_spatial_genes = moran_results.head(4)['gene'].tolist()
        spatial.plot_spatial_gene_expression(
            genes=top_spatial_genes,
            ncols=2,
            figsize=(12, 10),
            cmap='viridis',
            size=8.0,
            save_path="top_spatial_genes.png"
        )
    except Exception as e:
        print(f"Moran's I calculation failed: {e}")
        
    # Find spatial domains (regions with similar expression patterns)
    spatial.find_spatial_domains(
        n_clusters=10,             # Target number of domains
        method='leiden',           # Clustering method
        resolution=1.0             # Resolution parameter
    )
    
    # Visualize spatial domains
    spatial.plot_spatial_gene_expression(
        genes=['spatial_domains'],
        cmap='tab20',
        size=10.0,
        save_path="spatial_domains.png"
    )
else:
    print("No spatial coordinates found. Skipping spatial analysis.")
```

### Expert Tips: Spatial Data Analysis

Insights from experienced spatial transcriptomics analysts:

1. **Spatial resolution considerations**: The biological interpretation depends heavily on the resolution of your spatial technology - Visium (~100μm spots) captures tissue regions while MERFISH can resolve single cells.

2. **Binning strategies**: Choose bin size based on your biological question. Smaller bins preserve spatial resolution but have higher noise; larger bins reduce noise but may mask fine spatial patterns.

3. **Spatial statistics interpretation**: Moran's I ranges from -1 (dispersed) to 1 (clustered). Values near zero indicate random spatial distribution. Focus on genes with both high Moran's I and low p-values.

4. **Tissue architecture**: Compare spatial domains with histological images to connect molecular profiles with tissue structures.

5. **Comparison with non-spatial data**: When available, integrate with higher-resolution non-spatial scRNA-seq from similar tissue to improve cell type annotation.

## 10. Gene Set Scoring

Gene set scoring evaluates the activity of gene programs across cells.

```python
# Initialize gene set scoring
gss = GeneSetScoring(dr.adata)

# Define gene sets (marker genes for different cell types)
t_cell_markers = ['CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD4']
b_cell_markers = ['CD79A', 'CD79B', 'MS4A1', 'CD19']
myeloid_markers = ['LYZ', 'CST3', 'CD14', 'FCGR3A']

# Add gene sets
gss.add_gene_set("T_cells", t_cell_markers, "T cell markers")
gss.add_gene_set("B_cells", b_cell_markers, "B cell markers")
gss.add_gene_set("Myeloid", myeloid_markers, "Myeloid cell markers")

# Score gene sets using AUCell
gss.score_aucell()

# Alternatively, use Scanpy's scoring method
gss.score_scanpy()

# Visualize gene set scores on UMAP
gss.plot_gene_set_scores(
    gene_set="T_cells",
    score_type="AUCell_",
    groupby="leiden",
    save_path="t_cell_score.png"
)

# Compare scores across clusters with violin plots
gss.plot_gene_set_violin(
    gene_sets=["T_cells", "B_cells", "Myeloid"],
    groupby="leiden",
    save_path="cell_type_scores.png"
)

# Get enrichment statistics for each group
enrichment_stats = gss.get_gene_set_enrichment_by_group(
    gene_set="T_cells",
    groupby="leiden"
)
print("T cell enrichment by cluster:")
print(enrichment_stats)

# Find cells with high scores for a gene set
t_cell_mask = gss.find_cells_with_high_scores(
    gene_set="T_cells",
    percentile=95  # Top 5% of cells
)
print(f"Found {sum(t_cell_mask)} potential T cells")

# Cluster cells based on gene set scores
gss.cluster_by_gene_sets(
    gene_sets=["T_cells", "B_cells", "Myeloid"],
    n_clusters=5,
    cluster_key="cell_type_clusters"
)

# Visualize cell type clusters
viz.scatter(
    x='X_umap-0',
    y='X_umap-1',
    color='cell_type_clusters',
    title='Cell Types Based on Marker Genes',
    save_path="cell_type_clusters.png"
)
```

### Expert Tips: Gene Set Scoring Analysis

Key insights for effective gene set scoring:

1. **Gene set selection**: Include genes with specific biological relevance. Avoid general housekeeping genes that would be expressed across all cells.

2. **Scoring method comparison**: AUCell performs better for rare cell populations, while Scanpy's method works well for common cell types. When possible, compare both.

3. **Score interpretation**: Absolute scores can vary across experiments - focus on relative enrichment patterns across different cell groups.

4. **Validation**: Always validate computational cell type assignments with orthogonal evidence - either known markers or functional assays.

5. **Cutoff determination**: Rather than using a fixed percentile threshold, examine the distribution of scores to identify natural breakpoints that separate positive and negative populations.

## 11. Advanced Analyses

Here are some advanced analyses you might want to perform after completing the basic workflow.

### Trajectory Analysis

```python
import scanpy.external as sce

# Run diffusion map (foundation for trajectory analysis)
sc.tl.diffmap(dr.adata)

# Run PAGA for trajectory inference
sc.tl.paga(dr.adata, groups='leiden')

# Plot PAGA graph
sc.pl.paga(dr.adata, save='paga.png')

# Run pseudotime analysis with diffusion pseudotime
sce.tl.dpt(dr.adata, n_dcs=15)

# Visualize pseudotime on UMAP
viz.scatter(
    x='X_umap-0',
    y='X_umap-1',
    color='dpt_pseudotime',
    title='Pseudotime Trajectory',
    save_path="pseudotime.png"
)
```

### RNA Velocity Analysis (requires velocyto/scVelo data)

```python
import scvelo as scv

# Load RNA velocity data (assuming it's available)
try:
    # Read velocyto data
    velocity_adata = scv.read('path/to/velocyto_data.loom')
    
    # Merge with existing AnnData
    scv.utils.merge(dr.adata, velocity_adata)
    
    # Preprocess
    scv.pp.filter_and_normalize(dr.adata)
    scv.pp.moments(dr.adata)
    
    # Compute velocity
    scv.tl.velocity(dr.adata, mode='stochastic')
    scv.tl.velocity_graph(dr.adata)
    
    # Visualize
    scv.pl.velocity_embedding_stream(dr.adata, basis='umap', save="velocity_stream.png")
    
except Exception as e:
    print(f"RNA velocity analysis failed: {e}")
```

### Cell-Cell Communication Analysis

```python
try:
    import squidpy as sq
    
    # Calculate neighborhood graph
    sq.gr.spatial_neighbors(dr.adata, coord_type='generic', delaunay=True)
    
    # Calculate co-occurrence scores
    sq.gr.co_occurrence(dr.adata, cluster_key='leiden')
    
    # Visualize co-occurrence
    sq.pl.co_occurrence(dr.adata, cluster_key='leiden', 
                       save="co_occurrence.png")
except Exception as e:
    print(f"Cell-cell communication analysis failed: {e}")
```

## 12. Saving and Loading Results

Save your processed data and results for future use.

### Save to S3

```python
# Save AnnData to S3
s3.write_h5ad(dr.adata, "my-bucket", "results/processed_data.h5ad")

# Save figures to S3
s3.upload_directory("./figures", "my-bucket", "results/figures")

print("Analysis results saved to S3")
```

### Save Locally

```python
# Save the processed AnnData object
dr.adata.write("processed_data.h5ad")

# Export key metadata as CSV
dr.adata.obs.to_csv("cell_metadata.csv")

# Save UMAP coordinates for external plotting
umap_df = pd.DataFrame(
    dr.adata.obsm['X_umap'],
    index=dr.adata.obs_names,
    columns=['UMAP1', 'UMAP2']
)
umap_df.to_csv("umap_coordinates.csv")
```

## Conclusion

This tutorial covered a complete workflow for single-cell and spatial transcriptomics data analysis using our custom toolkit. The modular design allows you to customize each step for your specific needs while maintaining a standardized and reproducible analysis pipeline.

For project-specific questions or advanced analyses, consult the detailed documentation for each module or reach out to our bioinformatics team.
