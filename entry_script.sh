#!/bin/bash
# Entrypoint script for the Bioinformatics Toolkit Container
# This script sets up the environment and handles initialization

set -e  # Exit immediately if a command exits with a non-zero status

# Display welcome message
cat << 'EOF'
=======================================================
 _____  _____  _____  _____  _____  _____  _       _____ 
/  ___\/     \/  _  \/  _  \/  _  \/  _  \/   \   /  ___)
|  |__ |  |--||  |  ||  |  ||  |  ||  |  ||    \ /   /   
\___  \|     ||  |  ||  |  ||  |  ||  |  ||  |  \|   \_  
 ___| ||  |--||  |__||  |__||  |__||  |__||  |   \     | 
/_____/\_____/\_____/\_____/\_____/\_____/|__|    \____/ 
=======================================================
Single-Cell and Spatial Transcriptomics Analysis Toolkit
=======================================================

EOF

# Setup Python/R environment variables
export PYTHONPATH="/app:${PYTHONPATH}"
export R_LIBS_USER="${R_LIBS_USER}:/app/R"

# Create tmp directory if it doesn't exist
mkdir -p /tmp/data

# Check AWS credentials
if [ -f "$HOME/.aws/credentials" ]; then
    echo "✅ AWS credentials found"
else
    echo "⚠️  AWS credentials not found. S3 functionality will be limited."
    echo "   To use S3, mount credentials with:"
    echo "   -v ~/.aws/credentials:/home/biouser/.aws/credentials:ro"
fi

# Check for mounted data directory
if [ -d "/data" ] && [ "$(ls -A /data 2>/dev/null)" ]; then
    echo "✅ Mounted data directory detected"
else
    echo "⚠️  No data mounted or empty data directory"
    echo "   To mount data, use: -v /path/to/your/data:/data"
fi

# Display Python packages
echo "📦 Available Python packages:"
pip list | grep -E 'scanpy|anndata|loom|scvi|squid|numpy|pandas|scipy'

# Check if Jupyter is the command
if [[ "$1" == "jupyter" ]]; then
    echo "🚀 Starting Jupyter Lab server..."
    echo "   Access in your browser at: http://localhost:8888"
    echo "   (Token will be displayed in the output)"
    echo ""
elif [[ "$1" == "python" ]]; then
    echo "🐍 Starting Python environment..."
    echo "   Available modules: scanpy, anndata, numpy, pandas, matplotlib"
    echo "   Example: from sctools.qc import SingleCellQC"
    echo ""
elif [[ "$1" == "R" ]]; then
    echo "📊 Starting R environment..."
    echo "   Available packages: Seurat, scran, sctransform"
    echo ""
fi

echo "Ready to analyze single-cell and spatial data!"
echo "================================================"

# Execute the command
exec "$@"
