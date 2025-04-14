#!/bin/bash
# Development Jupyter container for Bioinformatics Toolkit
# This script runs a development container that mounts the local codebase

set -e  # Exit immediately if a command exits with a non-zero status

# Configuration
IMAGE_NAME="sctools-dev"
TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_CREDENTIALS="$HOME/.aws/credentials"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN}Running Development Jupyter Environment${NC}"
echo -e "${BLUE}==========================================${NC}"

# Check if Podman is installed
if ! command -v podman &> /dev/null; then
    echo -e "${RED}Error: Podman is not installed.${NC}"
    echo -e "Please install Podman first."
    exit 1
fi

# Check if image exists, build if not
if ! podman image exists ${FULL_IMAGE}; then
    echo -e "${YELLOW}Image not found. Building development image...${NC}"
    podman build --format docker -t ${FULL_IMAGE} -f ${REPO_ROOT}/docker/dev-container.Dockerfile ${REPO_ROOT}
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Build successful!${NC}"
fi

# Run the container
echo -e "${YELLOW}Starting Jupyter Lab...${NC}"
echo -e "${GREEN}The repository will be mounted at /data/repo in the container${NC}"
echo -e "${GREEN}Any changes you make to the code will persist in your local repository${NC}"
echo -e "${GREEN}Access Jupyter Lab in your browser at: http://localhost:8888${NC}"

# Create a directory for data if it doesn't exist
mkdir -p "${REPO_ROOT}/data"

# Run the container with the repository mounted
podman run -it --rm \
    -p 0.0.0.0:8888:8888 \
    -v "${REPO_ROOT}:/data/repo:Z" \
    -v "${REPO_ROOT}/data:/data/data:Z" \
    -e "PYTHONPATH=/data/repo" \
    -e "JUPYTER_ENABLE_LAB=yes" \
    ${FULL_IMAGE}

# podman run -it --rm \
#     -p 8888:8888 \
#     -v "${REPO_ROOT}:/data/repo:Z" \
#     -v "${REPO_ROOT}/data:/data/data:Z" \
#     -v "${S3_CREDENTIALS}:/home/jovyan/.aws/credentials:ro" \
#     -e "PYTHONPATH=/data/repo" \
#     ${FULL_IMAGE}