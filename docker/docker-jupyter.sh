#!/bin/bash
# Docker Development Jupyter container for Bioinformatics Toolkit
# This script runs a development container that mounts the local codebase and starts Jupyter Lab in the sctools-py environment

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
echo -e "${GREEN}Running Docker Development Jupyter Environment${NC}"
echo -e "${GREEN}Using sctools-py and sctools-r environments${NC}"
echo -e "${BLUE}==========================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo -e "Please install Docker first."
    exit 1
fi

# Always rebuild the image to ensure latest environment files are used
echo -e "${YELLOW}Building development image with conda environments...${NC}"
docker build --progress=plain --no-cache -t ${FULL_IMAGE} -f ${REPO_ROOT}/docker/dev-container.Dockerfile ${REPO_ROOT}

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Build successful!${NC}"

# Run the container
echo -e "${YELLOW}Starting Jupyter Lab in sctools-py environment...${NC}"
echo -e "${GREEN}The repository will be mounted at /data/repo in the container${NC}"
echo -e "${GREEN}Any changes you make to the code will persist in your local repository${NC}"
echo -e "${GREEN}Access Jupyter Lab in your browser at: http://localhost:8888${NC}"
echo -e "${BLUE}Both sctools-py and sctools-r kernels are available in Jupyter${NC}"

# Create a directory for data if it doesn't exist
mkdir -p "${REPO_ROOT}/data"

# Check if nvidia-smi is available (indicates NVIDIA GPU is present)
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected, enabling GPU support in container${NC}"
    HAS_GPU=true
else
    echo -e "${YELLOW}No NVIDIA GPU detected, running in CPU-only mode${NC}"
fi

# Run the container with the repository mounted
# For Windows users, convert Windows paths to Docker paths if using Git Bash or similar
if [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
    REPO_ROOT_DOCKER=$(echo ${REPO_ROOT} | sed 's/^\([a-zA-Z]\):/\/\1/' | sed 's/\\/\//g')
    DATA_PATH_DOCKER="${REPO_ROOT_DOCKER}/data"
    
    echo -e "${YELLOW}Detected Windows environment, converting paths for Docker...${NC}"
    echo -e "${BLUE}Using ${REPO_ROOT_DOCKER} as repository path${NC}"
    
    if [ "$HAS_GPU" = true ]; then
        docker run -it --rm \
            --gpus all \
            -p 8888:8888 \
            -v "${REPO_ROOT_DOCKER}:/data/repo" \
            -v "${DATA_PATH_DOCKER}:/data/data" \
            -e "PYTHONPATH=/data/repo" \
            -e "NVIDIA_VISIBLE_DEVICES=all" \
            --workdir /data/repo \
            ${FULL_IMAGE}
    else
        docker run -it --rm \
            -p 8888:8888 \
            -v "${REPO_ROOT_DOCKER}:/data/repo" \
            -v "${DATA_PATH_DOCKER}:/data/data" \
            -e "PYTHONPATH=/data/repo" \
            --workdir /data/repo \
            ${FULL_IMAGE}
    fi
else
    # Linux/Mac OS X path handling
    if [ "$HAS_GPU" = true ]; then
        docker run -it --rm \
            --gpus all \
            -p 8888:8888 \
            -v "${REPO_ROOT}:/data/repo" \
            -v "${REPO_ROOT}/data:/data/data" \
            -e "PYTHONPATH=/data/repo" \
            -e "NVIDIA_VISIBLE_DEVICES=all" \
            --workdir /data/repo \
            ${FULL_IMAGE}
    else
        docker run -it --rm \
            -p 8888:8888 \
            -v "${REPO_ROOT}:/data/repo" \
            -v "${REPO_ROOT}/data:/data/data" \
            -e "PYTHONPATH=/data/repo" \
            --workdir /data/repo \
            ${FULL_IMAGE}
    fi
fi

# Optional: Run with AWS credentials if needed
# Uncomment the following to mount AWS credentials
# For Windows:
# if [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
#     S3_CREDENTIALS_DOCKER=$(echo ${S3_CREDENTIALS} | sed 's/^\([a-zA-Z]\):/\/\1/' | sed 's/\\/\//g')
#     docker run -it --rm \
#         -p 8888:8888 \
#         -v "${REPO_ROOT_DOCKER}:/data/repo" \
#         -v "${DATA_PATH_DOCKER}:/data/data" \
#         -v "${S3_CREDENTIALS_DOCKER}:/home/developer/.aws/credentials:ro" \
#         -e "PYTHONPATH=/data/repo" \
#         --workdir /data/repo \
#         ${FULL_IMAGE}
# else
#     # Linux/Mac OS X AWS credentials
#     docker run -it --rm \
#         -p 8888:8888 \
#         -v "${REPO_ROOT}:/data/repo" \
#         -v "${REPO_ROOT}/data:/data/data" \
#         -v "${S3_CREDENTIALS}:/home/developer/.aws/credentials:ro" \
#         -e "PYTHONPATH=/data/repo" \
#         --workdir /data/repo \
#         ${FULL_IMAGE}
# fi
