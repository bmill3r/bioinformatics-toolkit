#!/bin/bash
# Podman build script for Bioinformatics Toolkit
# This script builds a Podman container image for the toolkit and provides instructions for usage

set -e  # Exit immediately if a command exits with a non-zero status

# Configuration
IMAGE_NAME="sctools"
TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"
LOCAL_MOUNT_DIR="$PWD"  # Default to current directory
S3_CREDENTIALS="$HOME/.aws/credentials"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN}Building Single-Cell Tools Podman Image${NC}"
echo -e "${BLUE}==========================================${NC}"

# Check if Podman is installed
if ! command -v podman &> /dev/null; then
    echo -e "${RED}Error: Podman is not installed.${NC}"
    echo -e "Please install Podman first:"
    echo -e "  - On Ubuntu/Debian: sudo apt-get install podman"
    echo -e "  - On Fedora/RHEL/CentOS: sudo dnf install podman"
    echo -e "  - On macOS: brew install podman"
    exit 1
fi

# Check for Dockerfile
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Error: Dockerfile not found in the current directory.${NC}"
    exit 1
fi

# Build the image
echo -e "${YELLOW}Building Podman image...${NC}"
podman build --format docker -t ${FULL_IMAGE} .

# Check build status
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Print usage instructions
echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN}Usage Instructions:${NC}"
echo -e "${BLUE}==========================================${NC}"

echo -e "${YELLOW}Run Python environment:${NC}"
echo -e "podman run -it --rm -v ${LOCAL_MOUNT_DIR}:/data:Z -v ${S3_CREDENTIALS}:/home/biouser/.aws/credentials:ro ${FULL_IMAGE} python"
echo -e ""

echo -e "${YELLOW}Run R environment:${NC}"
echo -e "podman run -it --rm -v ${LOCAL_MOUNT_DIR}:/data:Z -v ${S3_CREDENTIALS}:/home/biouser/.aws/credentials:ro --entrypoint /entrypoint-r.sh ${FULL_IMAGE} R"
echo -e ""

echo -e "${YELLOW}Run JupyterLab:${NC}"
echo -e "podman run -it --rm -p 8888:8888 -v ${LOCAL_MOUNT_DIR}:/data:Z -v ${S3_CREDENTIALS}:/home/biouser/.aws/credentials:ro ${FULL_IMAGE}"
echo -e ""

echo -e "${YELLOW}Notes:${NC}"
echo -e "1. The ${GREEN}/data${NC} directory in the container is mapped to ${GREEN}${LOCAL_MOUNT_DIR}${NC} on your local machine."
echo -e "2. The ${GREEN}:Z${NC} suffix on mounted volumes is required for SELinux systems (e.g., Fedora, RHEL)."
echo -e "3. AWS credentials are mounted read-only from ${GREEN}${S3_CREDENTIALS}${NC}."
echo -e "4. To use a different local directory, replace ${GREEN}${LOCAL_MOUNT_DIR}${NC} with your desired path."
echo -e ""

echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN}Available Podman Images:${NC}"
echo -e "${BLUE}==========================================${NC}"
podman images | grep ${IMAGE_NAME}

# Create helper scripts for common operations
echo -e "${YELLOW}Creating helper scripts...${NC}"

# Python shell script
cat > run_python.sh << EOF
#!/bin/bash
podman run -it --rm -v "\$PWD":/data:Z -v "\$HOME/.aws/credentials":/home/biouser/.aws/credentials:ro ${FULL_IMAGE} python "\$@"
EOF
chmod +x run_python.sh

# R shell script
cat > run_r.sh << EOF
#!/bin/bash
podman run -it --rm -v "\$PWD":/data:Z -v "\$HOME/.aws/credentials":/home/biouser/.aws/credentials:ro --entrypoint /entrypoint-r.sh ${FULL_IMAGE} R "\$@"
EOF
chmod +x run_r.sh

# JupyterLab shell script
cat > run_jupyter.sh << EOF
#!/bin/bash
podman run -it --rm -p 8888:8888 -v "\$PWD":/data:Z -v "\$HOME/.aws/credentials":/home/biouser/.aws/credentials:ro ${FULL_IMAGE}
EOF
chmod +x run_jupyter.sh

echo -e "${GREEN}Helper scripts created:${NC}"
echo -e "  - ${YELLOW}./run_python.sh${NC} - Run Python environment"
echo -e "  - ${YELLOW}./run_r.sh${NC} - Run R environment"
echo -e "  - ${YELLOW}./run_jupyter.sh${NC} - Run JupyterLab"

echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${BLUE}==========================================${NC}"
