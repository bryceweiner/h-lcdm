#!/bin/bash
# H-ΛCDM Docker Initialization Script
# ===================================
#
# This script builds the H-ΛCDM analysis Docker image and prepares
# the environment for reproducible scientific computing.
#
# Usage:
#   ./docker/init.sh [tag]
#
# Arguments:
#   tag: Docker image tag (default: latest)
#
# The script will:
# 1. Build the Docker image
# 2. Verify the build
# 3. Display usage instructions

set -e

# Configuration
IMAGE_NAME="hlcdm-analysis"
TAG=${1:-"latest"}
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        log_error "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    log_info "Docker version: $(docker --version)"
}

# Build the Docker image
build_image() {
    log_info "Building H-ΛCDM analysis Docker image: ${FULL_IMAGE_NAME}"

    # Get the project root directory (parent of docker directory)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

    log_info "Project root: ${PROJECT_ROOT}"

    # Build the image
    if docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${FULL_IMAGE_NAME}" "${PROJECT_ROOT}"; then
        log_success "Docker image built successfully: ${FULL_IMAGE_NAME}"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Verify the image
verify_image() {
    log_info "Verifying Docker image..."

    # Check if image exists
    if ! docker images "${IMAGE_NAME}" | grep -q "${TAG}"; then
        log_error "Image ${FULL_IMAGE_NAME} was not created successfully"
        exit 1
    fi

    # Test basic functionality
    log_info "Testing basic image functionality..."
    if docker run --rm "${FULL_IMAGE_NAME}" python -c "import numpy as np; print('NumPy version:', np.__version__)" > /dev/null 2>&1; then
        log_success "Basic image functionality verified"
    else
        log_warning "Basic functionality test failed - image may still work for analysis"
    fi

    # Get image size
    IMAGE_SIZE=$(docker images "${FULL_IMAGE_NAME}" --format "table {{.Size}}" | tail -n 1)
    log_info "Image size: ${IMAGE_SIZE}"
}

# Display usage instructions
show_usage() {
    echo
    log_success "H-ΛCDM Docker image ready!"
    echo
    echo "Usage Examples:"
    echo "==============="
    echo
    echo "1. Run gamma analysis with validation:"
    echo "   docker run -v \$(pwd)/downloaded_data:/app/downloaded_data \\"
    echo "              -v \$(pwd)/processed_data:/app/processed_data \\"
    echo "              -v \$(pwd)/results:/app/results \\"
    echo "              ${FULL_IMAGE_NAME} --gamma validate"
    echo
    echo "2. Run all pipelines with extended validation:"
    echo "   docker run -v \$(pwd)/downloaded_data:/app/downloaded_data \\"
    echo "              -v \$(pwd)/processed_data:/app/processed_data \\"
    echo "              -v \$(pwd)/results:/app/results \\"
    echo "              ${FULL_IMAGE_NAME} --all extended"
    echo
    echo "3. Run BAO and CMB analysis:"
    echo "   docker run -v \$(pwd)/downloaded_data:/app/downloaded_data \\"
    echo "              -v \$(pwd)/processed_data:/app/processed_data \\"
    echo "              -v \$(pwd)/results:/app/results \\"
    echo "              ${FULL_IMAGE_NAME} --bao --cmb validate"
    echo
    echo "4. Get help:"
    echo "   docker run --rm ${FULL_IMAGE_NAME} --help"
    echo
    echo "Important Notes:"
    echo "================="
    echo "- Mount the three data directories as shown above"
    echo "- downloaded_data/: Raw downloaded astronomical data"
    echo "- processed_data/: Intermediate processed data"
    echo "- results/: Analysis outputs, reports, and figures"
    echo "- The container runs as non-root user 'hlcdm' for security"
    echo
}

# Main execution
main() {
    log_info "H-ΛCDM Docker Initialization"
    log_info "============================"

    check_docker
    build_image
    verify_image
    show_usage

    log_success "Initialization complete!"
}

# Run main function
main "$@"
