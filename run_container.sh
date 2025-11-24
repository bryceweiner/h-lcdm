#!/bin/bash
# H-ΛCDM Analysis - Docker Runner
# ===============================
#
# This script provides convenient commands for running the H-ΛCDM
# analysis framework in Docker containers.
#
# Usage:
#   ./run_container.sh init        # Build the Docker image
#   ./run_container.sh gamma       # Run gamma analysis
#   ./run_container.sh bao         # Run BAO analysis
#   ./run_container.sh all         # Run all analyses
#   ./run_container.sh help        # Show help
#
# The script automatically mounts the required data directories
# and handles Docker volume mounting for reproducible analysis.

set -e

# Configuration
IMAGE_NAME="hlcdm-analysis"
TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
DOCKERFILE_DIR="docker"

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

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
}

# Check if image exists
check_image() {
    if ! docker images "${IMAGE_NAME}" | grep -q "${TAG}"; then
        log_warning "Docker image ${FULL_IMAGE_NAME} not found"
        log_info "Run './run_container.sh init' to build the image"
        exit 1
    fi
}

# Get volume mounts
get_volume_mounts() {
    # Ensure data directories exist
    mkdir -p downloaded_data processed_data results

    echo "-v $(pwd)/downloaded_data:/app/downloaded_data \
          -v $(pwd)/processed_data:/app/processed_data \
          -v $(pwd)/results:/app/results"
}

# Run Docker container
run_container() {
    local args="$*"
    local mounts=$(get_volume_mounts)

    log_info "Running H-ΛCDM analysis: $args"
    log_info "Using volumes: downloaded_data, processed_data, results"

    eval "docker run --rm $mounts $FULL_IMAGE_NAME $args"
}

# Initialize/build Docker image
cmd_init() {
    log_info "Initializing H-ΛCDM Docker environment..."

    if [ ! -f "${DOCKERFILE_DIR}/init.sh" ]; then
        log_error "init.sh script not found in ${DOCKERFILE_DIR}/"
        exit 1
    fi

    bash "${DOCKERFILE_DIR}/init.sh" "$@"
}

# Run gamma analysis
cmd_gamma() {
    check_image
    run_container --gamma validate "$@"
}

# Run BAO analysis
cmd_bao() {
    check_image
    run_container --bao validate "$@"
}

# Run CMB analysis
cmd_cmb() {
    check_image
    run_container --cmb validate "$@"
}

# Run void analysis
cmd_void() {
    check_image
    run_container --void validate "$@"
}

# Run all analyses
cmd_all() {
    check_image
    run_container --all validate "$@"
}

# Run custom command
cmd_run() {
    check_image
    run_container "$@"
}

# Show help
cmd_help() {
    cat << EOF
H-ΛCDM Analysis Framework - Docker Runner
=========================================

This script provides convenient commands for running the H-ΛCDM
analysis framework using Docker.

USAGE:
    ./run_container.sh <command> [options]

COMMANDS:
    init              Build the Docker image
    gamma             Run gamma analysis (γ(z), Λ(z))
    bao               Run BAO analysis (α predictions)
    cmb               Run CMB analysis (E-mode signatures)
    void              Run void analysis (E8×E8 alignments)
    all               Run all analyses
    run <args>        Run custom command in container
    help              Show this help message

EXAMPLES:
    # Initialize (build Docker image)
    ./run_container.sh init

    # Run gamma analysis with validation
    ./run_container.sh gamma

    # Run BAO analysis with extended validation
    ./run_container.sh bao extended

    # Run all analyses
    ./run_container.sh all

    # Run custom analysis
    ./run_container.sh run --gamma --bao validate

DATA DIRECTORIES:
    The script automatically mounts these host directories:
    - ./downloaded_data/  → Raw downloaded astronomical data
    - ./processed_data/   → Intermediate processed data
    - ./results/          → Analysis outputs, reports, figures

REQUIREMENTS:
    - Docker must be installed and running
    - Run './run_container.sh init' first to build the image

For more options, run:
    ./run_container.sh run --help

EOF
}

# Main command dispatcher
main() {
    check_docker

    case "${1:-help}" in
        init)
            shift
            cmd_init "$@"
            ;;
        gamma)
            shift
            cmd_gamma "$@"
            ;;
        bao)
            shift
            cmd_bao "$@"
            ;;
        cmb)
            shift
            cmd_cmb "$@"
            ;;
        void)
            shift
            cmd_void "$@"
            ;;
        all)
            shift
            cmd_all "$@"
            ;;
        run)
            shift
            cmd_run "$@"
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            log_error "Unknown command: $1"
            log_info "Run './run_container.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
