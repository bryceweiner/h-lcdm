#!/usr/bin/env bash
set -euo pipefail

# Ensure script runs from repository root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"

DOWNLOAD_DIR="$PROJECT_ROOT/downloaded_data"
PROCESSED_DIR="$PROJECT_ROOT/processed_data"
RESULTS_DIR="$PROJECT_ROOT/results"

mkdir -p "$DOWNLOAD_DIR" "$PROCESSED_DIR" "$RESULTS_DIR"

# Prepare the ML-specific Dockerfile
cp -f "$DOCKER_DIR/Dockerfile.ml" "$DOCKER_DIR/Dockerfile"

IMAGE_TAG="hlcdm-ml"
docker build -t "$IMAGE_TAG" "$DOCKER_DIR"

# Default to running the ML pipeline if no args provided
if [ "$#" -eq 0 ]; then
  set -- --ml
fi

docker run --rm \
  -v "$DOWNLOAD_DIR":/app/downloaded_data \
  -v "$PROCESSED_DIR":/app/processed_data \
  -v "$RESULTS_DIR":/app/results \
  "$IMAGE_TAG" "$@"

