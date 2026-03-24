#!/bin/bash
# Build Docker image and package Olares Application Chart
#
# Usage:
#   ./olares/package.sh                    # Build + package
#   ./olares/package.sh --push             # Build + push to Docker Hub + package
#   REGISTRY=ghcr.io/drlucaslu ./olares/package.sh --push  # Custom registry

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REGISTRY="${REGISTRY:-drlucaslu}"
IMAGE_NAME="${REGISTRY}/louter"
VERSION="${VERSION:-latest}"
CHART_DIR="$SCRIPT_DIR"

cd "$PROJECT_DIR"

echo "=== Building Docker image ==="
docker build -t "${IMAGE_NAME}:${VERSION}" .

if [[ "${1:-}" == "--push" ]]; then
    echo "=== Pushing to registry ==="
    docker push "${IMAGE_NAME}:${VERSION}"
    echo "Pushed: ${IMAGE_NAME}:${VERSION}"
fi

echo ""
echo "=== Packaging Olares chart ==="
# Update image tag in values.yaml
sed -i.bak "s|tag: .*|tag: \"${VERSION}\"|" "$CHART_DIR/values.yaml"
sed -i.bak "s|repository: .*|repository: ${IMAGE_NAME}|" "$CHART_DIR/values.yaml"
rm -f "$CHART_DIR/values.yaml.bak"

# Package as tgz
OUTPUT="${PROJECT_DIR}/louter-olares-${VERSION}.tgz"
tar -czf "$OUTPUT" -C "$(dirname "$CHART_DIR")" "$(basename "$CHART_DIR")"

echo ""
echo "=== Done ==="
echo "Docker image: ${IMAGE_NAME}:${VERSION}"
echo "Olares chart: ${OUTPUT}"
echo ""
echo "To deploy on Olares:"
echo "  1. Open Market → My Olares → Upload"
echo "  2. Select: ${OUTPUT}"
