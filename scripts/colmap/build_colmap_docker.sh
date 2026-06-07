#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

COLMAP_SOURCE_DIR="${COLMAP_SOURCE_DIR:-$REPO_ROOT/third_party/colmap}"
COLMAP_REF="${COLMAP_REF:-main}"
COLMAP_IMAGE="${COLMAP_DOCKER_IMAGE:-colmap:latest}"

if [ ! -d "$COLMAP_SOURCE_DIR/.git" ]; then
  mkdir -p "$(dirname "$COLMAP_SOURCE_DIR")"
  git clone https://github.com/colmap/colmap.git "$COLMAP_SOURCE_DIR"
else
  git -C "$COLMAP_SOURCE_DIR" fetch --all --tags --prune
fi

git -C "$COLMAP_SOURCE_DIR" checkout "$COLMAP_REF"

docker build "$COLMAP_SOURCE_DIR" \
  -f "$COLMAP_SOURCE_DIR/docker/Dockerfile" \
  -t "$COLMAP_IMAGE" \
  "$@"
