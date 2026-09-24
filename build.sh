#!/bin/bash
# Build, smoke-test, export, and package the competition submission.

set -euo pipefail

IMAGE_NAME="eda-cup-coverage-agent:v1"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$ROOT_DIR/deliverables"

mkdir -p "$OUT_DIR"

echo "[1/4] Building $IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$ROOT_DIR"

echo "[2/4] Smoke-testing inference import"
docker run --rm "$IMAGE_NAME"

echo "[3/4] Exporting submission.tar"
docker save "$IMAGE_NAME" -o "$OUT_DIR/submission.tar"
sha256sum "$OUT_DIR/submission.tar" | sed 's#  .*/#  #' > "$OUT_DIR/submission.tar.sha256"

echo "[4/4] Creating submission.zip"
python3 -c 'import pathlib,sys,zipfile; p=pathlib.Path(sys.argv[1]); z=zipfile.ZipFile(p/"submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9); z.write(p/"submission.tar", arcname="submission.tar"); z.close()' "$OUT_DIR"

echo "Ready: $OUT_DIR/submission.zip"
python3 -c 'import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); print("Archive entries:", z.namelist()); z.close()' "$OUT_DIR/submission.zip"
