#!/usr/bin/env bash
# Fetch the Nara SAR inference assets into ./models (about 430 MB).
# Only needed when ./models is missing or was not copied with the project.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

base="https://huggingface.co/YeeNyunn/nara-sar/resolve/main"

fetch() {
  local path="$1"
  local out="models/$(basename "$path")"
  if [ -s "$out" ]; then
    echo "skip $out (already present)"
    return
  fi
  echo "downloading $path"
  curl -fL --retry 3 --progress-bar -o "$out.part" "$base/$path"
  mv "$out.part" "$out"
}

fetch "sar.json"
fetch "language.lm"
fetch "onnx/acoustic.onnx"

echo
echo "done:"
ls -lh models
echo
echo "For the full-precision model instead of INT8:"
echo "  curl -fL -o models/acoustic-fp32.onnx $base/onnx/acoustic-fp32.onnx"
