#!/usr/bin/env bash
# Generate the small test clips used by the parity harness.
# Requires macOS `say`; pass your own WAV to the harness on other platforms.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p samples

if ! command -v say >/dev/null 2>&1; then
  echo "macOS 'say' not found. Drop a 16 kHz mono WAV into samples/ instead." >&2
  exit 1
fi

say -v Samantha -o samples/sample-en.wav --data-format=LEF32@16000 \
  "Hello, this is a test of the Nara speech recognition model. The weather is nice today."

say -v Samantha -o samples/sample-en-long.wav --data-format=LEF32@16000 \
  "Good morning. This is a longer recording used to check how the transcription handles several chunks of audio. The system splits the signal at quiet points, so a short pause between sentences is the natural place to cut. Speech recognition needs a clean sixteen kilohertz mono signal. We are testing whether the browser version produces exactly the same text as the Python reference implementation. Thank you for listening, and have a pleasant day."

ls -lh samples
