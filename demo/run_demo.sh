#!/usr/bin/env bash
# Start the demo locally (CPU only): FastAPI backend on :8000 + React (Vite) on :5173.
#   bash demo/run_demo.sh                      # final classifier from configs/inference.yaml (ensemble)
#   bash demo/run_demo.sh outputs/<job>/model  # a single model instead
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -ge 1 ]; then export MODEL_DIR="$1"; echo "model dir: $MODEL_DIR"; else echo "classifier: configs/inference.yaml"; fi
.venv/bin/python -m uvicorn demo.backend.app:app --port 8000 &
BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT
(cd demo/frontend && { [ -d node_modules ] || npm install --no-audit --no-fund; })
cd demo/frontend && npx vite --open
