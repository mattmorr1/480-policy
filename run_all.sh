#!/usr/bin/env bash
# End-to-end pipeline for the GDPR right-to-be-forgotten LLM demo.
#
# GPU usage:
#   bash run_all.sh
#
# Optional env vars:
#   PYTHON_BIN=python3
#   INSTALL_DEPS=1
#   OUTPUT_DIR=results
#   SYSTEMS=A,B,C,D
#   USE_MOCK_JUDGE=0
#
# Examples:
#   INSTALL_DEPS=1 bash run_all.sh
#   SYSTEMS=B,C USE_MOCK_JUDGE=1 bash run_all.sh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
SYSTEMS="${SYSTEMS:-A,B,C,D}"
USE_MOCK_JUDGE="${USE_MOCK_JUDGE:-0}"

log_step() {
  echo ""
  echo "=== $1 ==="
}

echo "Running with:"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  SYSTEMS=$SYSTEMS"
echo "  USE_MOCK_JUDGE=$USE_MOCK_JUDGE"

if command -v nvidia-smi >/dev/null 2>&1; then
  log_step "GPU detection"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
  echo ""
  echo "WARNING: nvidia-smi not found. Continuing, but GPU visibility is unknown."
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  log_step "Installing dependencies"
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

log_step "Step 1: Data preparation"
"$PYTHON_BIN" src/data_prep.py

log_step "Step 2: System A - shared LoRA training"
"$PYTHON_BIN" src/train_lora.py --mode shared --output_dir shared_adapter/

log_step "Step 3: System B - per-user LoRA training"
for id in forget_author retain_author_01 retain_author_02 retain_author_03 retain_author_04 retain_author_05; do
  echo "  Training $id..."
  "$PYTHON_BIN" src/train_lora.py --mode per_user --user_id "$id" --output_dir "adapters/$id/"
done

log_step "Step 4: System C - build RAG indices"
"$PYTHON_BIN" src/rag.py --build

log_step "Step 5: Evaluation"
EVAL_CMD=("$PYTHON_BIN" src/eval_harness.py --output_dir "$OUTPUT_DIR" --systems "$SYSTEMS")
if [[ "$USE_MOCK_JUDGE" == "1" ]]; then
  EVAL_CMD+=(--mock_judge)
fi
"${EVAL_CMD[@]}"

log_step "Step 6: Generate figures"
"$PYTHON_BIN" src/analyze.py --results_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR"

echo ""
echo "Done. Figures saved to $OUTPUT_DIR/"
ls "$OUTPUT_DIR"/*.pdf 2>/dev/null || echo "(no PDFs found — check for errors above)"
