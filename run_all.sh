#!/bin/bash
# End-to-end pipeline for the GDPR right-to-be-forgotten LLM demo.
# Run from the project root: bash run_all.sh
set -e
cd "$(dirname "$0")"

echo "=== Step 1: Data preparation ==="
python src/data_prep.py

echo ""
echo "=== Step 2: System A — shared LoRA training ==="
python src/train_lora.py --mode shared --output_dir shared_adapter/

echo ""
echo "=== Step 3: System B — per-user LoRA training ==="
for id in forget_author retain_author_01 retain_author_02 retain_author_03 retain_author_04 retain_author_05; do
    echo "  Training $id..."
    python src/train_lora.py --mode per_user --user_id "$id" --output_dir "adapters/$id/"
done

echo ""
echo "=== Step 4: System C — build RAG indices ==="
python src/rag.py --build

echo ""
echo "=== Step 5: Evaluation (all four systems) ==="
# Remove --mock_judge to use real Ollama judge (requires: ollama pull llama3.2:1b)
python src/eval_harness.py --output_dir results/

echo ""
echo "=== Step 6: Generate figures ==="
python src/analyze.py --results_dir results/ --output_dir results/

echo ""
echo "Done. Figures saved to results/"
ls results/*.pdf 2>/dev/null || echo "(no PDFs found — check for errors above)"
