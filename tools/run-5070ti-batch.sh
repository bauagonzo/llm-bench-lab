#!/bin/bash
# Batch run: 6 models, Vulkan + CUDA, on RTX 5070 Ti
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=(
    "/home/coulof/models/gemma-3-1b-it-Q4_K_M.gguf:gemma3-1b"
    "/home/coulof/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf:llama32-1b"
    "/home/coulof/models/Phi-4-mini-instruct-Q4_K_M.gguf:phi4-mini-3.8b"
    "/home/coulof/models/Ministral-8B-Instruct-2410-Q4_K_M.gguf:ministral-8b"
    "/home/coulof/models/gemma-3-12b-it-Q4_K_M.gguf:gemma3-12b"
    "/home/coulof/models/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf:mistral-nemo-12b"
)

export RESULTS_DIR="/home/coulof/src/llm-bench-lab/results/psyche-suse/5070ti/run-1"

echo "=== 5070 Ti Benchmark Batch ==="
echo "Results: $RESULTS_DIR"
echo "Models: ${#MODELS[@]}"
echo "Start: $(date)"
echo ""

FAILED=()
for entry in "${MODELS[@]}"; do
    model="${entry%%:*}"
    name="${entry##*:}"
    echo ">>> [$name] Starting at $(date)"
    if ./tools/run-scaling-test.sh "$model" "$name"; then
        echo ">>> [$name] Done at $(date)"
    else
        echo ">>> [$name] FAILED at $(date)"
        FAILED+=("$name")
    fi
    echo ""
done

echo "=== Batch Complete at $(date) ==="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}"
    exit 1
else
    echo "All 6 models passed."
    exit 0
fi
