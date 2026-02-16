#!/bin/bash
# run-cuda-pro6000.sh — Run CUDA benchmarks on RTX PRO 6000
# Matches 5070 Ti test models for direct comparison
set -euo pipefail

CUDA_DIR=~/src/tries/2026-02-05-localscore-llama-bench/localscore-bench/backends/cuda
MODELS_DIR=~/models
TOOLS_DIR=~/src/llm-bench-lab/tools
RESULTS_DIR=~/src/llm-bench-lab/results/psyche-suse/pro6000-cuda
BENCH="$CUDA_DIR/llama-bench"

# Quick test suite (same as scaling tests)
# pp1024+tg16, pp1024+tg1024, pp16+tg1536
BENCH_ARGS="-t 1 -ngl 999 -r 3 -o json"

declare -A MODELS=(
    ["gemma3-1b"]="gemma-3-1b-it-Q4_K_M.gguf"
    ["llama32-1b"]="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ["phi4-mini-3.8b"]="Phi-4-mini-instruct-Q4_K_M.gguf"
    ["ministral-8b"]="Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    ["gemma3-12b"]="gemma-3-12b-it-Q4_K_M.gguf"
    ["mistral-nemo-12b"]="Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
)

# Model order (bash doesn't preserve associative array order)
# 3 small models, repeated runs
ORDER=("gemma3-1b" "llama32-1b" "phi4-mini-3.8b")

mkdir -p "$RESULTS_DIR/run-1"

echo "━━━ RTX PRO 6000 CUDA Benchmark ━━━"
echo "Binary: $BENCH ($(cat $CUDA_DIR/VERSION))"
echo "Results: $RESULTS_DIR/run-1"
echo ""

for name in "${ORDER[@]}"; do
    model_file="${MODELS[$name]}"
    model_path="$MODELS_DIR/$model_file"

    echo "━━━ $name ━━━"

    # Pre-flight
    if ! "$TOOLS_DIR/gpu-preflight.sh" --model "$model_path"; then
        echo "SKIP: pre-flight failed for $name"
        continue
    fi

    echo ""
    echo "Running benchmark..."

    # GPU monitor
    export LD_LIBRARY_PATH="$CUDA_DIR:${LD_LIBRARY_PATH:-}"
    "$TOOLS_DIR/monitor-gpu.sh" -o "$RESULTS_DIR/run-1/${name}-cuda13.csv" -i 1 -g 0 -- \
        "$BENCH" -m "$model_path" \
        -p 1024,1024,16 -n 16,1024,1536 \
        $BENCH_ARGS > "$RESULTS_DIR/run-1/${name}-cuda13.json" 2>&1

    echo "Done: $name"
    echo ""

    # Brief cooldown between runs
    echo "Cooling 60s..."
    sleep 60
done

echo "━━━ All benchmarks complete ━━━"
