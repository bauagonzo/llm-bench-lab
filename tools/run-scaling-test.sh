#!/bin/bash
# run-scaling-test.sh — Run a single model through Vulkan + CUDA scaling test
# Usage: ./run-scaling-test.sh <model_gguf> <output_name> [--vulkan-only|--cuda-only]
set -euo pipefail

BENCH_DIR=/home/coulof/src/tries/2026-02-05-localscore-llama-bench/localscore-bench
VULKAN_BENCH="$BENCH_DIR/backends/vulkan/llama-bench"
CUDA_BENCH="$BENCH_DIR/backends/cuda/llama-bench"
RESULTS_DIR=/home/coulof/src/llm-bench-lab/results/psyche-suse/2026-02-09/scaling-test
PLOT_SCRIPT=/home/coulof/src/llm-bench-lab/tools/plot_gpu_usage.py

MODEL_PATH="$1"
OUTPUT_NAME="$2"
MODE="${3:-both}"  # both, --vulkan-only, --cuda-only

run_bench() {
    local backend="$1"  # vulkan or cuda13
    local bench_bin="$2"
    local gpu_index="$3"  # llama-bench device index (Vulkan=1 because 0=AMD iGPU, CUDA=0)
    local suffix="$4"

    local out_json="$RESULTS_DIR/${OUTPUT_NAME}-${suffix}.json"
    local out_csv="$RESULTS_DIR/${OUTPUT_NAME}-${suffix}.csv"
    local out_png="$RESULTS_DIR/${OUTPUT_NAME}-${suffix}.png"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $OUTPUT_NAME — $backend"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cd "$BENCH_DIR"
    
    python3 main.py \
        -m "$MODEL_PATH" \
        --llama-bench "$bench_bin" \
        -i "$gpu_index" \
        --quick \
        -n \
        --save-json "$out_json" \
        --monitor-gpu "$out_csv" \
        --monitor-interval 200 \
        -o console

    echo ""
    echo "📊 Generating chart..."
    
    local backend_label="Vulkan"
    [[ "$suffix" == "cuda13" ]] && backend_label="CUDA 13.1"
    
    python3 "$PLOT_SCRIPT" "$out_csv" \
        -o "$out_png" \
        -t "GPU Usage — $OUTPUT_NAME ($backend_label, RTX PRO 6000)" \
        2>/dev/null

    echo "✅ $backend done: $out_json, $out_csv, $out_png"
}

if [[ "$MODE" != "--cuda-only" ]]; then
    # Vulkan index 1 = NVIDIA (index 0 = AMD iGPU)
    run_bench "Vulkan" "$VULKAN_BENCH" 1 "vulkan"
fi

if [[ "$MODE" != "--vulkan-only" ]]; then
    # CUDA index 0 = only NVIDIA GPU visible to CUDA
    run_bench "CUDA 13.1" "$CUDA_BENCH" 0 "cuda13"
fi

echo ""
echo "🏁 All done for $OUTPUT_NAME"
