#!/bin/bash
# run-scaling-test.sh — Run a single model through Vulkan + CUDA scaling test
# Usage: ./run-scaling-test.sh <model_gguf> <output_name> [--vulkan-only|--cuda-only]
#
# Environment variables (all optional, with sensible defaults):
#   BENCH_DIR       — localscore-bench directory
#   RESULTS_DIR     — where to write results
#   VULKAN_DEVICE   — Vulkan device index (auto-detected if unset)
#   CUDA_DEVICE     — CUDA device index (default: 0)
#   GPU_LABEL       — GPU name for chart titles (auto-detected if unset)
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-/home/coulof/src/tries/2026-02-05-localscore-llama-bench/localscore-bench}"
VULKAN_BENCH="$BENCH_DIR/backends/vulkan/llama-bench"
CUDA_BENCH="$BENCH_DIR/backends/cuda/llama-bench"
RESULTS_DIR="${RESULTS_DIR:-/home/coulof/src/llm-bench-lab/results/psyche-suse/$(date +%Y-%m-%d)/scaling-test}"
PLOT_SCRIPT="$(dirname "$0")/plot_gpu_usage.py"

# Auto-detect Vulkan device index: find the NVIDIA GPU
if [[ -z "${VULKAN_DEVICE:-}" ]]; then
    VULKAN_DEVICE=$(vulkaninfo --summary 2>/dev/null \
        | grep -n "deviceName.*NVIDIA" \
        | head -1 \
        | awk -F: '{print NR-1}')
    # fallback: count GPU entries before the NVIDIA one
    VULKAN_DEVICE=$(vulkaninfo --summary 2>/dev/null \
        | grep "deviceName" \
        | awk '/NVIDIA/{print NR-1; exit}')
    VULKAN_DEVICE="${VULKAN_DEVICE:-0}"
    echo "Auto-detected Vulkan device index: $VULKAN_DEVICE"
fi
CUDA_DEVICE="${CUDA_DEVICE:-0}"

# Auto-detect GPU label for chart titles
if [[ -z "${GPU_LABEL:-}" ]]; then
    GPU_LABEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
    GPU_LABEL="${GPU_LABEL:-Unknown GPU}"
fi

MODEL_PATH="$1"
OUTPUT_NAME="$2"
MODE="${3:-both}"  # both, --vulkan-only, --cuda-only

mkdir -p "$RESULTS_DIR"

run_bench() {
    local backend="$1"  # vulkan or cuda13
    local bench_bin="$2"
    local gpu_index="$3"
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
        -t "GPU Usage — $OUTPUT_NAME ($backend_label, $GPU_LABEL)" \
        2>/dev/null || echo "⚠️  Chart generation failed (non-fatal)"

    echo "✅ $backend done: $out_json, $out_csv, $out_png"
}

if [[ "$MODE" != "--cuda-only" ]]; then
    run_bench "Vulkan" "$VULKAN_BENCH" "$VULKAN_DEVICE" "vulkan"
fi

if [[ "$MODE" != "--vulkan-only" ]]; then
    run_bench "CUDA 13.1" "$CUDA_BENCH" "$CUDA_DEVICE" "cuda13"
fi

echo ""
echo "🏁 All done for $OUTPUT_NAME"
