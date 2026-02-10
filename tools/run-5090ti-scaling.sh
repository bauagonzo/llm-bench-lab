#!/bin/bash
# run-5090ti-scaling.sh — Scaling test for RTX 5090 Ti (32GB VRAM)
# Same machine as PRO 6000 tests → pure GPU-to-GPU comparison
# Usage: ./run-5090ti-scaling.sh [cuda-only]
set -euo pipefail

REPO_DIR=/home/coulof/src/llm-bench-lab
DATE=$(date +%Y-%m-%d)
BASE_DIR="$REPO_DIR/results/psyche-suse-5090ti/$DATE/scaling-test"
PLOT_SCRIPT="$REPO_DIR/tools/plot_gpu_usage.py"
MONITOR="$REPO_DIR/tools/monitor-gpu.sh"

# System Vulkan llama-bench
VULKAN_BENCH="/usr/bin/llama-bench"
# CUDA llama-bench
CUDA_DIR=/home/coulof/src/tries/2026-02-05-localscore-llama-bench/localscore-bench/backends/cuda
CUDA_BENCH="$CUDA_DIR/llama-bench"
CUDA_LD="$CUDA_DIR:/usr/local/cuda-13.1/lib64"

CUDA_ONLY="${1:-}"

# 5090 Ti: 32GB VRAM — models up to ~19GB GGUF
# Same 7 models as PRO 6000 minus 70B (40GB won't fit)
MODELS=(
    "gemma-3-1b|/home/coulof/models/gemma-3-1b-it-Q4_K_M.gguf|Gemma 3 1B"
    "llama-3.2-1b|/home/coulof/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf|Llama 3.2 1B"
    "phi-4-mini-3.8b|/home/coulof/models/Phi-4-mini-instruct-Q4_K_M.gguf|Phi-4 Mini 3.8B"
    "ministral-8b|/home/coulof/models/Ministral-8B-Instruct-2410-Q4_K_M.gguf|Ministral 8B"
    "gemma-3-12b|/home/coulof/models/gemma-3-12b-it-Q4_K_M.gguf|Gemma 3 12B"
    "mistral-nemo-12b|/home/coulof/models/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf|Mistral Nemo 12B"
    "qwen3-32b|/home/coulof/models/Qwen3-32B-Q4_K_M.gguf|Qwen3 32B"
)

# RTX 5090 Ti specs for chart reference
GPU_NAME="RTX 5090 Ti"
GPU_TDP=450  # W

run_bench() {
    local slug="$1" model="$2" title="$3" backend="$4"
    local suffix csv png

    if [[ "$backend" == "vulkan" ]]; then
        suffix="vulkan"
    else
        suffix="cuda13"
    fi

    csv="$BASE_DIR/${slug}-${suffix}.csv"
    png="$BASE_DIR/${slug}-${suffix}.png"

    # Skip if already done
    if [[ -f "$csv" ]] && [[ $(wc -l < "$csv") -gt 5 ]]; then
        echo "⏭️  Skip $slug $backend (already complete)"
        return 0
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $title — $backend ($GPU_NAME)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    rm -f "$csv" "$png"

    if [[ "$backend" == "vulkan" ]]; then
        env -u LD_LIBRARY_PATH "$MONITOR" -o "$csv" -g 0 -- \
            "$VULKAN_BENCH" -m "$model" -p 512 -n 128 -r 3 -ngl 99 2>&1 | tail -8
    else
        env LD_LIBRARY_PATH="$CUDA_LD" "$MONITOR" -o "$csv" -g 0 -- \
            "$CUDA_BENCH" -m "$model" -p 512 -n 128 -r 3 -ngl 99 2>&1 | tail -8
    fi

    # Generate chart — 5090 Ti: TDP 450W, max 600W, 32GB VRAM (32768 MiB)
    if [[ -f "$csv" ]]; then
        python3 "$PLOT_SCRIPT" "$csv" -o "$png" \
            -t "$title Q4_K_M — $backend — $GPU_NAME" \
            --power-limit 450 --power-limit2 600 --mem-total 32768 \
            2>&1 | grep "✅" || true
    fi

    echo "  ✅ $slug $backend done"
}

mkdir -p "$BASE_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   SCALING TEST — $GPU_NAME (32GB)                   ║"
echo "║   Same machine as PRO 6000 — GPU swap comparison    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "Output: $BASE_DIR"
echo ""

# Record system info
echo "=== System Info ===" | tee "$BASE_DIR/system-info.txt"
echo "Date: $(date -Iseconds)" | tee -a "$BASE_DIR/system-info.txt"
uname -a | tee -a "$BASE_DIR/system-info.txt"
nvidia-smi -L 2>&1 | tee -a "$BASE_DIR/system-info.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.max_limit --format=csv,noheader 2>&1 | tee -a "$BASE_DIR/system-info.txt"
echo "" | tee -a "$BASE_DIR/system-info.txt"

for model_def in "${MODELS[@]}"; do
    IFS='|' read -r slug model_path title <<< "$model_def"

    if [[ ! -f "$model_path" ]]; then
        echo "⚠️  Skip $slug — model not found"
        continue
    fi

    # Run CUDA first (never crashes), then Vulkan (may crash on Blackwell)
    run_bench "$slug" "$model_path" "$title" "cuda13"

    if [[ "$CUDA_ONLY" != "cuda-only" ]]; then
        run_bench "$slug" "$model_path" "$title" "vulkan"
    fi

    # Commit and push after each model
    cd "$REPO_DIR"
    git add -A
    git commit -m "bench: $title — $GPU_NAME scaling test" 2>/dev/null || true
    git push origin rtx-5090ti-prep 2>/dev/null || true
    echo "📤 Pushed $title results"
done

echo ""
echo "🏁 All $GPU_NAME scaling tests complete!"
