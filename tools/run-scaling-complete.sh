#!/bin/bash
# run-scaling-complete.sh — Run complete scaling test, both backends, push after each model
# Usage: ./run-scaling-complete.sh
set -euo pipefail

REPO_DIR=/home/coulof/src/llm-bench-lab
BASE_DIR="$REPO_DIR/results/psyche-suse/2026-02-09/scaling-test"
PLOT_SCRIPT="$REPO_DIR/tools/plot_gpu_usage.py"
MONITOR="$REPO_DIR/tools/monitor-gpu.sh"

# System Vulkan llama-bench (has NVIDIA Vulkan support)
VULKAN_BENCH="/usr/bin/llama-bench"
# CUDA llama-bench from localscore-bench
CUDA_DIR=/home/coulof/src/tries/2026-02-05-localscore-llama-bench/localscore-bench/backends/cuda
CUDA_BENCH="$CUDA_DIR/llama-bench"
CUDA_LD="$CUDA_DIR:/usr/local/cuda-13.1/lib64"

QWEN_BLOB=$(find /home/coulof/.ollama/models/blobs -name "sha256-3291*" -type f 2>/dev/null | head -1)

MODELS=(
    "gemma-3-1b|/home/coulof/models/gemma-3-1b-it-Q4_K_M.gguf|Gemma 3 1B"
    "llama-3.2-1b|/home/coulof/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf|Llama 3.2 1B"
    "phi-4-mini-3.8b|/home/coulof/models/Phi-4-mini-instruct-Q4_K_M.gguf|Phi-4 Mini 3.8B"
    "ministral-8b|/home/coulof/models/Ministral-8B-Instruct-2410-Q4_K_M.gguf|Ministral 8B"
    "gemma-3-12b|/home/coulof/models/gemma-3-12b-it-Q4_K_M.gguf|Gemma 3 12B"
    "mistral-nemo-12b|/home/coulof/models/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf|Mistral Nemo 12B"
    "qwen3-32b|${QWEN_BLOB}|Qwen3 32B"
    "llama-3.3-70b|/home/coulof/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf|Llama 3.3 70B"
)

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
    echo "  $title — $backend"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    rm -f "$csv" "$png"

    if [[ "$backend" == "vulkan" ]]; then
        env -u LD_LIBRARY_PATH "$MONITOR" -o "$csv" -g 0 -- \
            "$VULKAN_BENCH" -m "$model" -p 512 -n 128 -r 3 -ngl 99 2>&1 | tail -8
    else
        env LD_LIBRARY_PATH="$CUDA_LD" "$MONITOR" -o "$csv" -g 0 -- \
            "$CUDA_BENCH" -m "$model" -p 512 -n 128 -r 3 -ngl 99 2>&1 | tail -8
    fi

    # Generate chart
    if [[ -f "$csv" ]]; then
        python3 "$PLOT_SCRIPT" "$csv" -o "$png" \
            -t "$title Q4_K_M — $backend — RTX PRO 6000" 2>&1 | grep "✅" || true
    fi

    echo "  ✅ $slug $backend done"
}

mkdir -p "$BASE_DIR"

for model_def in "${MODELS[@]}"; do
    IFS='|' read -r slug model_path title <<< "$model_def"

    if [[ ! -f "$model_path" ]]; then
        echo "⚠️  Skip $slug — model not found"
        continue
    fi

    run_bench "$slug" "$model_path" "$title" "vulkan"
    run_bench "$slug" "$model_path" "$title" "cuda13"

    # Commit and push after each model
    cd "$REPO_DIR"
    git add -A
    git commit -m "bench: $title — Vulkan + CUDA 13.1 scaling test" 2>/dev/null || true
    git push origin gpu-monitoring-tests 2>/dev/null || true
    echo "📤 Pushed $title results"
done

echo ""
echo "🏁 All scaling tests complete!"
