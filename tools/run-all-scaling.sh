#!/bin/bash
# run-all-scaling.sh — Run all scaling tests across multiple iterations
# Usage: ./run-all-scaling.sh [start_run] [end_run]
set -euo pipefail

BENCH_DIR=/home/coulof/src/tries/2026-02-05-localscore-llama-bench/localscore-bench
VULKAN_BENCH="/usr/bin/llama-bench"  # System build with working NVIDIA Vulkan
CUDA_BENCH="$BENCH_DIR/backends/cuda/llama-bench"
PLOT_SCRIPT=/home/coulof/src/llm-bench-lab/tools/plot_gpu_usage.py
REPO_DIR=/home/coulof/src/llm-bench-lab
BASE_DIR="$REPO_DIR/results/psyche-suse/2026-02-09/scaling-test"

START_RUN="${1:-1}"
END_RUN="${2:-3}"

# Model definitions: name|gguf_path
MODELS=(
    "llama-3.2-1b|/home/coulof/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    "gemma-3-1b|/home/coulof/models/gemma-3-1b-it-Q4_K_M.gguf"
    "phi-4-mini-3.8b|/home/coulof/models/Phi-4-mini-instruct-Q4_K_M.gguf"
    "ministral-8b|/home/coulof/models/Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    "gemma-3-12b|/home/coulof/models/gemma-3-12b-it-Q4_K_M.gguf"
    "mistral-nemo-12b|/home/coulof/models/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
    "qwen3-32b|/home/coulof/.ollama/models/blobs/sha256-3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312"
    "llama-3.3-70b|/home/coulof/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
)

run_single() {
    local model_name="$1"
    local model_path="$2"
    local backend="$3"    # vulkan or cuda13
    local run_num="$4"
    local run_dir="$5"

    local bench_bin gpu_index suffix backend_label
    if [[ "$backend" == "vulkan" ]]; then
        bench_bin="$VULKAN_BENCH"
        gpu_index=1  # Vulkan device 1 = NVIDIA (0 = AMD iGPU)
        suffix="vulkan"
        backend_label="Vulkan"
    else
        bench_bin="$CUDA_BENCH"
        gpu_index=0  # CUDA device 0 = only NVIDIA GPU
        suffix="cuda13"
        backend_label="CUDA 13.1"
    fi

    local out_csv="$run_dir/${model_name}-${suffix}.csv"
    local out_json="$run_dir/${model_name}-${suffix}.json"
    local out_png="$run_dir/${model_name}-${suffix}.png"

    # Skip if already completed (has JSON with results)
    if [[ -f "$out_json" ]] && python3 -c "
import json
with open('$out_json') as f:
    d = json.load(f)
results = d.get('results', d if isinstance(d, list) else [])
assert len(results) >= 2, 'incomplete'
" 2>/dev/null; then
        echo "⏭️  Skip $model_name $backend_label (run $run_num) — already complete"
        return 0
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Run $run_num | $model_name | $backend_label"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Remove partial results
    rm -f "$out_csv" "$out_json" "$out_png"

    local max_retries=2
    local attempt=0
    local success=false

    while [[ $attempt -lt $max_retries ]] && [[ "$success" == "false" ]]; do
        attempt=$((attempt + 1))
        [[ $attempt -gt 1 ]] && echo "  🔄 Retry attempt $attempt..."

        cd "$BENCH_DIR"
        if python3 main.py \
            -m "$model_path" \
            --llama-bench "$bench_bin" \
            -i "$gpu_index" \
            --quick \
            -n \
            --save-json "$out_json" \
            --monitor-gpu "$out_csv" \
            --monitor-interval 200 \
            -o console 2>&1; then
            success=true
        else
            echo "  ⚠️ Attempt $attempt failed"
            rm -f "$out_csv" "$out_json"
            sleep 5
        fi
    done

    if [[ "$success" == "false" ]]; then
        echo "  ❌ $model_name $backend_label FAILED after $max_retries attempts"
        return 1
    fi

    # Generate chart
    if [[ -f "$out_csv" ]]; then
        python3 "$PLOT_SCRIPT" "$out_csv" \
            -o "$out_png" \
            -t "GPU Usage — $model_name ($backend_label, RTX PRO 6000) [Run $run_num]" \
            2>/dev/null && echo "  📊 Chart saved" || echo "  ⚠️ Chart generation failed"
    fi

    echo "  ✅ $model_name $backend_label complete"
}

for run in $(seq "$START_RUN" "$END_RUN"); do
    run_dir="$BASE_DIR"
    if [[ "$run" -gt 1 ]]; then
        run_dir="$BASE_DIR/run-$run"
        mkdir -p "$run_dir"
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║              SCALING TEST — RUN $run                  ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo "Output: $run_dir"

    for model_def in "${MODELS[@]}"; do
        IFS='|' read -r model_name model_path <<< "$model_def"

        if [[ ! -f "$model_path" ]]; then
            echo "⚠️  Skipping $model_name — model not found: $model_path"
            continue
        fi

        run_single "$model_name" "$model_path" "vulkan" "$run" "$run_dir" || true
        run_single "$model_name" "$model_path" "cuda13" "$run" "$run_dir" || true
    done

    # Commit and push after each run
    echo ""
    echo "📤 Committing run $run..."
    cd "$REPO_DIR"
    git add results/psyche-suse/2026-02-09/scaling-test/ tools/
    git commit -m "Scaling test run $run — all models Vulkan + CUDA 13.1

Automated run via run-all-scaling.sh" 2>/dev/null || echo "  (nothing to commit)"
    git push 2>/dev/null || echo "  ⚠️ Push failed"
    echo "✅ Run $run committed and pushed"
done

echo ""
echo "🏁 All runs complete!"
