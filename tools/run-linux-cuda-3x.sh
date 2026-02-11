#!/bin/bash
# run-linux-cuda-3x.sh — 3 CUDA runs, small & medium models on Linux
# Results go to results/psyche-suse/YYYY-MM-DD/scaling-test/run{1,2,3}/
set -euo pipefail

BENCH_DIR=/home/coulof/src/tries/2026-02-05-localscore-llama-bench/localscore-bench
CUDA_BENCH="$BENCH_DIR/backends/cuda/llama-bench"
PLOT_SCRIPT=/home/coulof/src/llm-bench-lab/tools/plot_gpu_usage.py
REPO_DIR=/home/coulof/src/llm-bench-lab
DATE=$(date +%Y-%m-%d)
BASE_DIR="$REPO_DIR/results/psyche-suse/${DATE}/scaling-test"

NUM_RUNS=3

# Small & medium models only (up to ~12B)
MODELS=(
    "llama-3.2-1b|/home/coulof/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    "gemma-3-1b|/home/coulof/models/gemma-3-1b-it-Q4_K_M.gguf"
    "phi-4-mini-3.8b|/home/coulof/models/Phi-4-mini-instruct-Q4_K_M.gguf"
    "ministral-8b|/home/coulof/models/Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    "gemma-3-12b|/home/coulof/models/gemma-3-12b-it-Q4_K_M.gguf"
    "mistral-nemo-12b|/home/coulof/models/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
)

run_single() {
    local model_name="$1"
    local model_path="$2"
    local run_num="$3"
    local run_dir="$4"

    local out_csv="$run_dir/${model_name}-cuda13.csv"
    local out_json="$run_dir/${model_name}-cuda13.json"

    # Skip if already completed
    if [[ -f "$out_json" ]] && python3 -c "
import json
with open('$out_json') as f:
    d = json.load(f)
results = d.get('results', d if isinstance(d, list) else [])
assert len(results) >= 2, 'incomplete'
" 2>/dev/null; then
        echo "⏭️  Skip $model_name CUDA (run $run_num) — already complete"
        return 0
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Run $run_num | $model_name | CUDA 13.1 | Linux"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Remove partial results
    rm -f "$out_csv" "$out_json"

    # Start GPU monitor
    local monitor_pid=""
    if [[ -f "$REPO_DIR/tools/monitor-gpu.sh" ]]; then
        bash "$REPO_DIR/tools/monitor-gpu.sh" "$out_csv" &
        monitor_pid=$!
    fi

    cd "$BENCH_DIR"
    "$CUDA_BENCH" \
        -m "$model_path" \
        -ngl 99 \
        -t 8 \
        -r 3 \
        -o json \
        2>/dev/null > "$out_json" || {
            echo "  ❌ $model_name CUDA failed"
            [[ -n "$monitor_pid" ]] && kill "$monitor_pid" 2>/dev/null || true
            return 1
        }

    # Stop GPU monitor
    [[ -n "$monitor_pid" ]] && kill "$monitor_pid" 2>/dev/null || true
    sleep 1

    echo "  ✅ $model_name CUDA complete"
}

for run in $(seq 1 "$NUM_RUNS"); do
    run_dir="$BASE_DIR/run-$run"
    mkdir -p "$run_dir"

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║     LINUX CUDA SCALING — RUN $run of $NUM_RUNS              ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo "Output: $run_dir"

    for model_def in "${MODELS[@]}"; do
        IFS='|' read -r model_name model_path <<< "$model_def"

        if [[ ! -f "$model_path" ]]; then
            echo "⚠️  Skipping $model_name — model not found: $model_path"
            continue
        fi

        run_single "$model_name" "$model_path" "$run" "$run_dir" || true
    done

    # Commit after each run
    echo ""
    echo "📤 Committing run $run..."
    cd "$REPO_DIR"
    git add results/
    git commit -m "Linux CUDA run $run — small & medium models (${DATE})

Models: llama-3.2-1b, gemma-3-1b, phi-4-mini-3.8b, ministral-8b, gemma-3-12b, mistral-nemo-12b
Backend: CUDA 13.1, Driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
Automated via run-linux-cuda-3x.sh" 2>/dev/null || echo "  (nothing to commit)"
    echo "✅ Run $run committed"
done

echo ""
echo "🏁 All $NUM_RUNS runs complete!"
echo "📤 Pushing branch..."
cd "$REPO_DIR"
git push -u origin "$(git branch --show-current)" 2>/dev/null || echo "⚠️ Push failed"
echo "✅ Done!"
