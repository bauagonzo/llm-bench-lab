#!/bin/bash
# run-linux-cuda-large.sh — 3 CUDA runs, large models on Linux
# Stops immediately if GPU crashes
set -euo pipefail

BENCH_DIR=/home/coulof/src/tries/2026-02-05-localscore-llama-bench/localscore-bench
CUDA_BENCH="$BENCH_DIR/backends/cuda/llama-bench"
REPO_DIR=/home/coulof/src/llm-bench-lab
DATE=$(date +%Y-%m-%d)
BASE_DIR="$REPO_DIR/results/psyche-suse/${DATE}/scaling-test"

NUM_RUNS=3

MODELS=(
    "qwen3-32b|/home/coulof/models/Qwen3-32B-Q4_K_M.gguf"
    "llama-3.3-70b|/home/coulof/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
)

check_gpu() {
    if ! nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
        echo "🔥 GPU CRASH DETECTED — stopping immediately"
        cd "$REPO_DIR"
        git add results/ 2>/dev/null
        git commit -m "Large model runs (partial) — GPU crashed during $(date +%Y-%m-%d)" 2>/dev/null || true
        git push 2>/dev/null || true
        exit 1
    fi
}

for run in $(seq 1 "$NUM_RUNS"); do
    run_dir="$BASE_DIR/run-$run"
    mkdir -p "$run_dir"

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║   LINUX CUDA LARGE — RUN $run of $NUM_RUNS                  ║"
    echo "╚══════════════════════════════════════════════════════╝"

    for model_def in "${MODELS[@]}"; do
        IFS='|' read -r model_name model_path <<< "$model_def"

        local_json="$run_dir/${model_name}-cuda13.json"

        # Skip completed
        if [[ -f "$local_json" ]] && python3 -c "
import json
with open('$local_json') as f: d = json.load(f)
r = d if isinstance(d, list) else d.get('results', [])
assert len(r) >= 2
" 2>/dev/null; then
            echo "⏭️  Skip $model_name (run $run) — already complete"
            continue
        fi

        check_gpu

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Run $run | $model_name | CUDA 13.1 | Linux"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        rm -f "$local_json"

        cd "$BENCH_DIR"
        if ! "$CUDA_BENCH" -m "$model_path" -ngl 99 -t 8 -r 3 -o json 2>/dev/null > "$local_json"; then
            echo "  ❌ $model_name failed"
            check_gpu
            continue
        fi

        check_gpu
        echo "  ✅ $model_name CUDA complete"
    done

    echo "📤 Committing run $run..."
    cd "$REPO_DIR"
    git add results/
    git commit -m "Linux CUDA large run $run — qwen3-32b + llama-3.3-70b (${DATE})" 2>/dev/null || echo "  (nothing to commit)"
    echo "✅ Run $run committed"
done

echo "🏁 All large model runs complete!"
cd "$REPO_DIR"
git push 2>/dev/null || echo "⚠️ Push failed"

# Generate charts
python3 tools/generate_all_charts.py 2>&1
git add results/
git commit -m "Add charts for large model runs" 2>/dev/null || true
git push 2>/dev/null || true
echo "✅ Done!"
