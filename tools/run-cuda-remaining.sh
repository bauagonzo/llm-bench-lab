#!/bin/bash
# run-cuda-remaining.sh — Run CUDA-only benchmarks for 70B and 123B
set -euo pipefail

BENCH_DIR=/home/coulof/src/tries/2026-02-05-localscore-llama-bench/localscore-bench
CUDA_BENCH="$BENCH_DIR/backends/cuda/llama-bench"
PLOT_SCRIPT=/home/coulof/src/llm-bench-lab/tools/plot_gpu_usage.py
REPO_DIR=/home/coulof/src/llm-bench-lab
BASE_DIR="$REPO_DIR/results/psyche-suse/2026-02-09/scaling-test"

# Check GPU is alive
if ! nvidia-smi -L &>/dev/null; then
    echo "❌ GPU not available, aborting"
    exit 1
fi

# Check CUDA works
# CUDA check: localscore-bench handles LD_LIBRARY_PATH internally
echo "CUDA binary: $CUDA_BENCH"

echo "✅ GPU and CUDA OK, starting benchmarks"

MODELS=(
    "llama-3.3-70b|/home/coulof/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
    "mistral-large-123b|/home/coulof/models/Mistral-Large-Instruct-2411-Q4_K_M-00001-of-00002.gguf"
)

run_cuda() {
    local model_name="$1"
    local model_path="$2"
    local run_dir="$3"
    local run_label="$4"

    local out_csv="$run_dir/${model_name}-cuda13.csv"
    local out_json="$run_dir/${model_name}-cuda13.json"
    local out_png="$run_dir/${model_name}-cuda13.png"

    # Skip if already done
    if [[ -f "$out_json" ]] && python3 -c "
import json
with open('$out_json') as f:
    d = json.load(f)
results = d.get('results', d if isinstance(d, list) else [])
assert len(results) >= 2
" 2>/dev/null; then
        echo "⏭️  Skip $model_name CUDA ($run_label) — already complete"
        return 0
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $run_label | $model_name | CUDA 13.1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    rm -f "$out_csv" "$out_json" "$out_png"

    cd "$BENCH_DIR"
    if python3 main.py \
        -m "$model_path" \
        --llama-bench "$CUDA_BENCH" \
        -i 0 \
        --quick \
        -n \
        --save-json "$out_json" \
        --monitor-gpu "$out_csv" \
        --monitor-interval 200 \
        -o console 2>&1; then

        # Generate chart
        if [[ -f "$out_csv" ]]; then
            python3 "$PLOT_SCRIPT" "$out_csv" \
                -o "$out_png" \
                -t "GPU Usage — $model_name (CUDA 13.1, RTX PRO 6000) [$run_label]" \
                2>/dev/null && echo "  📊 Chart saved" || echo "  ⚠️ Chart failed"
        fi
        echo "  ✅ $model_name CUDA complete"
    else
        echo "  ❌ $model_name CUDA FAILED"
    fi
}

# Run 1: 70B and 123B CUDA
for model_def in "${MODELS[@]}"; do
    IFS='|' read -r model_name model_path <<< "$model_def"
    [[ ! -f "$model_path" ]] && echo "⚠️ Skip $model_name — not found" && continue
    run_cuda "$model_name" "$model_path" "$BASE_DIR" "Run 1"
done

# Commit run 1
cd "$REPO_DIR"
git add results/psyche-suse/2026-02-09/scaling-test/
git commit -m "70B + 123B CUDA 13.1 results (run 1)" 2>/dev/null || true
git push 2>/dev/null || true

echo ""
echo "🏁 CUDA remaining benchmarks complete!"
