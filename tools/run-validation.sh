#!/bin/bash
# run-validation.sh — 5x repeat validation runs for CUDA vs Vulkan
# Two models: Gemma 3 12B (most CUDA-favorable) and Mistral Nemo 12B (most Vulkan-favorable)
set -euo pipefail

CUDA_DIR=~/src/tries/2026-02-05-localscore-llama-bench/localscore-bench/backends/cuda
VULKAN_DIR=~/src/tries/2026-02-05-localscore-llama-bench/localscore-bench/backends/vulkan
MODELS_DIR=~/models
TOOLS_DIR=~/src/llm-bench-lab/tools
RESULTS_BASE=~/src/llm-bench-lab/results/psyche-suse/validation-feb15

declare -A MODELS=(
    ["gemma3-12b"]="gemma-3-12b-it-Q4_K_M.gguf"
    ["mistral-nemo-12b"]="Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
)

# Vulkan needs device 1 (NVIDIA), device 0 is AMD iGPU
VULKAN_DEVICE="-dev Vulkan1"

for run in 1 2 3 4 5; do
    for backend in cuda vulkan; do
        RDIR="$RESULTS_BASE/run-$run"
        mkdir -p "$RDIR"

        if [[ "$backend" == "cuda" ]]; then
            BENCH="$CUDA_DIR/llama-bench"
            export LD_LIBRARY_PATH="$CUDA_DIR:${LD_LIBRARY_PATH:-}"
            EXTRA_ARGS=""
        else
            BENCH="$VULKAN_DIR/llama-bench"
            export LD_LIBRARY_PATH="$VULKAN_DIR:${LD_LIBRARY_PATH:-}"
            EXTRA_ARGS="$VULKAN_DEVICE"
        fi

        for name in gemma3-12b mistral-nemo-12b; do
            model_path="$MODELS_DIR/${MODELS[$name]}"

            echo "━━━ Run $run | $backend | $name ━━━"

            # Pre-flight
            if ! "$TOOLS_DIR/gpu-preflight.sh" --model "$model_path" 2>&1 | tail -1; then
                echo "SKIP: preflight failed"
                continue
            fi

            # Run benchmark
            "$TOOLS_DIR/monitor-gpu.sh" -o "$RDIR/${name}-${backend}.csv" -i 1 -g 0 -- \
                "$BENCH" -m "$model_path" \
                -p 1024,1024,16 -n 16,1024,1536 \
                -t 1 -ngl 999 -r 3 -o json \
                $EXTRA_ARGS \
                > "$RDIR/${name}-${backend}.json" 2>&1

            echo "Done: $name ($backend)"

            # Wait for GPU to drop below 75C
            echo "Cooling..."
            while true; do
                TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
                if [[ ${TEMP%.*} -lt 75 ]]; then
                    echo "  GPU at ${TEMP}°C — ready"
                    break
                fi
                sleep 15
            done
        done
    done
done

echo "━━━ ALL VALIDATION RUNS COMPLETE ━━━"
