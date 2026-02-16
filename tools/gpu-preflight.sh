#!/bin/bash
# gpu-preflight.sh — Pre-flight GPU health check before benchmark runs
# Usage: source gpu-preflight.sh [--model <path>]
#   or:  ./gpu-preflight.sh [--model <path>]
# Returns 0 if GPU is healthy and ready, non-zero otherwise.
# If --model is given, also checks VRAM vs model file size.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL_PATH=""
FAIL=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift 2 ;;
        *) echo "Usage: $0 [--model <path>]" >&2; exit 1 ;;
    esac
done

echo "━━━ GPU Pre-flight Check ━━━"

# 1. nvidia-smi responds at all (GPU not in bad PCIe state)
echo -n "1. nvidia-smi responsive... "
if ! nvidia-smi &>/dev/null; then
    echo -e "${RED}FAIL${NC} — GPU unreachable (PCIe error / bad state). Needs reboot."
    exit 1
fi
echo -e "${GREEN}OK${NC}"

# 2. Parse GPU info
GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_PERF=$(nvidia-smi --query-gpu=pstate --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

echo "   GPU: $GPU_NAME"
echo "   VRAM: ${GPU_MEM_USED}MiB / ${GPU_MEM_TOTAL}MiB"
echo "   Temp: ${GPU_TEMP}°C | Power: ${GPU_POWER}W | Perf: $GPU_PERF"

# 3. Temperature check (100°C = hard limit, warn above 85°C)
echo -n "2. Temperature (< 100°C)... "
if [[ ${GPU_TEMP%.*} -ge 100 ]]; then
    echo -e "${RED}FAIL${NC} — ${GPU_TEMP}°C. GPU overheating. Stop immediately."
    FAIL=1
elif [[ ${GPU_TEMP%.*} -ge 85 ]]; then
    echo -e "${YELLOW}WARN${NC} — ${GPU_TEMP}°C is hot. Watch for throttling."
else
    echo -e "${GREEN}OK${NC} (${GPU_TEMP}°C)"
fi

# 4. No other processes using GPU
echo -n "3. GPU not in use... "
if [[ ${GPU_MEM_USED%.*} -gt 100 ]]; then
    echo -e "${RED}FAIL${NC} — ${GPU_MEM_USED}MiB in use. Another process is running."
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || true
    FAIL=1
else
    echo -e "${GREEN}OK${NC} (${GPU_MEM_USED}MiB used)"
fi

# 5. No uncorrected ECC errors
echo -n "4. ECC errors... "
if [[ "$GPU_ECC" == "N/A" ]] || [[ "$GPU_ECC" == "0" ]]; then
    echo -e "${GREEN}OK${NC} (none)"
else
    echo -e "${RED}FAIL${NC} — $GPU_ECC uncorrected ECC errors. GPU may be unstable."
    FAIL=1
fi

# 6. PCIe link check
echo -n "5. PCIe link... "
PCIE_WIDTH=$(nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null | head -1)
PCIE_GEN=$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader,nounits 2>/dev/null | head -1)
PCIE_MAX_WIDTH=$(nvidia-smi --query-gpu=pcie.link.width.max --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ "$PCIE_WIDTH" != "$PCIE_MAX_WIDTH" ]]; then
    echo -e "${YELLOW}WARN${NC} — PCIe x${PCIE_WIDTH} (max x${PCIE_MAX_WIDTH}). Link degraded."
else
    echo -e "${GREEN}OK${NC} (Gen${PCIE_GEN} x${PCIE_WIDTH})"
fi

# 7. VRAM vs model size check
if [[ -n "$MODEL_PATH" ]]; then
    echo -n "6. Model fits in VRAM... "
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo -e "${RED}FAIL${NC} — Model file not found: $MODEL_PATH"
        FAIL=1
    else
        MODEL_SIZE_MB=$(( $(stat -c%s "$MODEL_PATH") / 1024 / 1024 ))
        VRAM_FREE=$(( ${GPU_MEM_TOTAL%.*} - ${GPU_MEM_USED%.*} ))
        # Need ~10% overhead for KV cache and runtime
        VRAM_NEEDED=$(( MODEL_SIZE_MB + MODEL_SIZE_MB / 10 ))
        if [[ $VRAM_NEEDED -gt $VRAM_FREE ]]; then
            echo -e "${RED}FAIL${NC} — Model ${MODEL_SIZE_MB}MiB + overhead > ${VRAM_FREE}MiB free VRAM"
            FAIL=1
        else
            echo -e "${GREEN}OK${NC} (model ${MODEL_SIZE_MB}MiB, VRAM free ${VRAM_FREE}MiB)"
        fi
    fi
fi

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ $FAIL -ne 0 ]]; then
    echo -e "${RED}PRE-FLIGHT FAILED${NC} — Do not start benchmark."
    exit 1
else
    echo -e "${GREEN}PRE-FLIGHT PASSED${NC} — GPU ready."
    exit 0
fi
