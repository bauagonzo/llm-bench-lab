#!/bin/bash
# monitor-gpu.sh — Run a command with nvidia-smi GPU logging
# Usage: ./monitor-gpu.sh [options] -- <command> [args...]
#   -o <file>    Output CSV path (default: gpu_YYYYMMDD_HHMMSS.csv)
#   -i <seconds> Sample interval (default: 1)
#   -g <gpu_id>  GPU index (default: 0)

set -euo pipefail

INTERVAL=1
GPU_ID=0
OUTFILE=""

while getopts "o:i:g:" opt; do
    case $opt in
        o) OUTFILE="$OPTARG" ;;
        i) INTERVAL="$OPTARG" ;;
        g) GPU_ID="$OPTARG" ;;
        *) echo "Usage: $0 [-o outfile] [-i interval] [-g gpu_id] -- <command>" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# Skip the -- separator if present
[[ "${1:-}" == "--" ]] && shift

if [[ $# -eq 0 ]]; then
    echo "Error: No command specified" >&2
    echo "Usage: $0 [-o outfile] [-i interval] [-g gpu_id] -- <command>" >&2
    exit 1
fi

[[ -z "$OUTFILE" ]] && OUTFILE="gpu_$(date +%Y%m%d_%H%M%S).csv"

echo "🔍 GPU Monitor: logging to $OUTFILE (GPU $GPU_ID, ${INTERVAL}s interval)"
echo "🚀 Running: $*"

nvidia-smi --id="$GPU_ID" \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.graphics,clocks.current.memory,temperature.gpu \
    --format=csv,nounits \
    -l "$INTERVAL" > "$OUTFILE" &
SMI_PID=$!

trap "kill $SMI_PID 2>/dev/null; wait $SMI_PID 2>/dev/null" EXIT

"$@"
EXIT_CODE=$?

kill $SMI_PID 2>/dev/null
wait $SMI_PID 2>/dev/null
trap - EXIT

LINES=$(wc -l < "$OUTFILE")
echo "✅ Done. GPU log: $OUTFILE ($((LINES - 1)) samples)"
exit $EXIT_CODE
