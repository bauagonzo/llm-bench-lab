# Plan: RTX 5090 Ti vs RTX PRO 6000 Comparison

## Goal
Pure GPU-to-GPU comparison on the **same machine** (CPU, RAM, OS, drivers held constant).
Only variable: the GPU card itself.

## Hardware

| | RTX PRO 6000 | RTX 5090 Ti |
|---|---|---|
| Chip | GB202GL | GB203 (TBC) |
| Architecture | Blackwell | Blackwell |
| VRAM | 96 GB GDDR7 | 32 GB GDDR7 |
| TDP | 250W | ~450W (TBC) |
| Interface | PCIe 5.0 x16 | PCIe 5.0 x16 |
| Compute Cap | 12.0 | 12.0 (TBC) |

**Same machine:** AMD Ryzen 7 9800X3D, 60GB RAM, openSUSE Linux 6.18

## Models (7 of 8 — 70B won't fit in 32GB)

| # | Model | GGUF Size | Fits 32GB? |
|---|-------|-----------|------------|
| 1 | Gemma 3 1B | 769 MB | ✅ |
| 2 | Llama 3.2 1B | 771 MB | ✅ |
| 3 | Phi-4 Mini 3.8B | 2.4 GB | ✅ |
| 4 | Ministral 8B | 4.6 GB | ✅ |
| 5 | Gemma 3 12B | 6.8 GB | ✅ |
| 6 | Mistral Nemo 12B | 7.0 GB | ✅ |
| 7 | Qwen3 32B | 19 GB | ✅ |
| 8 | Llama 3.3 70B | 40 GB | ❌ |

## What to Compare

### Primary: PRO 6000 vs 5090 Ti (same model, same backend)
- Raw throughput difference (PP and TG)
- Power efficiency (t/s per watt)
- Thermal behavior (does higher TDP = cooler or hotter?)
- Price/performance ratio

### Secondary: Does the Vulkan vs CUDA story change?
- Does 5090 Ti have the same Vulkan crashes?
- Does the crossover point shift with more CUDA cores but less VRAM?
- Does Mistral Nemo 12B still show the CUDA anomaly?

## Steps

1. Shut down machine, swap GPU
2. Boot, verify `nvidia-smi` sees 5090 Ti
3. Check driver version (may need update)
4. Run: `bash tools/run-5090ti-scaling.sh`
   - Or `bash tools/run-5090ti-scaling.sh cuda-only` if Vulkan crashes
5. Results go to `results/psyche-suse-5090ti/<date>/scaling-test/`
6. Generate comparison charts

## Run Script
`tools/run-5090ti-scaling.sh` — pre-configured for 5090 Ti:
- 7 models (no 70B)
- CUDA runs first (stable), then Vulkan
- Power limit lines at 450W TDP / 600W max
- Memory Y-axis at 32GB
- Pass `cuda-only` argument to skip Vulkan if it crashes

## After Benchmarks
- Generate cross-GPU comparison charts
- Update blog draft with 3-way comparison (PRO 6000 Linux / 5090 Ti Linux / PRO 6000 Windows)
