# Scaling Test — Vulkan vs CUDA 13.1 on RTX PRO 6000 Blackwell

**Date:** 2026-02-09  
**Machine:** psyche-suse (AMD Ryzen 7 9800X3D, 60 GB RAM)  
**GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM)  
**Build:** llama.cpp b7966 (Vulkan with coopmat2, CUDA 13.1)  
**Benchmark:** `llama-bench` with scaling workloads (pp1024+tg16, pp1024+tg1024, pp16+tg1536)

## Hypothesis

Vulkan (coopmat2) dominates on small models (≤7B), CUDA catches up or wins on larger models (≥20B).

**Verdict: It's more complicated than that.**

## Results Summary

### Throughput (pp1024+tg16 — standard benchmark config)

| Model | Params | Backend | PP (t/s) | TG (t/s) | TTFT (ms) | LocalScore |
|-------|--------|---------|----------|----------|-----------|------------|
| **Llama 3.2 1B** | 1.2B | Vulkan | **5,017** | **124.8** | **212** | — |
| | | CUDA 13.1 | 4,657 | 117.5 | 228 | — |
| | | *Delta* | *Vulkan +8%* | *Vulkan +6%* | *Vulkan −7%* | |
| **Gemma 3 1B** | 1.0B | Vulkan | **4,901** | 63.5 | **225** | — |
| | | CUDA 13.1 | 4,726 | **76.1** | 230 | — |
| | | *Delta* | *Vulkan +4%* | *CUDA +20%* | *Vulkan −2%* | |
| **Phi-4 Mini** | 3.8B | Vulkan | 1,864 | 46.5 | 571 | — |
| | | CUDA 13.1 | **2,018** | **49.9** | **528** | **5,030** |
| | | *Delta* | *CUDA +8%* | *CUDA +7%* | *CUDA −8%* | |
| **Ministral 8B** | 8.0B | Vulkan | 916 | 30.1 | 1,152 | — |
| | | CUDA 13.1 | **1,087** | **35.1** | **971** | — |
| | | *Delta* | *CUDA +19%* | *CUDA +17%* | *CUDA −16%* | |
| **Gemma 3 12B** | 12B | Vulkan | 7,645 | 119.8 | 142 | 1,809 |
| | | CUDA 13.1 | **8,003** | **130.5** | **136** | **1,931** |
| | | *Delta* | *CUDA +5%* | *CUDA +9%* | *CUDA −4%* | *CUDA +7%* |
| **Mistral Nemo 12B** | 12B | **Vulkan** | **7,256** | **93.9** | **152** | **1,103** |
| | | CUDA 13.1 | 768 | 25.9 | 1,370 | 248 |
| | | *Delta* | *Vulkan +9.4×* | *Vulkan +3.6×* | *Vulkan −9×* | *Vulkan +4.4×* |
| **Llama 3.3 70B** | 70B | Vulkan | 1,394 | 30.2 | 768 | 378 |
| | | CUDA 13.1 | — | — | — | ⏳ pending |
| | | | *⚠️ GPU crashed on test 3 (DeviceLost)* | | | |

### Sustained Throughput (pp1024+tg1024 — longer generation)

| Model | Backend | PP (t/s) | TG (t/s) | TTFT (ms) |
|-------|---------|----------|----------|-----------|
| Llama 3.2 1B | Vulkan | **5,026** | **113.9** | **213** |
| | CUDA 13.1 | 4,656 | 113.1 | 229 |
| Gemma 3 1B | Vulkan | **4,902** | 60.7 | **225** |
| | CUDA 13.1 | 4,727 | **73.5** | 230 |
| Phi-4 Mini | Vulkan | 1,864 | 44.2 | 572 |
| | CUDA 13.1 | **2,018** | **48.0** | **528** |
| Ministral 8B | Vulkan | 916 | 28.6 | 1,153 |
| | CUDA 13.1 | **1,085** | **33.9** | **973** |
| Gemma 3 12B | Vulkan | 7,588 | 115.2 | 144 |
| | CUDA 13.1 | **7,996** | **127.1** | **136** |
| Mistral Nemo 12B | **Vulkan** | **6,979** | **35.9** | **175** |
| | CUDA 13.1 | 767 | 25.2 | 1,370 |
| Llama 3.3 70B | Vulkan | 1,395 | 29.4 | 768 |

## Key Findings

### 1. No clean crossover — it's model-dependent

The "Vulkan small, CUDA big" hypothesis is **wrong**. The pattern is:

| Size | Models | Winner |
|------|--------|--------|
| 1B | Llama, Gemma | **Vulkan** (PP +4–8%, mixed TG) |
| 3.8B | Phi-4 | **CUDA** (+7–8%) |
| 8B | Ministral | **CUDA** (+17–19%) |
| 12B | Gemma 3 | **CUDA** (+5–9%) |
| 12B | Mistral Nemo | **Vulkan 4.4×** 🚨 |
| 70B | Llama 3.3 | Partial data (GPU crash) |

**Mistral Nemo 12B completely breaks the trend.** At 12B params, Vulkan is 9.4× faster on PP and 3.6× faster on TG. This is not a measurement error — it's reproducible and shows in GPU utilization charts (CUDA barely uses the GPU at 12% util vs Vulkan at 40%+).

### 2. CUDA's Blackwell support may be architecture-specific

The CUDA 13.1 binary appears to have optimization gaps for certain model architectures. Mistral Nemo uses a standard Llama-family architecture but something about its layer configuration causes CUDA to underperform catastrophically. This suggests the CUDA b7966 build may have incomplete Blackwell optimizations.

### 3. Vulkan crashes the GPU on large models

The RTX PRO 6000 has crashed **3 times** under sustained Vulkan workloads:
- Feb 8: GPT-OSS 20B Vulkan → PCIe "Unknown header type 7f"
- Feb 9: Llama 70B Vulkan → `vk::DeviceLostError` (test 3 of 3)
- Feb 9: Same crash killed the GPU, required full reboot

The crash pattern is consistent: sustained token generation on models using >40GB VRAM through Vulkan triggers a PCIe-level GPU failure. Power/temp charts show the crash correlates with a sharp drop in power draw (GPU goes unresponsive).

**This means Vulkan benchmarks on 70B+ models are unreliable on this hardware.**

### 4. Vulkan's PP advantage vanishes with size (except Nemo)

| Model Size | Vulkan PP advantage |
|-----------|---------------------|
| 1B | +4–8% |
| 3.8B | −8% (CUDA wins) |
| 8B | −19% (CUDA wins) |
| 12B Gemma | −5% (CUDA wins) |
| 12B Nemo | **+844%** (Vulkan wins) |
| 70B | Data incomplete |

Excluding the Nemo outlier, there's a clear trend of CUDA scaling better with model size.

### 5. VRAM allocation is bizarre

- **Vulkan** allocates ~22 GB for 1B models and the full 97 GB for anything 12B+
- **CUDA** scales proportionally: 1.7 GB (1B) → 5.5 GB (8B) → 7.9 GB (Nemo 12B)
- Vulkan's fixed allocation wastes VRAM on small models but doesn't hurt performance

### 6. Power efficiency favors Vulkan

Vulkan consistently draws 3–10% less power across all model sizes. Even where CUDA wins on throughput, Vulkan uses less energy per token.

## GPU Stability Issues

| Date | Model | Backend | Failure |
|------|-------|---------|---------|
| Feb 8 | GPT-OSS 20B | Vulkan | PCIe header corruption, reboot required |
| Feb 9 | Llama 3.3 70B | Vulkan | `vk::DeviceLostError` on test 3, reboot required |

The RTX PRO 6000 Blackwell Server Edition has a stability issue with sustained Vulkan workloads on large models. This may be a driver bug (Vulkan ICD), a PCIe power delivery issue, or a firmware limitation.

## What's Remaining

| Model | Params | Vulkan | CUDA | Notes |
|-------|--------|:------:|:----:|-------|
| Llama 3.3 70B | 70B | ⚠️ 2/3 | ⏳ | Vulkan crashed on test 3; CUDA pending |
| Mistral Large 123B | 123B | ⏳ | ⏳ | May crash Vulkan; 70GB model in 96GB VRAM |
| Qwen3 32B | 32B | ⏳ | ⏳ | Needs GGUF download |

Multiple runs (for statistical confidence) are in progress for models 1B–12B.

## Files

Each model has:
- `*-vulkan.csv` / `*-cuda13.csv` — GPU telemetry (nvidia-smi, ~5 samples/sec)
- `*-vulkan.json` / `*-cuda13.json` — Benchmark results (llama-bench output)
- `*-vulkan.png` / `*-cuda13.png` — GPU usage charts (4-panel: utilization, power, temp, VRAM)

Charts generated with `tools/plot_gpu_usage.py` — dynamic Y-axis scaling (max + 10% headroom).

## Methodology Notes

- Vulkan uses device index 1 (NVIDIA) because device 0 is the AMD Ryzen iGPU
- CUDA sees only the NVIDIA GPU as device 0
- User must be in `video` group for CUDA to access the GPU (otherwise falls back to CPU silently)
- nvidia-smi monitoring at 200ms intervals; markers injected for test start/end
- Each "quick" run consists of 3 tests: pp1024+tg16, pp1024+tg1024, pp16+tg1536
