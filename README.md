# llm-bench-lab

LLM benchmarking results on NVIDIA RTX PRO 6000 Blackwell — Vulkan vs CUDA 13.1.

## Machine

| Component | Detail |
|-----------|--------|
| CPU | AMD Ryzen 7 9800X3D |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM) |
| RAM | 60 GB |
| OS | openSUSE Linux 6.18 |

## Scaling Test — Vulkan vs CUDA 13.1 (2026-02-09/10, llama-bench b7966)

10 models from 1B to 123B parameters, tested with both backends.

| Model | Params | Backend | PP t/s | TG t/s | TTFT (ms) | Winner |
|-------|--------|---------|--------|--------|-----------|--------|
| Gemma 3 1B | 1.0B | Vulkan | 3,390 | 61 | 170 | |
| | | CUDA | 3,273 | **74** | 173 | **CUDA TG +21%** |
| Llama 3.2 1B | 1.2B | **Vulkan** | **3,526** | **117** | **155** | **Vulkan +5% PP, +2% TG** |
| | | CUDA | 3,348 | 114 | 163 | |
| Phi-4 Mini | 3.8B | CUDA | **1,490** | **49** | **371** | **CUDA +13% PP, +8% TG** |
| | | Vulkan | 1,315 | 45 | 413 | |
| Ministral 8B | 8.0B | CUDA | **1,086** | **35** | 972 | **CUDA +67% PP, +19% TG** |
| | | Vulkan | 652 | 29 | **823** | |
| Gemma 3 12B | 11.8B | CUDA | **5,729** | **123** | **98** | **CUDA +7% PP, +6% TG** |
| | | Vulkan | 5,341 | 117 | 105 | |
| Mistral Nemo 12B | 12.2B | **Vulkan** | **4,776** | **51** | **181** | **Vulkan 8.3× PP, 2× TG** 🔥 |
| | | CUDA | 576 | 25 | 956 | |
| Qwen3 32B | 32.8B | Vulkan | 1,956 | **58** | 280 | **Split: CUDA PP +14%, Vulkan TG +41%** |
| | | CUDA | **2,221** | 41 | **276** | |
| Llama 3.3 70B | 70.6B | CUDA | **1,613** | 25 | **677** | **Split: CUDA PP +16%, Vulkan TG +18%** |
| | | Vulkan | 1,394 | **30** | 768 | |
| Mistral Large 123B | 123B | CUDA | partial ⚠️ | — | — | Both failed (exceeds RAM) |
| | | Vulkan | — | — | — | |

> PP = Prompt Processing (tokens/sec). TG = Token Generation (tokens/sec). TTFT = Time To First Token.
> All values are averages across the quick test suite (3 test configs).

### Key Findings

1. **Crossover at ~3-4B parameters:** Vulkan wins ≤1.2B, CUDA generally wins ≥3.8B for prompt processing
2. **Token generation is more nuanced:** Vulkan wins TG at 32B (+41%) and 70B (+18%) despite losing PP
3. **Mistral Nemo 12B is a massive outlier:** Vulkan is 8.3× faster PP and 2× faster TG — CUDA barely uses the GPU (possible architecture gap in CUDA kernel support)
4. **CUDA 13.1 + Blackwell stability:** CUDA never crashed. Vulkan caused 3 GPU crashes (PCIe `DeviceLostError`) on sustained generation with large models (20B+, 70B)
5. **123B models** need >60GB RAM — causes swap thrashing on this machine, both backends fail
6. **Vulkan coopmat2** is well-optimized for Blackwell on smaller models
7. **CUDA `video` group is required** — without it, silently falls back to CPU (~10× slower)

### Crossover Visualization

```
PP Winner by Model Size:

1B   ████████████████ Vulkan (+3-5%)
1.2B ████████████████ Vulkan (+5%)
3.8B ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CUDA (+13%)
8B   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CUDA (+67%)
12B  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CUDA (+7%)  [Nemo: Vulkan 8.3×]
32B  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CUDA (+14%)
70B  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CUDA (+16%)

TG Winner by Model Size:

1B   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CUDA (+21%)  [Gemma only]
1.2B ████████████████ Vulkan (+2%)
3.8B ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CUDA (+8%)
8B   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CUDA (+19%)
12B  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CUDA (+6%)  [Nemo: Vulkan 2×]
32B  ████████████████ Vulkan (+41%)  ← crossover back
70B  ████████████████ Vulkan (+18%)
```

### GPU Stability Notes

- **3 GPU crashes** under sustained Vulkan on large models (20B, 70B) — PCIe `vk::DeviceLostError`
- Pattern: happens during sustained token generation with 40GB+ VRAM usage
- PCIe rescan never recovers — requires full reboot
- CUDA never triggers this issue
- Vulkan device 0 = AMD iGPU, device 1 = NVIDIA; CUDA device 0 = NVIDIA only

---

## Earlier Results — Vulkan vs CUDA 13.1 (2026-02-08, 4 models)

| Model | Backend | LocalScore | PP t/s | TG t/s | TTFT |
|-------|---------|------------|--------|--------|------|
| **Llama 3.2 1B Q4_K_M** | Vulkan | **9277** | 36,595 | 795 | 36 ms |
| | CUDA 13.1 | 8638 | 32,494 | 781 | 39 ms |
| **Granite 4.0 tiny Q2_K** | Vulkan | **2950** | 10,032 | 291 | 114 ms |
| | CUDA 13.1 | 970 | 3,752 | 95 | 389 ms |
| **GPT-OSS 20B Q2_K** | Vulkan | **2240** | 6,378 | 318 | 181 ms |
| | CUDA 13.1 | 2048 | 7,121 | 219 | 181 ms |
| **Qwen3 Coder 80B.A3B Q4_K_XL** | Vulkan | **124** | 378 | 19 | 3.7 s |
| | CUDA 13.1 ⚠️ | 33 | 67 | 10 | 17.1 s |

> ⚠️ Qwen3 CUDA run fell back to CPU (46GB model exceeds VRAM without Vulkan's smarter memory management).

> Note: Feb 8 tests used the full 9-test LocalScore suite. Feb 9/10 scaling tests used the quick 3-test suite for faster iteration.

---

## Structure

```
results/
  psyche-suse/
    2026-02-07/          # Initial Vulkan benchmarks
    2026-02-08/          # First Vulkan vs CUDA comparison (4 models)
    2026-02-09/
      scaling-test/      # Full scaling test (10 models, Vulkan + CUDA)
tools/                   # Benchmarking scripts and chart generators
docs/                    # Charts, methodology notes
```

## Methodology

Benchmarks run using [localscore-bench](https://github.com/bauagonzo/localscore-bench) — a Python reimplementation of Mozilla's [LocalScore](https://localscore.ai) using `llama-bench` as the engine.

**Backends:**
- **Vulkan** — from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases) releases (with NV_coopmat2)
- **CUDA 13.1** — from [ai-dock/llama.cpp-cuda](https://github.com/ai-dock/llama.cpp-cuda/releases) releases

**Quick test suite (scaling test):** 3 configurations — pp1024+tg16, pp1024+tg1024, pp16+tg1536

**Full test suite (LocalScore):** 9 configurations covering classification → creative writing spectrum

**Score formula:** `LocalScore = 10 × ∛(avg_prompt_tps × avg_gen_tps × 1000/avg_ttft_ms)`

## GPU Monitoring Charts

Each benchmark includes an nvidia-smi GPU monitoring chart (PNG) showing utilization, VRAM, temperature, and power over time with test phase markers.

Generated by `tools/plot_gpu_usage.py` from CSV data collected at 200ms intervals.
