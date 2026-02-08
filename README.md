# llm-bench-lab

LLM benchmarking results, tools, and analysis.

## Structure

```
results/
  psyche-suse/      # Results organized by machine and date
    2026-02-07/
    2026-02-08/
tools/               # Benchmarking scripts and utilities
docs/                # Notes, methodology, comparisons
```

## Machines

| Machine | CPU | GPU | RAM | OS |
|---------|-----|-----|-----|----|
| psyche-suse | AMD Ryzen 7 9800X3D | NVIDIA RTX PRO 6000 Blackwell Server Edition | 60 GB | openSUSE Linux 6.18 |

## Models

| Model | Family | Params | Quant | Size |
|-------|--------|--------|-------|------|
| Llama-3.2-1B-Instruct-Q4_K_M | Llama 3.2 | 1B | Q4_K_M | ~800 MB |
| granite-4.0-h-tiny-Q2_K | Granite 4.0 | tiny | Q2_K | ~2.5 GB |
| gpt-oss-20b-Q2_K | GPT-OSS | 20B | Q2_K | ~11 GB |
| Qwen3-Coder-Next-UD-Q4_K_XL | Qwen3 Coder | 80B.A3B (MoE) | Q4_K_XL | ~46 GB |

## Results Summary — Vulkan vs CUDA 13.1 (2026-02-08, llama-bench b7966)

| Model | Backend | LocalScore | PP t/s | TG t/s | TTFT |
|-------|---------|------------|--------|--------|------|
| **Llama 3.2 1B Q4_K_M** | Vulkan | **9277** | 36,595 | 795 | 36 ms |
| | CUDA 13.1 | 8638 | 32,494 | 781 | 39 ms |
| | _Delta_ | _Vulk +7%_ | _Vulk +13%_ | _Vulk +2%_ | _Vulk -8%_ |
| **Granite 4.0 tiny Q2_K** | Vulkan | **2950** | 10,032 | 291 | 114 ms |
| | CUDA 13.1 | 970 | 3,752 | 95 | 389 ms |
| | _Delta_ | _Vulk +3x_ | _Vulk +2.7x_ | _Vulk +3.1x_ | _Vulk -3.4x_ |
| **GPT-OSS 20B Q2_K** | Vulkan | **2240** | 6,378 | 318 | 181 ms |
| | CUDA 13.1 | 2048 | 7,121 | 219 | 181 ms |
| | _Delta_ | _Vulk +9%_ | _CUDA +12%_ | _Vulk +45%_ | _Tie_ |
| **Qwen3 Coder 80B.A3B Q4_K_XL** | Vulkan | **124** | 378 | 19 | 3.7 s |
| | CUDA 13.1 ⚠️ | 33 | 67 | 10 | 17.1 s |
| | _Delta_ | _Vulk +3.8x_ | _Vulk +5.6x_ | _Vulk +1.9x_ | _Vulk -4.6x_ |

> ⚠️ Qwen3 CUDA run fell back to CPU (46GB model exceeds VRAM). Not a fair GPU comparison.

### Key Findings

- **Vulkan dominates on Blackwell** across all models tested
- CUDA only wins Prompt Processing on the 20B model (+12%)
- Granite shows the biggest gap: Vulkan 3x faster overall
- CUDA 12.8 binary may not fully optimize for Blackwell (compute capability 12.0)
- Vulkan's NV_coopmat2 support appears well-tuned for this card

---

### Detailed Results

<details>
<summary>Llama 3.2 1B Q4_K_M — Vulkan</summary>

| Test | PP t/s | TG t/s | TTFT |
|------|--------|--------|------|
| pp1024+tg16 | 48,217 | 835 | 22.43 ms |
| pp4096+tg256 | 30,536 | 832 | 135.34 ms |
| pp2048+tg256 | 42,068 | 835 | 49.88 ms |
| pp2048+tg768 | 42,130 | 798 | 49.87 ms |
| pp1024+tg1024 | 48,587 | 784 | 22.35 ms |
| pp1280+tg3072 | 45,525 | 732 | 29.48 ms |
| pp384+tg1152 | 46,693 | 779 | 9.51 ms |
| pp64+tg1024 | 20,703 | 782 | 4.37 ms |
| pp16+tg1536 | 4,892 | 775 | 4.56 ms |

</details>

<details>
<summary>Llama 3.2 1B Q4_K_M — CUDA 13.1</summary>

| Test | PP t/s | TG t/s | TTFT |
|------|--------|--------|------|
| pp1024+tg16 | 43,550 | 794 | 24.77 ms |
| pp4096+tg256 | 29,095 | 801 | 142.03 ms |
| pp2048+tg256 | 38,687 | 800 | 54.19 ms |
| pp2048+tg768 | 38,682 | 770 | 54.24 ms |
| pp1024+tg1024 | 43,692 | 774 | 24.73 ms |
| pp1280+tg3072 | 39,952 | 762 | 33.35 ms |
| pp384+tg1152 | 39,063 | 777 | 11.12 ms |
| pp64+tg1024 | 14,307 | 775 | 5.76 ms |
| pp16+tg1536 | 5,419 | 777 | 4.24 ms |

</details>

<details>
<summary>Granite 4.0 tiny Q2_K — Vulkan</summary>

| Test | PP t/s | TG t/s | TTFT |
|------|--------|--------|------|
| pp1024+tg16 | 12,478 | 289 | 85.53 ms |
| pp4096+tg256 | 12,543 | 293 | 329.96 ms |
| pp2048+tg256 | 12,679 | 293 | 164.94 ms |
| pp2048+tg768 | 12,697 | 292 | 164.72 ms |
| pp1024+tg1024 | 12,537 | 291 | 85.12 ms |
| pp1280+tg3072 | 11,210 | 289 | 117.64 ms |
| pp384+tg1152 | 11,033 | 291 | 38.24 ms |
| pp64+tg1024 | 4,032 | 291 | 19.31 ms |
| pp16+tg1536 | 1,078 | 291 | 18.27 ms |

</details>

<details>
<summary>Granite 4.0 tiny Q2_K — CUDA 13.1</summary>

| Test | PP t/s | TG t/s | TTFT |
|------|--------|--------|------|
| pp1024+tg16 | 9,493 | 198 | 112.93 ms |
| pp4096+tg256 | 6,387 | 170 | 647.20 ms |
| pp2048+tg256 | 6,326 | 158 | 330.06 ms |
| pp2048+tg768 | 6,408 | 118 | 328.06 ms |
| pp1024+tg1024 | 1,972 | 71 | 533.54 ms |
| pp1280+tg3072 | 1,754 | 35 | 758.87 ms |
| pp384+tg1152 | 848 | 35 | 481.65 ms |
| pp64+tg1024 | 407 | 35 | 186.05 ms |
| pp16+tg1536 | 175 | 35 | 120.19 ms |

</details>

<details>
<summary>GPT-OSS 20B Q2_K — Vulkan</summary>

| Test | PP t/s | TG t/s | TTFT |
|------|--------|--------|------|
| pp1024+tg16 | 8,526 | 335 | 123.08 ms |
| pp4096+tg256 | 7,364 | 330 | 559.23 ms |
| pp2048+tg256 | 8,074 | 330 | 256.69 ms |
| pp2048+tg768 | 8,059 | 319 | 257.25 ms |
| pp1024+tg1024 | 8,476 | 306 | 124.09 ms |
| pp1280+tg3072 | 7,581 | 304 | 172.14 ms |
| pp384+tg1152 | 6,975 | 314 | 58.25 ms |
| pp64+tg1024 | 1,859 | 315 | 37.61 ms |
| pp16+tg1536 | 485 | 312 | 36.19 ms |

</details>

<details>
<summary>GPT-OSS 20B Q2_K — CUDA 13.1</summary>

| Test | PP t/s | TG t/s | TTFT |
|------|--------|--------|------|
| pp1024+tg16 | 10,796 | 325 | 97.93 ms |
| pp4096+tg256 | 9,004 | 329 | 457.92 ms |
| pp2048+tg256 | 10,068 | 329 | 206.46 ms |
| pp2048+tg768 | 10,070 | 316 | 206.53 ms |
| pp1024+tg1024 | 10,795 | 297 | 98.23 ms |
| pp1280+tg3072 | 10,115 | 206 | 131.40 ms |
| pp384+tg1152 | 2,610 | 79 | 159.85 ms |
| pp64+tg1024 | 449 | 46 | 164.23 ms |
| pp16+tg1536 | 180 | 46 | 110.26 ms |

</details>

<details>
<summary>Qwen3 Coder 80B.A3B Q4_K_XL — Vulkan</summary>

| Test | PP t/s | TG t/s | TTFT |
|------|--------|--------|------|
| pp1024+tg16 | 423 | 20 | 2.47 s |
| pp4096+tg256 | 415 | 20 | 9.92 s |
| pp2048+tg256 | 420 | 20 | 4.93 s |
| pp2048+tg768 | 420 | 20 | 4.93 s |
| pp1024+tg1024 | 423 | 19 | 2.47 s |
| pp1280+tg3072 | 394 | 19 | 3.30 s |
| pp384+tg1152 | 371 | 19 | 1.09 s |
| pp64+tg1024 | 159 | 14 | 474 ms |

Note: Test 9 (pp16+tg1536) missing — only 8 tests completed.

</details>

<details>
<summary>Qwen3 Coder 80B.A3B Q4_K_XL — CUDA 13.1 (⚠️ CPU fallback)</summary>

| Test | PP t/s | TG t/s | TTFT |
|------|--------|--------|------|
| pp1024+tg16 | 75 | 5 | 13.91 s |
| pp4096+tg256 | 80 | 10 | 51.47 s |
| pp2048+tg256 | 82 | 10 | 25.23 s |
| pp2048+tg768 | 80 | 10 | 25.71 s |
| pp1024+tg1024 | 82 | 10 | 12.59 s |
| pp1280+tg3072 | 77 | 11 | 16.74 s |
| pp384+tg1152 | 71 | 11 | 5.47 s |
| pp64+tg1024 | 40 | 11 | 1.68 s |
| pp16+tg1536 | 19 | 11 | 921.69 ms |

⚠️ Model exceeded VRAM — ran on CPU only.

</details>

---

### Historical Results

<details>
<summary>Llama 3.2 1B Q4_K_M — Vulkan (2026-02-07, llama-bench b7935)</summary>

**LocalScore: 9318**

| Metric | Value |
|--------|-------|
| Token Generation | 797.70 tok/s |
| Prompt Processing | 36,811.14 tok/s |
| Time to First Token | 36.29 ms |

</details>

## Methodology

Benchmarks run using [localscore-bench](https://github.com/bauagonzo/localscore-bench) — a Python reimplementation of Mozilla's [LocalScore](https://localscore.ai) using `llama-bench` as the engine.

**Backends:**
- **Vulkan** — from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases) releases
- **CUDA** — from [ai-dock/llama.cpp-cuda](https://github.com/ai-dock/llama.cpp-cuda/releases) releases

**Score formula:** `score = 10 × ∛(avg_prompt_tps × avg_gen_tps × 1000/avg_ttft_ms)`

9 test configurations covering the spectrum from classification (high prompt, low gen) to creative writing (low prompt, high gen).
