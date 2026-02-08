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
| gpt-oss-20b-Q2_K | GPT-OSS | 20B | Q2_K | ~11 GB |
| granite-4.0-h-tiny-Q2_K | Granite 4.0 | tiny | Q2_K | ~2.5 GB |
| Qwen3-Coder-Next-UD-Q4_K_XL | Qwen3 Coder | — | Q4_K_XL | ~46 GB |

## Results

### Llama 3.2 1B Q4_K_M — Vulkan vs CUDA (RTX PRO 6000 Blackwell, 2026-02-08)

| Metric | Vulkan | CUDA | Winner |
|--------|--------|------|--------|
| **LocalScore** | **9277** | 8440 | Vulkan +10% |
| **Prompt Processing** | **36,595 t/s** | 32,426 t/s | Vulkan +13% |
| **Token Generation** | **795 t/s** | 734 t/s | Vulkan +8% |
| **Time to First Token** | **36.4 ms** | 39.6 ms | Vulkan -8% |

> llama-bench b7966. Vulkan uses `-dev Vulkan1` (NVIDIA), CUDA auto-selects device 0.

<details>
<summary>Full test breakdown — Vulkan</summary>

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
<summary>Full test breakdown — CUDA</summary>

| Test | PP t/s | TG t/s | TTFT |
|------|--------|--------|------|
| pp1024+tg16 | 42,833 | 792 | 25.17 ms |
| pp4096+tg256 | 29,163 | 800 | 141.70 ms |
| pp2048+tg256 | 38,767 | 800 | 54.08 ms |
| pp2048+tg768 | 38,627 | 349 | 55.88 ms |
| pp1024+tg1024 | 43,458 | 774 | 24.86 ms |
| pp1280+tg3072 | 39,980 | 762 | 33.33 ms |
| pp384+tg1152 | 39,300 | 774 | 11.06 ms |
| pp64+tg1024 | 14,318 | 776 | 5.76 ms |
| pp16+tg1536 | 5,392 | 776 | 4.26 ms |

</details>

**Note:** CUDA test 4 (pp2048+tg768) shows anomalous TG of 349 t/s — possible warm-up or scheduling issue.

---

### Llama 3.2 1B Q4_K_M — RTX PRO 6000 Blackwell (Vulkan, 2026-02-07)

**LocalScore: 9318** (llama-bench b7935)

| Metric | Value |
|--------|-------|
| Token Generation | 797.70 tok/s |
| Prompt Processing | 36,811.14 tok/s |
| Time to First Token | 36.29 ms |

<details>
<summary>Full test breakdown</summary>

| Test | PP t/s | TG t/s | TTFT |
|------|--------|--------|------|
| pp1024+tg16 | 48,869 | 838 | 22.15 ms |
| pp4096+tg256 | 30,562 | 837 | 135.21 ms |
| pp2048+tg256 | 42,211 | 837 | 49.71 ms |
| pp2048+tg768 | 42,228 | 801 | 49.75 ms |
| pp1024+tg1024 | 49,016 | 787 | 22.16 ms |
| pp1280+tg3072 | 45,821 | 733 | 29.30 ms |
| pp384+tg1152 | 46,819 | 780 | 9.48 ms |
| pp64+tg1024 | 20,864 | 786 | 4.34 ms |
| pp16+tg1536 | 4,907 | 776 | 4.55 ms |

</details>

## Methodology

Benchmarks run using [localscore-bench](https://github.com/bauagonzo/localscore-bench) — a Python reimplementation of Mozilla's [LocalScore](https://localscore.ai) using `llama-bench` as the engine.

**Backends:**
- **Vulkan** — from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases) releases
- **CUDA** — from [ai-dock/llama.cpp-cuda](https://github.com/ai-dock/llama.cpp-cuda/releases) releases

**Score formula:** `score = 10 × ∛(avg_prompt_tps × avg_gen_tps × 1000/avg_ttft_ms)`

9 test configurations covering the spectrum from classification (high prompt, low gen) to creative writing (low prompt, high gen).
