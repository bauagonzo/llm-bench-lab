# llm-bench-lab

LLM benchmarking results, tools, and analysis.

## Structure

```
results/
  by-machine/   # Results organized by hardware
  by-model/     # Results organized by model
tools/           # Benchmarking scripts and utilities
docs/            # Notes, methodology, comparisons
```

## Machines

| Machine | CPU | GPU | RAM | OS |
|---------|-----|-----|-----|----|
| psyche-suse | AMD Ryzen 7 9800X3D | NVIDIA RTX PRO 6000 Blackwell Server Edition | 60 GB | openSUSE Linux 6.18 |

## Models

| Model | Family | Params | Quant | Size |
|-------|--------|--------|-------|------|
| Llama-3.2-1B-Instruct-Q4_K_M | Llama 3.2 | 1B | Q4_K_M | ~800 MB |

## Results

### Llama 3.2 1B Q4_K_M — RTX PRO 6000 Blackwell (2026-02-07)

**LocalScore: 9318**

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

Benchmarks run using [localscore-bench](../src/tries/2026-02-05-localscore-llama-bench/localscore-bench/) — a Python reimplementation of Mozilla's [LocalScore](https://localscore.ai) using `llama-bench` (build 7935) as the engine.

**Score formula:** `score = 10 × ∛(avg_prompt_tps × avg_gen_tps × 1000/avg_ttft_ms)`

9 test configurations covering the spectrum from classification (high prompt, low gen) to creative writing (low prompt, high gen).
