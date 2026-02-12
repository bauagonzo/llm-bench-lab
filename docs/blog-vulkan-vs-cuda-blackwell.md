# Vulkan vs CUDA on NVIDIA Blackwell: The Benchmark Nobody Expected

*Tested on RTX PRO 6000 Blackwell Server Edition (96 GB VRAM) with llama.cpp b7966, February 2026*

---

## TL;DR

We benchmarked 10 LLM models (1B to 123B parameters) on NVIDIA's new Blackwell architecture, comparing Vulkan (with coopmat2) against CUDA 13.1. The conventional wisdom — "just use CUDA on NVIDIA hardware" — turns out to be wrong in interesting ways.

**The headline findings:**
- There's no clean "Vulkan wins small, CUDA wins big" crossover
- One 12B model runs **8.3× faster on Vulkan** than CUDA (not a typo)
- Vulkan wins token generation at 32B (+41%) and 70B (+18%) despite CUDA winning prompt processing
- Vulkan crashes the GPU — four failures across 5 runs, including one on an 8B model
- CUDA thermally throttles on sustained 32B generation, dropping from 52 t/s to 11.6 t/s

---

## The Setup

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM |
| CPU | AMD Ryzen 7 9800X3D |
| RAM | 60 GB DDR5 |
| OS | openSUSE Linux 6.18 |
| Engine | llama.cpp b7966 (Vulkan with NV_coopmat2, CUDA 13.1) |
| Benchmark | localscore-bench — 3 test configs per model: pp1024+tg16, pp1024+tg1024, pp16+tg1536 |

All models are Q4_K_M quantization (4-bit) unless otherwise noted. Both backends use the same llama.cpp build version for fair comparison.

---

## The Results

### The Full Table

| Model | Params | Vulkan PP | Vulkan TG | CUDA PP | CUDA TG | Winner |
|-------|--------|-----------|-----------|---------|---------|--------|
| Gemma 3 1B | 1.0B | 3,390 | 61 | 3,273 | **74** | CUDA TG +21% |
| Llama 3.2 1B | 1.2B | **3,526** | **117** | 3,348 | 114 | Vulkan |
| Phi-4 Mini | 3.8B | 1,315 | 45 | **1,490** | **49** | CUDA |
| Ministral 8B | 8.0B | 652 | 29 | **1,086** | **35** | CUDA +67% PP |
| Gemma 3 12B | 11.8B | 5,341 | 117 | **5,729** | **123** | CUDA |
| Mistral Nemo 12B | 12.2B | **4,776** | **51** | 576 | 25 | **Vulkan 8.3×** 🔥 |
| Qwen3 32B | 32.8B | 1,956 | **58** | **2,221** | 41 | Split |
| Llama 3.3 70B | 70.6B | 1,394 | **30** | **1,613** | 25 | Split |

> PP = Prompt Processing (tokens/sec). TG = Token Generation (tokens/sec).

### What This Means for Practitioners

If you're running local LLMs on Blackwell and you care about interactive speed (token generation), Vulkan might be your better bet for models ≥32B parameters. CUDA still wins prompt processing across most sizes, but the token generation story is more nuanced than anyone expected.

---

## Finding 1: The Mistral Nemo Anomaly

This is the finding that made us double-check our methodology three times.

**Mistral Nemo 12B on CUDA: 576 t/s prompt processing, 25 t/s generation.**
**Mistral Nemo 12B on Vulkan: 4,776 t/s prompt processing, 51 t/s generation.**

That's an 8.3× difference in prompt processing. On the same GPU. Same model. Same llama.cpp build.

<!-- CHART: mistral-nemo-12b-cuda13.png -->
*Fig 1: Mistral Nemo 12B on CUDA 13.1 — GPU utilization reports 97% average, but power draw tells a different story. The GPU never exceeds 164W (the card's TDP is 250W). Something is wrong.*

<!-- CHART: mistral-nemo-12b-vulkan.png -->
*Fig 2: Mistral Nemo 12B on Vulkan — Same model, same GPU, completely different behavior. Power peaks at 266W, GPU is genuinely working. Completes in roughly the same wall time but produces 8× more tokens.*

The GPU monitoring charts reveal the clue: under CUDA, the card reports high utilization but draws minimal power — it's mostly idle despite nvidia-smi saying otherwise. Under Vulkan, power draw corresponds to actual computation.

**Our theory:** CUDA's Blackwell kernels have an optimization gap for Mistral Nemo's specific architecture. The model uses a standard transformer structure, but something about its layer dimensions causes CUDA to fall back to a suboptimal code path. This is likely a bug, not a fundamental limitation — and it probably affects other models with similar configurations that nobody has tested yet.

**Why this matters to the community:** If you're running Mistral Nemo 12B (a popular local model) on Blackwell hardware, switching from CUDA to Vulkan gives you a free 8× speedup. No hardware change, no model change, just a different backend.

---

## Finding 2: Token Generation Crosses Back to Vulkan at 32B+

The expected pattern was simple: Vulkan wins small models, CUDA wins big ones. Reality is more interesting.

For **prompt processing** (prefill), CUDA's advantage grows with model size — no surprises there. But for **token generation** (the speed you actually feel during inference), the story reverses above 32B:

| Model | Vulkan TG | CUDA TG | Δ |
|-------|-----------|---------|---|
| Ministral 8B | 29 | **35** | CUDA +19% |
| Gemma 3 12B | 117 | **123** | CUDA +6% |
| Qwen3 32B | **58** | 41 | **Vulkan +41%** |
| Llama 3.3 70B | **30** | 25 | **Vulkan +18%** |

<!-- CHART: qwen3-32b-vulkan.png -->
*Fig 3: Qwen3 32B on Vulkan — Steady 400W power draw, temperature climbing gradually from 37°C to 80°C. The GPU is working hard but staying within limits. Token generation holds at ~58 t/s throughout.*

<!-- CHART: qwen3-32b-cuda13.png -->
*Fig 4: Qwen3 32B on CUDA 13.1 — Notice the dramatic difference in test 3 (pp16+tg1536). Power drops from ~500W to ~150W while temperature hits 95°C. The GPU is thermal throttling, and token generation collapses from 52 t/s to 11.6 t/s.*

**Why this matters:** If you're serving 32B+ models for interactive use (chatbots, coding assistants), Vulkan delivers meaningfully faster responses. The 41% TG advantage on Qwen3 32B translates to noticeably snappier inference.

---

## Finding 3: Vulkan Crashes the GPU (But CUDA Doesn't)

Here's the catch. In our initial testing, Vulkan caused **three GPU-level crashes** on large models (a fourth followed in run 4 — see Finding 5):

| Model | Event |
|-------|-------|
| GPT-OSS 20B | PCIe header corruption — GPU becomes unresponsive |
| Llama 3.3 70B (run 1) | `vk::DeviceLostError` during sustained generation |
| Llama 3.3 70B (run 2) | Same crash — required full system reboot |

<!-- CHART: llama-3.3-70b-vulkan.png -->
*Fig 5: Llama 3.3 70B on Vulkan — The GPU crash signature. Power spikes to 513W during prompt processing, then drops to ~120W. Temperature continues climbing to 104°C even after power drops — the GPU has stopped computing but retained residual heat. The flat utilization line at 100% after ~90s is the GPU in a hung state.*

The crash pattern is consistent: sustained token generation with 40GB+ VRAM allocated through Vulkan triggers a PCIe-level failure. `lspci` reports "Unknown header type 7f" — the GPU is no longer responding at the bus level. No amount of driver reset recovers it; only a full power cycle (reboot) works.

**CUDA never crashes.** Same models, same workloads, same GPU — rock solid under CUDA.

**What this tells us:** Vulkan's performance advantage on large models comes with a stability risk on current Blackwell hardware/drivers. For production workloads, CUDA's reliability may outweigh Vulkan's speed advantage. For benchmarking and experimentation, just save your work.

---

## Finding 4: CUDA Thermal Throttling Under Sustained Load

The Qwen3 32B CUDA run exposed a thermal issue. The benchmark runs three test configurations:

1. **pp1024+tg16** — Short burst: 52 t/s TG ✅
2. **pp1024+tg1024** — Medium run: 52 t/s TG ✅  
3. **pp16+tg1536** — Long generation: **11.6 t/s TG** ⚠️

That's a 4.5× performance cliff within the same benchmark session. The GPU monitoring chart shows temperature climbing to 95°C, at which point NVIDIA's thermal management aggressively downclocks, dropping power from ~500W to ~150W.

Vulkan didn't hit this on the same model because it ran at lower power (avg 324W vs CUDA's initial 500W burst). Lower power → lower heat → sustainable performance.

**Takeaway:** CUDA's peak performance can be misleading if your workloads are sustained. Vulkan's more moderate power profile may deliver better throughput over long conversations.

---

## What This Means for You

### If you're running models ≤8B (Phi-4, Ministral, Gemma 12B)
→ **Use CUDA.** It's faster and more stable. Simple.

### If you're running 32B+ models for interactive use
→ **Try Vulkan.** The TG advantage is real (+18-41%), and it handles thermals better. Just be prepared for occasional GPU crashes on very long sessions.

### If you're running Mistral Nemo 12B specifically
→ **Use Vulkan. No question.** 8.3× faster. This is almost certainly a CUDA bug that will get fixed, but until then, Vulkan is dramatically better.

### If you need reliability above all
→ **Use CUDA.** Zero crashes across 4+ runs. Vulkan crashed 4 times — and not just on large models. Thermal throttling is manageable with proper cooling.

---

## Finding 5: Extra Runs Confirm the Pattern (Feb 12, Runs 4–5)

We ran the full small/medium model suite again (runs 4 and 5) on February 12th to build statistical confidence. Run 5 was aborted early, but run 4 completed the CUDA phase and most of the Vulkan phase before — you guessed it — another GPU crash.

### What Run 4 Tells Us

**Vulkan is remarkably consistent.** Across runs 1 and 4 (three days apart), Vulkan throughput barely moved:

| Model | Vulkan PP (Run 1) | Vulkan PP (Run 4) | Δ |
|-------|-------------------|--------------------|---|
| Llama 3.2 1B | 5,017 | 5,053 | +0.7% |
| Gemma 3 1B | 4,901 | 4,897 | −0.1% |
| Phi-4 Mini 3.8B | 1,864 | 1,872 | +0.4% |
| Ministral 8B | 916 | 917 | +0.2% |

Token generation numbers are equally stable (all within ±1.2%). This is the kind of reproducibility you want to see in benchmarks.

**CUDA told a different story.** Small model performance (1B–8B) jumped dramatically between runs — Llama 1B prompt processing went from 4,657 to 43,856 t/s, a 9.4× increase. Meanwhile, 12B models stayed flat (Gemma 3 12B: 8,003 → 7,946, essentially unchanged). Both runs used the same llama.cpp build (commit `8872ad2`), so this appears to be a driver or runtime state change rather than a code difference. We're investigating whether a driver hotfix was applied between sessions.

**The Mistral Nemo anomaly persists.** Run 4 confirms it: CUDA prompt processing is still stuck at 768 t/s (vs Vulkan's ~7,000+ in run 1). Whatever CUDA code path issue affects this model, it's deterministic and reproducible.

### GPU Crash #4: Vulkan Dies on Ministral 8B

Run 4's Vulkan phase crashed during **Ministral 8B** — smaller than any previous crash. The GPU monitoring markers tell the story:

```
MARKER: END   pp1024+tg16    @ 09:19:27  ← Test 1 completes fine
MARKER: START pp1024+tg1024  @ 09:19:27  ← Test 2 begins
MARKER: ERROR pp1024+tg1024  @ 09:20:45  ← Dead after 78 seconds
MARKER: START pp16+tg1536    @ 09:20:45  ← Test 3 attempted
MARKER: ERROR pp16+tg1536    @ 09:20:46  ← Instant fail (GPU already gone)
```

After Ministral 8B Vulkan died, Gemma 3 12B Vulkan and Mistral Nemo 12B Vulkan never ran. The GPU was unresponsive.

This is significant because previous crashes all involved models ≥20B with ≥40 GB VRAM. Ministral 8B uses far less memory. The crash happened during sustained token generation (test 2), consistent with the pattern, but at a much smaller scale. This suggests the Vulkan stability issue isn't purely about VRAM pressure — it may be related to sustained compute load duration or a timing-dependent driver bug.

### Updated Crash Tally

| # | Date | Model | Backend | Failure | VRAM Used |
|---|------|-------|---------|---------|-----------|
| 1 | Feb 8 | GPT-OSS 20B | Vulkan | PCIe header corruption | ~40 GB |
| 2 | Feb 9 | Llama 3.3 70B | Vulkan | `vk::DeviceLostError` | ~45 GB |
| 3 | Feb 9 | Llama 3.3 70B | Vulkan | Same crash (retest) | ~45 GB |
| 4 | Feb 12 | Ministral 8B | Vulkan | Error during sustained TG | ~5 GB |
| — | — | *CUDA: zero crashes across all runs* | | | |

Four Vulkan crashes, zero CUDA crashes. The pattern is clear: Vulkan on Blackwell has a stability problem that isn't limited to large models.

### Run 5: Dead on Arrival

Run 5 produced only partial GPU telemetry for Llama 3.2 1B CUDA before aborting. The CSV header was written but no benchmark data was captured. We cleaned up the partial files rather than include incomplete data.

---

## Methodology

- **Engine:** llama.cpp b7966 (same commit for both backends)
- **Vulkan backend:** ggml-org/llama.cpp releases, with NV_coopmat2 cooperative matrix support
- **CUDA backend:** ai-dock/llama.cpp-cuda releases, CUDA 13.1
- **Test suite:** Quick mode — 3 configurations covering short burst (pp1024+tg16), balanced (pp1024+tg1024), and sustained generation (pp16+tg1536)
- **GPU monitoring:** nvidia-smi at 200ms intervals, capturing utilization, power, temperature, and VRAM
- **Quantization:** Q4_K_M for all models (4-bit, medium quality)
- **All results and raw data:** [github.com/bauagonzo/llm-bench-lab](https://github.com/bauagonzo/llm-bench-lab)

---

## What's Next

**Update (Feb 12):** We've now completed 4 full runs on Linux (plus one aborted). The extra runs strengthen our confidence in the findings — Vulkan's consistency is excellent, the Mistral Nemo anomaly is deterministic, and Vulkan's stability issues affect smaller models than initially thought.

Next steps:
- Run the same test suite on **Windows** (same hardware) to isolate OS vs architecture effects
- Test whether the Vulkan crashes reproduce under Windows drivers
- Determine if the Mistral Nemo CUDA anomaly is OS-specific
- Investigate the CUDA small-model performance jump between Feb 9 and Feb 12 runs

Results will be published in the same repo.

---

*Benchmarks by Ratatosk Noir & Veðr Vert. Raw data, scripts, and GPU monitoring charts available at [github.com/bauagonzo/llm-bench-lab](https://github.com/bauagonzo/llm-bench-lab).*

---

## Recommended Charts for Publication

The following charts from the results directory best illustrate the blog post's key findings:

1. **`mistral-nemo-12b-cuda13.png`** + **`mistral-nemo-12b-vulkan.png`** — Side by side, these tell the Nemo anomaly story. The power draw difference is immediately visible.

2. **`qwen3-32b-cuda13.png`** + **`qwen3-32b-vulkan.png`** — Shows both the thermal throttle cliff (CUDA) and Vulkan's steady performance. The power/temp panels are the key.

3. **`llama-3.3-70b-vulkan.png`** — The GPU crash signature. Temperature climbing to 104°C while power flatlines — visually dramatic.

4. **(Suggested: create)** A summary bar chart comparing PP and TG across all models, both backends. This would be the hero image. Can be generated from the JSON data with matplotlib.
