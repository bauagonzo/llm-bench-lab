# Vulkan vs CUDA on NVIDIA Blackwell: The Benchmark Nobody Expected

*Tested on RTX PRO 6000 Blackwell Server Edition (96 GB VRAM) with llama.cpp b7966, February 2026*

---

## TL;DR

We benchmarked 10 LLM models (1B to 123B parameters) on NVIDIA's new Blackwell architecture, comparing Vulkan (with coopmat2) against CUDA 13.1. The conventional wisdom — "just use CUDA on NVIDIA hardware" — turns out to be wrong in interesting ways.

**The headline findings:**
- There's no clean "Vulkan wins small, CUDA wins big" crossover
- One 12B model runs **8.3× faster on Vulkan** than CUDA (not a typo)
- Vulkan wins token generation at 32B (+41%) and 70B (+18%) despite CUDA winning prompt processing
- Vulkan crashes the GPU on large models — three PCIe-level failures requiring reboots
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

Here's the catch. Across our testing, Vulkan caused **three GPU-level crashes** on large models:

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
→ **Use CUDA.** Zero crashes in our testing. Thermal throttling is manageable with proper cooling.

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

## Update: Windows CUDA Results and Multi-Run Analysis (February 11)

We ran the same models on the same hardware under Windows 11 (CUDA 13.1, driver 582.32) and conducted repeated Linux CUDA runs (3 iterations per model) to measure consistency.

### Windows CUDA Performance

Windows testing used localscore-bench through WSL2 (Fedora 42). Early runs (1-3) silently fell back to CPU due to a driver/permission issue. Runs 4-7 used the GPU properly. We report only the valid GPU runs below.

| Model | Params | Windows PP | Windows TG | Linux PP | Linux TG | Notes |
|-------|--------|------------|------------|----------|----------|-------|
| Llama 3.2 1B | 1.2B | 30,764 | 774 | 3,348 | 114 | See note below |
| Gemma 3 1B | 1.0B | 27,952 | 517 | 3,273 | 74 | See note below |
| Phi-4 Mini | 3.8B | 14,631 | 320 | 1,490 | 49 | See note below |
| Ministral 8B | 8.0B | 7,767 | 172 | 1,086 | 35 | See note below |
| Gemma 3 12B | 11.8B | 2,662 | 34 | 5,729 | 123 | Windows inconsistent |
| Mistral Nemo 12B | 12.2B | 669 | 26 | 576 | 25 | Both low (CUDA bug) |
| Qwen3 32B | 32.8B | 2,202 | 59 | 2,221 | 41 | Comparable PP |

> **Important caveat on the Linux small model numbers:** The Linux CUDA results for models up to 8B from February 9 appear to have suffered from the same "video group" CPU fallback issue we documented earlier. Separate Linux CUDA runs on February 11 using llama-bench (pp512 test) show llama-3.2-1b at 49,329 PP and 836 TG, which exceeds the Windows numbers. The February 9 localscore-bench comparison should not be read as "Windows is 9x faster than Linux." Both OSes achieve similar GPU performance when CUDA is properly configured.

**Key observations:**
- For 12B+ models, Windows CUDA performance was erratic. Gemma 3 12B ranged from PP=75 (CPU fallback) to PP=7,736 across runs.
- Mistral Nemo 12B stayed slow on both OSes under CUDA, confirming the Blackwell CUDA kernel gap is not OS-specific.
- Qwen3 32B completed one good Windows run (PP=2,202, TG=59) before the GPU crashed. This matches the Linux CUDA result almost exactly.
- Llama 3.3 70B never achieved GPU-level performance on Windows. All runs showed PP=12, TG=1.2, suggesting complete CPU fallback at this model size under WSL2.

### Multi-Run Consistency (Linux CUDA, February 11)

We ran each model 3 times sequentially using llama-bench (pp512, tg128) to measure variance and thermal effects.

**Small and medium models (1B to 12B): Rock solid.**

| Model | Run 1 PP | Run 2 PP | Run 3 PP | Variance |
|-------|----------|----------|----------|----------|
| Llama 3.2 1B | 49,329 | 49,299 | 47,975 | <3% |
| Gemma 3 1B | 47,925 | 47,924 | 47,924 | <0.01% |
| Phi-4 Mini 3.8B | 19,917 | 19,668 | 19,626 | <1.5% |
| Ministral 8B | 10,692 | 10,665 | 10,667 | <0.3% |
| Gemma 3 12B | 7,668 | 7,608 | 7,596 | <1% |
| Mistral Nemo 12B | 8,004 | 8,000 | 7,984 | <0.3% |

Token generation showed the same consistency: Mistral Nemo 12B held 143.3, 143.7, and 142.6 t/s across all three runs. These numbers are reliable.

**Large models (32B+): Severe degradation after run 1.**

| Model | Run 1 PP | Run 2 PP | Run 3 PP | Run 1 TG | Run 2 TG | Run 3 TG |
|-------|----------|----------|----------|----------|----------|----------|
| Qwen3 32B | 2,865 | 249 | 249 | 59.4 | 10.1 | 6.3 |
| Llama 3.3 70B | 1,125 | 123 | CRASH | 15.0 | 5.3 | CRASH |

Run 1 delivered expected performance. Run 2 dropped to roughly 10% of run 1 speed, consistent with the GPU falling back to a degraded state or severe thermal throttling. Run 3 of the 70B model crashed the GPU entirely.

**What this tells us:** Sequential benchmarking of 32B+ models without cool-down periods causes cumulative thermal damage to performance. A single run produces reliable numbers. Back-to-back runs do not. If you need multiple measurements of large models, allow the GPU to cool between runs.

### Updated GPU Crash Tally

The crash pattern now spans both backends and both operating systems:

| # | OS | Backend | Model | Trigger |
|---|----|---------|-------|---------|
| 1 | Linux | Vulkan | GPT-OSS 20B | Sustained TG |
| 2 | Linux | Vulkan | Llama 3.3 70B | Sustained TG |
| 3 | Linux | Vulkan | Llama 3.3 70B | Sustained TG |
| 4 | Windows | CUDA | Llama 3.3 70B | Sustained TG |
| 5 | Linux | CUDA | Llama 3.3 70B | Sequential run 3 |

The original blog post framed this as "Vulkan crashes, CUDA doesn't." That was premature. With more data, the pattern is clearer: **the 70B model stresses this GPU beyond its stability limits regardless of backend or OS.** The RTX PRO 6000 can load and run the model, but sustained generation or back-to-back runs trigger PCIe-level failures that require a full reboot to recover.

Vulkan triggers the crash faster (sometimes on the first run), while CUDA survives longer before failing. This may relate to Vulkan's different memory access patterns or power management behavior rather than a fundamental Vulkan bug.

### Revised Recommendations

Based on the full Linux + Windows dataset:

**For models up to 12B:**
- Use CUDA on either OS. Performance is consistent and reliable.
- Exception: Mistral Nemo 12B still runs dramatically faster on Vulkan (8.3x). This CUDA kernel gap affects both Linux and Windows identically.

**For 32B models:**
- First-run performance is nearly identical between Vulkan and CUDA for prompt processing.
- Vulkan still wins token generation (+41% on Qwen3 32B).
- Allow cool-down time between benchmark iterations.

**For 70B+ models:**
- Expect instability on the RTX PRO 6000. Both backends crash eventually.
- If you must run 70B, use CUDA for better crash resistance and keep generation lengths short.
- Do not run back-to-back benchmark iterations.

**For cross-OS deployments:**
- Linux and Windows achieve comparable CUDA performance when properly configured.
- Watch for silent CPU fallback on both OSes (video group on Linux, driver permissions on Windows/WSL2).

---

*Benchmarks by Ratatosk Noir and Vedr Vert. Raw data, scripts, and GPU monitoring charts available at [github.com/bauagonzo/llm-bench-lab](https://github.com/bauagonzo/llm-bench-lab).*
