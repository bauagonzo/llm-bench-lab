# Vulkan vs CUDA on NVIDIA Blackwell: The Benchmark Nobody Expected

*Tested on RTX PRO 6000 Blackwell Server Edition (96 GB VRAM) with llama.cpp b7966, February 2026*

---

## TL;DR

We benchmarked 8 LLM models (1B to 70B parameters) across two backends, two operating systems, and six runs. Vulkan with coopmat2 challenged CUDA 13.1 on NVIDIA's newest architecture. The results broke every assumption we had.

**The headlines:**

- Mistral Nemo 12B runs **8.3x faster on Vulkan** than CUDA (not a typo)
- Vulkan wins token generation at 32B (+41%) and 70B (+18%)
- Six GPU crashes across all testing: four Vulkan, one CUDA Linux, one CUDA Windows
- CUDA thermal throttles on sustained 32B generation, dropping from 52 t/s to 11.6 t/s
- A mystery driver change boosted CUDA small-model performance 9x between sessions

---

## The Setup

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM |
| CPU | AMD Ryzen 7 9800X3D |
| RAM | 60 GB DDR5 |
| OS | openSUSE Linux 6.18 + Windows (same hardware) |
| Engine | llama.cpp b7966 (Vulkan with NV_coopmat2, CUDA 13.1) |
| Benchmark | localscore-bench: 3 configs per model (pp1024+tg16, pp1024+tg1024, pp16+tg1536) |

All models use Q4_K_M quantization (4-bit). Both backends share the same llama.cpp build version.

---

## The Results

### Linux: Vulkan vs CUDA (Feb 9, Initial Runs)

| Model | Params | Vulkan PP | Vulkan TG | CUDA PP | CUDA TG | Winner |
|-------|--------|-----------|-----------|---------|---------|--------|
| Gemma 3 1B | 1.0B | 3,390 | 61 | 3,273 | **74** | CUDA TG +21% |
| Llama 3.2 1B | 1.2B | **3,526** | **117** | 3,348 | 114 | Vulkan |
| Phi-4 Mini | 3.8B | 1,315 | 45 | **1,490** | **49** | CUDA |
| Ministral 8B | 8.0B | 652 | 29 | **1,086** | **35** | CUDA +67% PP |
| Gemma 3 12B | 11.8B | 5,341 | 117 | **5,729** | **123** | CUDA |
| Mistral Nemo 12B | 12.2B | **4,776** | **51** | 576 | 25 | **Vulkan 8.3x** :material-fire: |
| Qwen3 32B | 32.8B | 1,956 | **58** | **2,221** | 41 | Split |
| Llama 3.3 70B | 70.6B | 1,394 | **30** | **1,613** | 25 | Split |

> PP = Prompt Processing (tokens/sec). TG = Token Generation (tokens/sec). Values are averages across three test configurations.

### What This Means for Practitioners

Care about interactive speed? Vulkan delivers faster token generation for models at 32B and above. CUDA wins prompt processing at most sizes, but the generation story surprised us.

---

## Finding 1: The Mistral Nemo Anomaly

This finding made us recheck our methodology three times.

**Mistral Nemo 12B on CUDA:** 576 t/s prompt processing, 25 t/s generation.
**Mistral Nemo 12B on Vulkan:** 4,776 t/s prompt processing, 51 t/s generation.

That is an 8.3x gap in prompt processing. Same GPU. Same model. Same llama.cpp build.

<!-- CHART: mistral-nemo-12b-cuda13.png -->
*Fig 1: Mistral Nemo 12B on CUDA 13.1. GPU utilization reports 97% average, but power draw tells a different story. The GPU never exceeds 164W on a 250W TDP card. Something is wrong.*

<!-- CHART: mistral-nemo-12b-vulkan.png -->
*Fig 2: Mistral Nemo 12B on Vulkan. Same model, same GPU, completely different behavior. Power peaks at 266W. The GPU works for real this time.*

Under CUDA, the card reports high utilization but draws minimal power. It is mostly idle despite what nvidia-smi claims. Under Vulkan, power draw matches actual computation.

**Our theory:** CUDA's Blackwell kernels hit a suboptimal code path for Mistral Nemo's layer dimensions. This looks like a bug, not a fundamental limitation. We confirmed the anomaly persists across all runs (Feb 9, Feb 11, Feb 12). It is deterministic and reproducible.

**Why this matters:** If you run Mistral Nemo 12B on Blackwell hardware, switching from CUDA to Vulkan gives you a free 8x speedup. No hardware change required.

---

## Finding 2: Token Generation Crosses Back to Vulkan at 32B+

We expected a clean pattern: Vulkan wins small, CUDA wins big. The data refused to cooperate.

For **prompt processing**, CUDA's advantage grows with model size. No surprises there. But for **token generation** (the speed you feel during inference), the story reverses above 32B:

| Model | Vulkan TG | CUDA TG | Delta |
|-------|-----------|---------|-------|
| Ministral 8B | 29 | **35** | CUDA +19% |
| Gemma 3 12B | 117 | **123** | CUDA +6% |
| Qwen3 32B | **58** | 41 | **Vulkan +41%** |
| Llama 3.3 70B | **30** | 25 | **Vulkan +18%** |

<!-- CHART: qwen3-32b-vulkan.png -->
*Fig 3: Qwen3 32B on Vulkan. Steady 400W power draw, temperature climbing from 37C to 80C. Token generation holds at ~58 t/s throughout.*

<!-- CHART: qwen3-32b-cuda13.png -->
*Fig 4: Qwen3 32B on CUDA 13.1. Notice test 3 (pp16+tg1536). Power drops from ~500W to ~150W as temperature hits 95C. Token generation collapses from 52 t/s to 11.6 t/s.*

**The takeaway:** For 32B+ models in interactive use (chatbots, coding assistants), Vulkan delivers faster responses. The 41% advantage on Qwen3 32B translates to noticeably snappier inference.

---

## Finding 3: Both Backends Crash (Not Just Vulkan)

Our initial three runs pointed to a simple story: Vulkan crashes, CUDA stays rock solid. Then we ran more tests. The full picture tells a different story.

### The Complete Crash Log

| # | Date | Model | Backend | OS | Failure |
|---|------|-------|---------|----|---------|
| 1 | Feb 8 | GPT-OSS 20B | Vulkan | Linux | PCIe header corruption |
| 2 | Feb 9 | Llama 3.3 70B | Vulkan | Linux | `vk::DeviceLostError` |
| 3 | Feb 9 | Llama 3.3 70B | Vulkan | Linux | Same crash on retest |
| 4 | Feb 11 | Llama 3.3 70B | CUDA | Linux | Crash dump instead of JSON (run 3) |
| 5 | Feb 11 | Llama 3.3 70B | CUDA | Windows | CPU fallback (GPU unresponsive) |
| 6 | Feb 12 | Ministral 8B | Vulkan | Linux | Error during sustained TG (run 4) |

Six crashes total. Four from Vulkan, two from CUDA. Every 70B attempt on this GPU carries risk regardless of backend.

The Vulkan crashes follow a consistent pattern: sustained token generation triggers a PCIe-level failure. The GPU reports "Unknown header type 7f" via `lspci`. No amount of driver reset recovers it. Only a full power cycle works.

**Crash #6 surprised us most.** Ministral 8B uses only ~5 GB VRAM. Previous crashes all involved models at 20B or larger with 40+ GB allocated. This broke our "large model only" hypothesis. The Vulkan stability issue relates to sustained compute duration, not just memory pressure.

**CUDA's 70B crashes tell a parallel story.** Linux run 3 produced a crash dump instead of benchmark results. Windows refused to load 70B onto the GPU at all, falling back to CPU (1.2 t/s). The 70B model pushes this hardware to its limits on both backends.

<!-- CHART: llama-3.3-70b-vulkan.png -->
*Fig 5: Llama 3.3 70B on Vulkan. Power spikes to 513W during prompt processing, then drops to ~120W. Temperature climbs to 104C even after power drops. The GPU has stopped computing but retains residual heat.*

---

## Finding 4: CUDA Thermal Throttling Under Sustained Load

The Qwen3 32B CUDA run exposed a thermal problem. The benchmark runs three configurations:

1. **pp1024+tg16** (short burst): 52 t/s TG :material-check:
2. **pp1024+tg1024** (medium run): 52 t/s TG :material-check:
3. **pp16+tg1536** (long generation): **11.6 t/s TG** :material-alert:

That is a 4.5x performance cliff within a single benchmark session. Temperature climbed to 95C. NVIDIA's thermal management aggressively downclocked the GPU, dropping power from ~500W to ~150W.

Vulkan avoided this on the same model. It ran at lower power (avg 324W vs CUDA's initial 500W burst). Lower power means lower heat means sustainable performance.

**The lesson:** CUDA's peak performance can mislead you if your workloads run long. Vulkan's more moderate power profile delivers better throughput over extended conversations.

---

## Finding 5: The Ghost in the Driver

Something changed between our Feb 9 and Feb 12 CUDA runs. We did not expect it.

Small model CUDA performance jumped dramatically between sessions. Gemma 3 1B prompt processing went from 4,726 t/s (Feb 9) to 41,590 t/s (Feb 12, run 4). That is an 8.8x increase on the same hardware, same llama.cpp build, same model file.

| Model | CUDA PP (Feb 9) | CUDA PP (Feb 12) | Change |
|-------|-----------------|-------------------|--------|
| Gemma 3 1B | 4,726 | 41,590 | **+780%** |
| Llama 3.2 1B | 4,727 | 44,107 | **+833%** |
| Phi-4 Mini 3.8B | 1,490 | 14,681 | **+885%** |
| Ministral 8B | 1,086 | 8,458 | **+679%** |
| Gemma 3 12B | 5,729 | 5,708 | -0.4% |
| Mistral Nemo 12B | 576 | 577 | +0.2% |

The pattern is sharp. Models under 12B gained 7 to 9x. Models at 12B stayed flat. Both sessions used the same llama.cpp commit (`8872ad2`). We suspect a CUDA driver hotfix landed between sessions, though we cannot confirm the exact change.

**Vulkan told the opposite story.** Across the same gap (Feb 9 to Feb 12), Vulkan throughput barely moved. Gemma 3 1B: 3,390 vs 3,383 (less than 0.2% drift). This level of consistency is exactly what you want from a benchmark.

**Why this matters:** CUDA performance depends on factors outside your control. Driver updates can change results by nearly an order of magnitude. Vulkan provides more predictable, reproducible performance today.

---

## Finding 6: The Real Bottleneck Was Never the Backend

Here is the twist that reframes everything above. Our crashes, thermal throttling, and performance cliffs share a common root cause. It is not Vulkan. It is not CUDA. It is cooling.

### A Passive GPU in a Consumer Case

The RTX PRO 6000 Blackwell Server Edition has no fans. Run `nvidia-smi` and Fan Speed reads N/A. This card draws up to 600W TDP (configurable from 300W to 600W). NVIDIA designed it for rack servers like the [Dell PowerEdge XE9680 and R760xa](https://www.dell.com/en-us/shop/dell-poweredge-servers/sf/poweredge), where engineered front-to-back airflow tunnels force high-pressure air through the heatsink fins. The [Central Computer overview](https://www.centralcomputer.com/blog/post/understanding-the-nvidia-rtx-6000-pro-blackwell-lineup-workstation-max-q-and-server-editions) and [VAST AI comparison](https://vast.ai/article/which-nvidia-rtx-6000-is-right-for-you) both stress this point: the Server Edition requires external chassis airflow. No exceptions.

We put this card in an [Antec C5](https://www.antec.com/product/case/c5) mid-tower case. Seven Antec P12 120mm ARGB fans provide airflow: six reversed as intake (bottom and side panels) and one rear exhaust. The case uses a vertical bottom-to-top airflow scheme. Each P12 pushes roughly 50 to 60 CFM at 1.5 to 2.0 mm H2O static pressure. Total theoretical intake: about 330 CFM.

That sounds like plenty. It is not.

### Why 330 CFM Falls Short

Three problems turn 330 CFM into a fraction of what the GPU needs.

**Direction.** The P12 fans push air vertically and from the side. The GPU heatsink fins run front to back. Air flows around the card, not through it. Without baffles or ducting, the path of least resistance bypasses the heatsink entirely.

**Static pressure.** Dense passive heatsink fins need 5 to 10+ mm H2O of static pressure to force air through them. The P12 fans deliver 1.5 to 2.0 mm H2O. Consumer case fans simply cannot push through server-grade fin density.

**No ducting.** Nothing seals the airflow path. Hot air recirculates. Cold air takes shortcuts. The GPU sits in a pocket of turbulence, not a cooling tunnel.

Realistic estimate: only 30 to 40% of total chassis airflow actually passes through the GPU heatsink. That puts effective GPU airflow at roughly 100 to 130 CFM.

### The Math Says We Are at the Edge

A standard electronic cooling formula gives the required airflow through a heat source:

> **CFM = (Watts x 3.16) / Delta T (degrees C)**

For our 600W card:

| Allowed Air Temp Rise | Required CFM Through Card |
|-----------------------|---------------------------|
| 10 degrees C | ~190 CFM |
| 15 degrees C | ~125 CFM |

Our estimated 100 to 130 CFM sits right at the boundary for a 15 degree C rise. Any sustained load that pushes power above 400W will exceed what our airflow can dissipate. The GPU temperature climbs until thermal protection kicks in.

### Community Results Confirm the Problem

Other builders on [r/LocalLLM](https://www.reddit.com/r/LocalLLM/comments/1mmqghu/rtx_pro_6000_se_is_crushing_it/) tested the same card in non-server enclosures. Their results match our analysis:

| Fan Setup | CFM (Directed) | Idle Temp | Load Temp | Verdict |
|-----------|----------------|-----------|-----------|---------|
| Thermalright TY-143 (single) | ~130 | 50 C | 85 C | Marginal, throttles under sustained load |
| Wathai 120x38mm server fan + custom duct | ~220 | 33 C | 61-62 C | Stable, no throttling |

The difference is not raw CFM. It is directed, high-pressure airflow through the heatsink with proper ducting. The Wathai setup uses a server-grade fan (high static pressure) mounted in a custom shroud that forces every cubic foot of air through the card.

### Reframing Our Results

Go back through Findings 1 through 5 with this lens. The pattern clicks into place.

CUDA's thermal throttling on Qwen3 32B (Finding 4) happened because CUDA bursts to 500W. At that power level, our airflow cannot keep up. Temperature hits 95 C. The GPU downclocks to survive.

Vulkan ran the same model at a steadier 324W average. Lower power means lower heat means sustainable performance. Vulkan did not "handle thermals better." It simply drew less power, staying within our cooling budget.

The 70B crashes (Finding 3) hit both backends at peak power draw. Vulkan's sustained compute patterns push the thermal envelope longer, which explains its higher crash count. But CUDA crashed too when it drew enough power for long enough.

**The real story:** a 600W passive GPU in a consumer mid-tower without directed airflow will hit thermal limits regardless of backend. Vulkan's higher sustained power draw just gets there faster. Our benchmark is testing cooling as much as it is testing backends.

---

## Cross-OS Comparison: Linux vs Windows CUDA

We ran the full suite on Windows (same hardware, driver 582.32) on February 11. After fixing initial GPU access issues (runs 1 through 3 fell back to CPU), runs 4 through 7 produced valid results.

| Model | Linux CUDA PP | Windows CUDA PP | Linux CUDA TG | Windows CUDA TG |
|-------|---------------|-----------------|---------------|-----------------|
| Gemma 3 1B | 28,712 | 27,952 | 539 | 517 |
| Llama 3.2 1B | 31,091 | 30,764 | 785 | 774 |
| Phi-4 Mini 3.8B | 14,681 | 14,631 | 334 | 320 |
| Ministral 8B | 8,458 | 7,767 | 208 | 171 |
| Qwen3 32B | N/A | 2,202 | N/A | 59 |
| Llama 3.3 70B | Crash (run 3) | CPU fallback | N/A | 1.2 |

> Linux values from Feb 12 run 4. Windows values averaged from runs 4 through 7 (GPU-accelerated runs only).

**Key observations:**

- Small models (1B to 8B) perform within 5 to 10% across operating systems. Linux holds a slight edge.
- Qwen3 32B produced valid Windows results (2,202 PP, 59 TG), closely matching Linux Vulkan numbers.
- Llama 3.3 70B failed on both platforms. Linux CUDA crashed on run 3. Windows CUDA fell back to CPU.
- The Mistral Nemo CUDA anomaly persists on Windows (575 PP vs Vulkan's 4,776 PP on Linux). This confirms an architecture-level issue, not an OS-specific driver bug.

---

## What This Means for You

### Models 8B and smaller
Use CUDA. It is faster and more stable after recent driver improvements. Vulkan remains consistent but CUDA now leads by a wide margin on small models.

### Models at 32B for interactive use
Try Vulkan. The token generation advantage is real (+18 to 41%). Vulkan also handles thermals better under sustained load. Expect occasional GPU crashes on long sessions.

### Mistral Nemo 12B specifically
Use Vulkan. No question. An 8.3x speedup is not something you leave on the table. This is almost certainly a CUDA bug that will get fixed, but until then, Vulkan wins by a landslide.

### Production reliability
Neither backend is crash-free at 70B on this hardware. CUDA offers better stability overall (2 failures vs 4 for Vulkan). For mission-critical workloads, add crash recovery to your pipeline regardless of backend choice.

---

## Methodology

- **Engine:** llama.cpp b7966 (same commit for both backends)
- **Vulkan backend:** ggml-org/llama.cpp releases, with NV_coopmat2 cooperative matrix support
- **CUDA backend:** ai-dock/llama.cpp-cuda releases, CUDA 13.1
- **Test suite:** Quick mode with 3 configurations covering short burst (pp1024+tg16), balanced (pp1024+tg1024), and sustained generation (pp16+tg1536)
- **GPU monitoring:** nvidia-smi at 200ms intervals, capturing utilization, power, temperature, and VRAM
- **Quantization:** Q4_K_M for all models (4-bit, medium quality)
- **Runs:** 4 complete runs on Linux (plus 1 aborted), 7 runs on Windows (4 with valid GPU access)
- **All results and raw data:** [github.com/bauagonzo/llm-bench-lab](https://github.com/bauagonzo/llm-bench-lab)

---

## What Comes Next

The Blackwell architecture is new. Drivers change fast. Our next steps:

- **Vulkan on Windows:** Test whether Windows Vulkan drivers include coopmat2 optimizations
- **Driver bisection:** Pin down which CUDA driver update caused the 9x small-model speedup
- **RTX 5090 Ti comparison:** Same test suite on consumer Blackwell silicon
- **Flash Attention investigation:** Determine whether the Feb 12 CUDA boost relates to Flash Attention enablement

We will publish updates in the same repository as results come in.

---

*Benchmarks by Ratatosk Noir and Vedr Vert. Raw data, scripts, and GPU monitoring charts available at [github.com/bauagonzo/llm-bench-lab](https://github.com/bauagonzo/llm-bench-lab).*
