# Plan: Windows Benchmark Runs (2026-02-11)

## Goal
Run the same scaling test on Windows to compare Vulkan/CUDA performance across OS.
Reproduce the Linux results on the same hardware (RTX PRO 6000 Blackwell) under Windows.

## Linux Reference Results (openSUSE, 2026-02-09/10)

### Machine
- CPU: AMD Ryzen 7 9800X3D
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM)
- RAM: 60 GB
- OS: openSUSE Linux 6.18
- llama-bench build: b7966 (commit 8872ad2)

### Scaling Test Results (Quick suite: pp1024+tg16, pp1024+tg1024, pp16+tg1536)

| # | Model | Params | File | Size | Vulkan PP | Vulkan TG | CUDA PP | CUDA TG | Winner |
|---|-------|--------|------|------|-----------|-----------|---------|---------|--------|
| 1 | Gemma 3 1B | 1.0B | gemma-3-1b-it-Q4_K_M.gguf | 769 MB | 3,390 | 61 | 3,273 | **74** | CUDA TG +21% |
| 2 | Llama 3.2 1B | 1.2B | Llama-3.2-1B-Instruct-Q4_K_M.gguf | 771 MB | **3,526** | **117** | 3,348 | 114 | Vulkan |
| 3 | Phi-4 Mini | 3.8B | Phi-4-mini-instruct-Q4_K_M.gguf | 2.4 GB | 1,315 | 45 | **1,490** | **49** | CUDA |
| 4 | Ministral 8B | 8.0B | Ministral-8B-Instruct-2410-Q4_K_M.gguf | 4.6 GB | 652 | 29 | **1,086** | **35** | CUDA |
| 5 | Gemma 3 12B | 11.8B | gemma-3-12b-it-Q4_K_M.gguf | 6.8 GB | 5,341 | 117 | **5,729** | **123** | CUDA |
| 6 | Mistral Nemo 12B | 12.2B | Mistral-Nemo-Instruct-2407-Q4_K_M.gguf | 7.0 GB | **4,776** | **51** | 576 | 25 | Vulkan 8.3× 🔥 |
| 7 | Qwen3 32B | 32.8B | Qwen3-32B-Q4_K_M.gguf | 19 GB | 1,956 | **58** | **2,221** | 41 | Split |
| 8 | Llama 3.3 70B | 70.6B | Llama-3.3-70B-Instruct-Q4_K_M.gguf | 40 GB | 1,394 | **30** | **1,613** | 25 | Split |
| 9 | Mistral Large 123B | 123B | Mistral-Large-*-00001/00002-of-00002.gguf | 70 GB | ❌ | ❌ | ❌ | ❌ | Both failed (RAM) |

> PP = Prompt Processing (t/s avg). TG = Token Generation (t/s avg). Quick suite = 3 test configs.

### Key Linux Findings
- Crossover at ~3-4B: Vulkan wins small models PP, CUDA wins larger
- TG crosses back to Vulkan at 32B+ (+41% at 32B, +18% at 70B)
- Mistral Nemo 12B: massive Vulkan advantage (8.3× PP) — possible CUDA arch gap
- 3 GPU crashes under sustained Vulkan on large models (PCIe DeviceLostError)
- CUDA thermal throttled on Qwen3 32B test 3 (95°C → TG dropped to 11.6 t/s)

## Models to Copy (USB Drive)

**For the 8 working models (skip 123B — needs >60GB RAM):**

| Model | File(s) | Size |
|-------|---------|------|
| Gemma 3 1B | gemma-3-1b-it-Q4_K_M.gguf | 769 MB |
| Llama 3.2 1B | Llama-3.2-1B-Instruct-Q4_K_M.gguf | 771 MB |
| Phi-4 Mini 3.8B | Phi-4-mini-instruct-Q4_K_M.gguf | 2.4 GB |
| Ministral 8B | Ministral-8B-Instruct-2410-Q4_K_M.gguf | 4.6 GB |
| Gemma 3 12B | gemma-3-12b-it-Q4_K_M.gguf | 6.8 GB |
| Mistral Nemo 12B | Mistral-Nemo-Instruct-2407-Q4_K_M.gguf | 7.0 GB |
| Qwen3 32B | Qwen3-32B-Q4_K_M.gguf | 19 GB |
| Llama 3.3 70B | Llama-3.3-70B-Instruct-Q4_K_M.gguf | 40 GB |
| **Total** | | **~81 GB** |

> Optional: also copy Mistral Large 123B (70 GB, 2 files) if Windows machine has >64GB RAM. Total with 123B: ~151 GB.

USB drive: **128 GB minimum** (8 models), 256 GB if including 123B.

**Format: exFAT** (no 4GB file limit, readable by both Linux and Windows).

## Setup Steps (Windows)

### 1. Get llama-bench binaries
Download matching build (b7966 / 8872ad2) from:
- **Vulkan:** https://github.com/ggml-org/llama.cpp/releases → `llama-b7966-bin-win-vulkan-x64.zip`
- **CUDA 13.1:** https://github.com/ai-dock/llama.cpp-cuda/releases → matching build

Or build from source at the same commit for exact parity.

### 2. Install localscore-bench
```powershell
git clone https://github.com/bauagonzo/localscore-bench
cd localscore-bench
pip install -r requirements.txt
```

### 3. Verify GPU access
```powershell
nvidia-smi
# Check: driver version, GPU name, VRAM
# Confirm: same GPU (RTX PRO 6000) or document differences
```

### 4. Run benchmarks
Use the same `run-scaling-test.sh` logic, adapted for Windows:

```powershell
# Per model, Vulkan then CUDA:
python main.py -m <model.gguf> --llama-bench backends\vulkan\llama-bench.exe -i 0 --quick -n --save-json <output>.json --monitor-gpu <output>.csv --monitor-interval 200 -o console

python main.py -m <model.gguf> --llama-bench backends\cuda\llama-bench.exe -i 0 --quick -n --save-json <output>.json --monitor-gpu <output>.csv --monitor-interval 200 -o console
```

> Note: GPU device index may differ on Windows. Check `llama-bench --list-devices` first.

### 5. Run order (smallest first, crash-safe)
1. Gemma 3 1B
2. Llama 3.2 1B
3. Phi-4 Mini 3.8B
4. Ministral 8B
5. Gemma 3 12B
6. Mistral Nemo 12B ← watch for the CUDA anomaly
7. Qwen3 32B
8. Llama 3.3 70B ← may crash Vulkan, run CUDA first

### 6. Collect results
```
results/
  <windows-machine>/
    2026-02-11/
      scaling-test/
        gemma-3-1b-vulkan.{json,csv,png}
        gemma-3-1b-cuda13.{json,csv,png}
        ...
```

## What to Compare

### Primary: Linux vs Windows (same GPU)
- Same model, same backend → OS overhead / driver difference
- Power/thermal behavior under different OS drivers
- Vulkan stability (does Windows avoid the PCIe crashes?)

### Secondary: Vulkan vs CUDA on Windows
- Does the crossover point shift?
- Does Mistral Nemo 12B still show the CUDA anomaly?
- Does CUDA thermal throttle differently under Windows?

### Record
- Windows version + build number
- NVIDIA driver version (`nvidia-smi`)
- GPU name (confirm same card or document)
- Ambient temperature if possible (for thermal comparison)

## Open Questions
- Does Windows Vulkan driver have the same coopmat2 optimizations?
- Will PCIe stability be better under Windows?
- Is the Mistral Nemo CUDA anomaly OS-specific or arch-specific?
