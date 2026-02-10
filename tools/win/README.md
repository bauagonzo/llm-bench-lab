# Windows Benchmark Scripts

## Quick Start

```powershell
# 1. Check environment
.\setup.ps1

# 2. Run all models (smallest first)
.\run-all-scaling.ps1 -ModelsDir D:\models

# 3. Run a single model
.\run-scaling-test.ps1 -ModelPath D:\models\Qwen3-32B-Q4_K_M.gguf -OutputName qwen3-32b

# 4. Run only one backend
.\run-all-scaling.ps1 -ModelsDir D:\models -Mode vulkan
.\run-all-scaling.ps1 -ModelsDir D:\models -Mode cuda

# 5. Resume after a crash (skip models 1-4)
.\run-all-scaling.ps1 -ModelsDir D:\models -StartFrom 5
```

## Prerequisites

- Python 3.10+
- NVIDIA drivers (nvidia-smi must work)
- `localscore-bench` cloned as sibling directory to `llm-bench-lab`
- llama-bench.exe binaries (Vulkan + CUDA) in `localscore-bench/backends/`

## GPU Device Index

On Linux, Vulkan index 1 = NVIDIA (0 = AMD iGPU). On Windows this may differ.

Check with:
```powershell
.\backends\vulkan\llama-bench.exe --list-devices
.\backends\cuda\llama-bench.exe --list-devices
```

Edit `$VulkanGpuIndex` / `$CudaGpuIndex` in `run-scaling-test.ps1` if needed.

## After Benchmarks

Copy results back to Linux and generate charts:
```bash
python tools/plot_gpu_usage.py results/psyche-win/2026-02-11/scaling-test/*.csv
```
