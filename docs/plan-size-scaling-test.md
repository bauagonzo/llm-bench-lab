# Plan: Vulkan vs CUDA Size-Scaling Test

## Hypothesis
Vulkan (coopmat2) wins on small models (≤7B), CUDA wins on larger models (≥20B).
Crossover point is somewhere in between.

## Test Matrix

We need models spanning 1B → 70B+ across multiple families, all at Q4_K_M for fair comparison.
Using bartowski's GGUF uploads (consistent quanting methodology).

### Models to Download

| # | Family | Model | Params | Q4_K_M Size | HuggingFace Repo | Status |
|---|--------|-------|--------|-------------|------------------|--------|
| 1 | Llama | Llama 3.2 1B | 1.2B | 0.8 GB | ~/models/ | ✅ have |
| 2 | Gemma | Gemma 3 1B | 1.0B | 0.8 GB | ggml-org/gemma-3-1b-it-GGUF | ✅ downloaded |
| 3 | Phi | Phi-4 Mini 3.8B | 3.8B | 2.5 GB | lmstudio-community/Phi-4-mini-instruct-GGUF | ✅ downloaded |
| 4 | Mistral | Ministral 8B | 8.0B | 4.9 GB | bartowski/Ministral-8B-Instruct-2410-GGUF | ✅ downloaded |
| 5 | Gemma | Gemma 3 12B | 12B | 7.3 GB | ggml-org/gemma-3-12b-it-GGUF | ✅ downloaded |
| 6 | Mistral | Mistral Nemo 12B | 12B | 7.5 GB | bartowski/Mistral-Nemo-Instruct-2407-GGUF | ✅ downloaded |
| 7 | Qwen | Qwen3 32B | 32B | 19 GB | via Ollama | ✅ have |
| 8 | Llama | Llama 3.3 70B | 70B | ~42 GB | bartowski/Llama-3.3-70B-Instruct-GGUF | ⏳ downloading |
| 9 | Mistral | Mistral Large 2 123B | 123B | ~73 GB | lmstudio-community/Mistral-Large-Instruct-2411-GGUF (2 splits) | ⏳ downloading |

**Total download: ~145 GB** (excluding what we already have)
**Max single model: ~73 GB** (Mistral Large — fits in 96GB VRAM)

### Size Tiers
- **Tiny (1-2B):** Llama 1B, Gemma 1B → expect Vulkan to dominate
- **Small (3-8B):** Phi-4 Mini, Ministral 8B → expect Vulkan advantage
- **Medium (12-32B):** Gemma 12B, Mistral Nemo, Qwen3 32B → crossover zone
- **Large (70B+):** Llama 70B, Mistral Large 123B → expect CUDA to win

## Execution Plan

### Phase 1: Download models (~1-2 hours depending on bandwidth)
```bash
cd ~/models
# Use huggingface-cli or wget for each model
huggingface-cli download bartowski/<repo> <filename>.gguf --local-dir .
```

### Phase 2: Run benchmarks (both backends, all models)
For each model:
1. Vulkan run with GPU monitoring
2. CUDA run with GPU monitoring
3. Generate comparison charts

Estimated time: ~30-45 min for all runs (small models are fast, large ones take longer)

### Phase 3: Analysis
- Plot t/s vs model size for both backends
- Find the exact crossover point
- Check if the pattern holds across families or is model-specific
- Compare power efficiency (t/s per watt)

## Controls
- Same benchmark parameters: `-p 512 -n 128 -r 3 -ngl 99`
- Same GPU monitoring: 1s interval nvidia-smi
- Same quant: Q4_K_M for all (except models we already have in other quants)
- ⚠️ Different llama.cpp builds (Vulkan b7789 vs CUDA 8872ad2) — known confound

## Risk
- Mistral Large 123B at Q4_K_M is ~73GB — fits in 96GB VRAM but tight
- May want to run it last in case of GPU issues
- Skip if VRAM is insufficient — try Q3_K_M or IQ4_XS instead

## Results Directory
```
results/psyche-suse/2026-02-10/  (or whenever we run)
  scaling-test/
    gemma-3-1b-Q4_K_M-vulkan.csv
    gemma-3-1b-Q4_K_M-cuda13.csv
    ...
```
