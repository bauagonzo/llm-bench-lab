# RTX PRO 6000 Blackwell Server Edition — Thermal Stability Findings

## Summary

The RTX PRO 6000 Blackwell Server Edition (96GB VRAM) experiences GPU crashes under sustained inference workloads when the GPU temperature reaches **104°C** — the thermal shutdown threshold.

## Evidence

### Crash Pattern (consistent across Linux Vulkan and Windows CUDA)

1. Temperature climbs steadily during sustained token generation (TG phase)
2. At 104°C, GPU clock throttles from ~2000 MHz down to **180 MHz**
3. Power draw stays at ~128W despite throttling
4. After ~18 seconds at 104°C, the GPU falls off the PCIe bus ("GPU is lost")
5. `nvidia-smi` reports: `Unable to determine the device handle for GPU0: GPU is lost`
6. No recovery possible without full system reboot
7. No NVIDIA driver events logged in Windows Event Log — silent PCIe disconnect

### Temperature Profile (Mistral Nemo 12B, Windows CUDA, pp16+tg1536 phase)

```
97°C → 98 → 99 → 100 → 101 → 102 → 103 → 104°C (91 samples, ~18s) → [GPU lost]
Clock: 180 MHz (thermal throttled), Power: ~128W
```

### Affected Workloads

- **Models ≥12B parameters** under sustained token generation
- Occurs on both Vulkan (Linux openSUSE) and CUDA 13.1 (Windows 11)
- Small models (1B-8B) run hot (90-104°C) but survive — shorter inference time
- 5 crashes total observed: 3× Linux Vulkan (20B, 70B, 70B), 2× Windows CUDA (12B+)

### Root Cause

The Server Edition card is designed for **server chassis with high-velocity airflow** (passive or blower cooler). When installed in a desktop/workstation case with standard airflow, it cannot dissipate heat fast enough during sustained GPU-bound workloads.

## Mitigation Options

1. **Open case / improve airflow** — immediate, helps but may not be sufficient
2. **Power limit** — `nvidia-smi -pl 100` to cap TDP below 130W, reduces peak temperature
3. **Cooldown between runs** — 60-120s sleep between models to allow thermal recovery
4. **Switch driver** — RTX (Game Ready/Studio) driver vs Tesla/Data Center driver may have different thermal management behavior
5. **Fan curve** — aggressive fan profile if case fans are controllable

## Driver Notes

- **Server driver 591.74** — no Vulkan ICD included (Vulkan unavailable on Windows)
- Plan to test RTX Enterprise Production Branch driver for:
  - Vulkan support
  - Potentially different thermal management / power states
  - Better compatibility with Blackwell compute 12.0 (some models fall back to CPU on CUDA)
