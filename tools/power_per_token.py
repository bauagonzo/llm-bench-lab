#!/usr/bin/env python3
"""Power-per-token analysis for GPU benchmarks.

Reads nvidia-smi CSV power logs and benchmark JSON results to calculate
energy efficiency (Wh per million tokens) for prompt processing (PP) and
text generation (TG) phases.
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(10_000_000)

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results" / "psyche-suse"
OUTPUT_DIR = RESULTS / "power-analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model size mapping (billions of parameters) for chart ordering
MODEL_SIZES = {
    "gemma3-1b": 1.0, "gemma-3-1b": 1.0,
    "llama32-1b": 1.2, "llama-3.2-1b": 1.2,
    "phi4-mini-3.8b": 3.8, "phi-4-mini-3.8b": 3.8,
    "ministral-8b": 8.0,
    "gemma3-12b": 12.0, "gemma-3-12b": 12.0,
    "mistral-nemo-12b": 12.0,
    "qwen3-32b": 32.0,
    "llama-3.3-70b": 70.0,
}

# Canonical model names for display
CANONICAL_NAMES = {
    "gemma3-1b": "Gemma 3 1B", "gemma-3-1b": "Gemma 3 1B",
    "llama32-1b": "Llama 3.2 1B", "llama-3.2-1b": "Llama 3.2 1B",
    "phi4-mini-3.8b": "Phi-4 Mini 3.8B", "phi-4-mini-3.8b": "Phi-4 Mini 3.8B",
    "ministral-8b": "Ministral 8B",
    "gemma3-12b": "Gemma 3 12B", "gemma-3-12b": "Gemma 3 12B",
    "mistral-nemo-12b": "Mistral Nemo 12B",
    "qwen3-32b": "Qwen3 32B",
    "llama-3.3-70b": "Llama 3.3 70B",
}


def parse_csv_power(csv_path):
    """Parse nvidia-smi CSV and return list of (timestamp_seconds, power_watts)."""
    readings = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Strip whitespace from headers
        header = [h.strip() for h in header]
        
        ts_idx = header.index("timestamp")
        pwr_idx = header.index("power.draw [W]")
        temp_idx = header.index("temperature.gpu") if "temperature.gpu" in header else None
        
        for row in reader:
            if len(row) <= pwr_idx:
                continue
            try:
                ts_str = row[ts_idx].strip()
                dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
                ts_sec = dt.timestamp()
                pwr_str = row[pwr_idx].strip().replace(" W", "")
                pwr = float(pwr_str)
                temp = None
                if temp_idx is not None:
                    temp = float(row[temp_idx].strip())
                readings.append((ts_sec, pwr, temp))
            except (ValueError, IndexError):
                continue
    return readings


def calc_energy_wh(readings):
    """Calculate total energy in Wh from power readings using trapezoidal integration."""
    if len(readings) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    total_joules = 0.0
    for i in range(1, len(readings)):
        dt = readings[i][0] - readings[i-1][0]
        avg_power = (readings[i][1] + readings[i-1][1]) / 2.0
        total_joules += avg_power * dt
    
    duration_s = readings[-1][0] - readings[0][0]
    avg_power = total_joules / duration_s if duration_s > 0 else 0
    temps = [r[2] for r in readings if r[2] is not None]
    avg_temp = np.mean(temps) if temps else 0
    max_temp = max(temps) if temps else 0
    
    return total_joules / 3600.0, duration_s, avg_power, max_temp


def parse_localscore_json(json_path):
    """Parse localscore-bench wrapper JSON. Returns list of result dicts."""
    with open(json_path) as f:
        data = json.load(f)
    
    results = []
    for r in data.get("results", []):
        entry = {
            "n_prompt": r["n_prompt"],
            "n_gen": r["n_gen"],
            "avg_time_ms": r.get("avg_time_ms", 0),
            "prompt_tps": r.get("prompt_tps", 0),
            "gen_tps": r.get("gen_tps", 0),
            "model_n_params": r.get("model_n_params", 0),
        }
        results.append(entry)
    return results


def parse_raw_llama_json(json_path):
    """Parse raw llama-bench JSON array format."""
    with open(json_path) as f:
        data = json.load(f)
    
    results = []
    for r in data:
        is_pp = r["n_prompt"] > 0 and r["n_gen"] == 0
        is_tg = r["n_gen"] > 0 and r["n_prompt"] == 0
        entry = {
            "n_prompt": r["n_prompt"],
            "n_gen": r["n_gen"],
            "avg_ts": r["avg_ts"],
            "avg_ns": r["avg_ns"],
            "prompt_tps": r["avg_ts"] if is_pp else 0,
            "gen_tps": r["avg_ts"] if is_tg else 0,
            "model_n_params": r.get("model_n_params", 0),
        }
        results.append(entry)
    return results


def extract_model_backend(filename):
    """Extract model name and backend from filename like 'gemma3-1b-vulkan.json'."""
    stem = Path(filename).stem  # e.g. gemma3-1b-vulkan
    # Backend is last part after last hyphen that matches known backends
    for backend in ["vulkan", "cuda13"]:
        if stem.endswith(f"-{backend}"):
            model = stem[: -(len(backend) + 1)]
            return model, backend
    return stem, "unknown"


def get_tg_tokens_and_tps(results):
    """Get TG tokens count and tokens/sec from benchmark results.
    Pick the entry with the most generation tokens (typically n_gen=128 or n_gen=1536)."""
    tg_entries = [r for r in results if r.get("gen_tps", 0) > 0 and r["n_gen"] > 0]
    if not tg_entries:
        return 0, 0
    # Use the one with largest n_gen for most representative TG
    best = max(tg_entries, key=lambda r: r["n_gen"])
    return best["n_gen"], best["gen_tps"]


def get_pp_tokens_and_tps(results):
    """Get PP tokens count and tokens/sec."""
    pp_entries = [r for r in results if r.get("prompt_tps", 0) > 0 and r["n_prompt"] > 0]
    if not pp_entries:
        return 0, 0
    best = max(pp_entries, key=lambda r: r["n_prompt"])
    return best["n_prompt"], best["prompt_tps"]


def process_dataset(directory, run_label, gpu_name, is_raw_format=False, note=""):
    """Process all JSON+CSV pairs in a directory."""
    results = []
    directory = Path(directory)
    
    json_files = sorted(directory.glob("*.json"))
    for jf in json_files:
        model, backend = extract_model_backend(jf.name)
        csv_file = jf.with_suffix(".csv")
        
        # Parse JSON
        if is_raw_format:
            bench_results = parse_raw_llama_json(jf)
        else:
            bench_results = parse_localscore_json(jf)
        
        tg_tokens, tg_tps = get_tg_tokens_and_tps(bench_results)
        pp_tokens, pp_tps = get_pp_tokens_and_tps(bench_results)
        
        # Parse CSV if exists
        energy_wh = 0.0
        duration_s = 0.0
        avg_power = 0.0
        max_temp = 0.0
        has_power = False
        
        if csv_file.exists():
            readings = parse_csv_power(csv_file)
            if readings:
                energy_wh, duration_s, avg_power, max_temp = calc_energy_wh(readings)
                has_power = True
        
        # Calculate energy per million tokens
        # Total benchmark duration covers all tests; we estimate per-phase energy
        # by ratio of time spent on each phase
        pp_time_s = pp_tokens / pp_tps if pp_tps > 0 else 0
        tg_time_s = tg_tokens / tg_tps if tg_tps > 0 else 0
        total_bench_time = pp_time_s + tg_time_s
        
        # Energy per million tokens (Wh/MT)
        # Use average power * time for each phase
        if has_power and tg_tps > 0:
            # Wh to process tg_tokens tokens = avg_power * tg_time_s / 3600
            # Per million tokens: scale up
            tg_wh_per_mt = (avg_power * (1.0 / tg_tps) * 1_000_000) / 3600.0
        else:
            tg_wh_per_mt = 0.0
        
        if has_power and pp_tps > 0:
            pp_wh_per_mt = (avg_power * (1.0 / pp_tps) * 1_000_000) / 3600.0
        else:
            pp_wh_per_mt = 0.0
        
        canonical = CANONICAL_NAMES.get(model, model)
        size = MODEL_SIZES.get(model, 0)
        
        results.append({
            "gpu": gpu_name,
            "backend": backend,
            "model": model,
            "model_display": canonical,
            "model_size_b": size,
            "run": run_label,
            "pp_tps": pp_tps,
            "tg_tps": tg_tps,
            "avg_power_w": avg_power,
            "max_temp_c": max_temp,
            "total_energy_wh": energy_wh,
            "duration_s": duration_s,
            "tg_wh_per_mt": tg_wh_per_mt,
            "pp_wh_per_mt": pp_wh_per_mt,
            "has_power": has_power,
            "note": note,
        })
    
    return results


def make_charts(all_results):
    """Generate analysis charts."""
    # Filter to results with power data
    powered = [r for r in all_results if r["has_power"] and r["tg_wh_per_mt"] > 0]
    
    # --- Chart 1: TG Wh/MT across all GPU+backend combos ---
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Group by gpu+backend
    groups = defaultdict(list)
    for r in powered:
        key = f"{r['gpu']} {r['backend'].upper()}"
        groups[key].append(r)
    
    # Sort models by size
    all_models = sorted(set(r["model_display"] for r in powered),
                       key=lambda m: next((r["model_size_b"] for r in powered if r["model_display"] == m), 0))
    
    x = np.arange(len(all_models))
    width = 0.8 / max(len(groups), 1)
    
    for i, (label, entries) in enumerate(sorted(groups.items())):
        vals = []
        for m in all_models:
            match = [e for e in entries if e["model_display"] == m]
            vals.append(match[0]["tg_wh_per_mt"] if match else 0)
        offset = (i - len(groups)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, zorder=3)
        # Add value labels
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Wh per 1M Tokens (TG)")
    ax.set_title("Text Generation Energy Efficiency: Wh per 1M Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tg_wh_per_mt_all.png", dpi=150)
    plt.close()
    
    # --- Chart 2: PRO 6000 vs 5070 Ti comparison (common models, same backend) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Find common models between PRO 6000 and 5070 Ti
    pro_results = {(r["model_display"], r["backend"]): r for r in powered if "PRO 6000" in r["gpu"]}
    ti_results = {(r["model_display"], r["backend"]): r for r in powered if "5070 Ti" in r["gpu"]}
    
    common = []
    for key in sorted(ti_results.keys(), key=lambda k: MODEL_SIZES.get(
            next((r["model"] for r in powered if r["model_display"] == k[0]), ""), 0)):
        if any(k for k in pro_results if k[0] == key[0]):
            common.append(key)
    
    if common:
        labels = [f"{m}\n({b.upper()})" for m, b in common]
        pro_vals = []
        ti_vals = []
        for model_disp, backend in common:
            pro_key = (model_disp, backend)
            pro_vals.append(pro_results.get(pro_key, {}).get("tg_wh_per_mt", 0))
            ti_vals.append(ti_results.get((model_disp, backend), {}).get("tg_wh_per_mt", 0))
        
        x = np.arange(len(common))
        w = 0.35
        ax.bar(x - w/2, pro_vals, w, label="RTX PRO 6000", color="#2196F3", zorder=3)
        ax.bar(x + w/2, ti_vals, w, label="RTX 5070 Ti", color="#FF9800", zorder=3)
        
        for xi, (pv, tv) in enumerate(zip(pro_vals, ti_vals)):
            if pv > 0:
                ax.text(xi - w/2, pv, f'{pv:.2f}', ha='center', va='bottom', fontsize=8)
            if tv > 0:
                ax.text(xi + w/2, tv, f'{tv:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Wh per 1M Tokens (TG)")
        ax.set_title("Energy Efficiency Comparison: RTX PRO 6000 vs RTX 5070 Ti")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pro6000_vs_5070ti.png", dpi=150)
    plt.close()
    
    # --- Chart 3: Power efficiency vs model size ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = {'vulkan': 'o', 'cuda13': 's'}
    colors_gpu = {"RTX PRO 6000": "#2196F3", "RTX 5070 Ti": "#FF9800"}
    
    for r in powered:
        marker = markers.get(r["backend"], 'x')
        color = colors_gpu.get(r["gpu"], "gray")
        label = f"{r['gpu']} {r['backend'].upper()}"
        ax.scatter(r["model_size_b"], r["tg_wh_per_mt"], 
                  marker=marker, c=color, s=80, zorder=3)
    
    # Create legend manually to avoid duplicates
    from matplotlib.lines import Line2D
    handles = []
    seen = set()
    for r in powered:
        key = f"{r['gpu']} {r['backend'].upper()}"
        if key not in seen:
            seen.add(key)
            marker = markers.get(r["backend"], 'x')
            color = colors_gpu.get(r["gpu"], "gray")
            handles.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                                  markersize=10, label=key))
    
    ax.set_xlabel("Model Size (Billion Parameters)")
    ax.set_ylabel("Wh per 1M Tokens (TG)")
    ax.set_title("Energy Cost vs Model Size")
    ax.set_xscale('log')
    ax.legend(handles=handles)
    ax.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "efficiency_vs_model_size.png", dpi=150)
    plt.close()


def generate_markdown(all_results):
    """Generate the analysis markdown document."""
    powered = [r for r in all_results if r["has_power"]]
    unpowered = [r for r in all_results if not r["has_power"]]
    
    md = []
    md.append("# Power-per-Token Analysis")
    md.append("")
    md.append("## Methodology")
    md.append("")
    md.append("Energy efficiency is calculated by combining two data sources:")
    md.append("")
    md.append("1. **Power readings**: nvidia-smi CSV logs at ~200ms intervals recording GPU power draw (W)")
    md.append("2. **Benchmark results**: llama-bench JSON output with tokens processed and throughput (tokens/sec)")
    md.append("")
    md.append("### Energy Calculation")
    md.append("")
    md.append("- Power readings are integrated over time using trapezoidal rule to get total energy (Joules)")
    md.append("- Average power (W) during the benchmark run is computed")
    md.append("- Energy per million tokens: `avg_power_W × (1/tokens_per_sec) × 1,000,000 / 3600` = Wh/MT")
    md.append("- This represents the GPU energy cost to generate 1 million tokens")
    md.append("")
    md.append("### Data Sources")
    md.append("")
    md.append("| Dataset | GPU | Backend | Date | Format | Power Data |")
    md.append("|---------|-----|---------|------|--------|------------|")
    md.append("| PRO 6000 Vulkan | RTX PRO 6000 (96GB) | Vulkan | Feb 9, 2026 | localscore-bench | ✅ |")
    md.append("| PRO 6000 CUDA | RTX PRO 6000 (96GB) | CUDA | Feb 11, 2026 | raw llama-bench | ❌ (no CSV) |")
    md.append("| 5070 Ti Vulkan | RTX 5070 Ti (16GB) | Vulkan | Feb 12, 2026 | localscore-bench | ✅ |")
    md.append("| 5070 Ti CUDA | RTX 5070 Ti (16GB) | CUDA | Feb 12, 2026 | localscore-bench | ✅ |")
    md.append("")
    md.append("> ⚠️ **Note**: The Feb 9 PRO 6000 data for small models (1B, 3.8B) was collected while the GPU")
    md.append("> was in a degraded/thermally throttled state. These efficiency numbers may not represent")
    md.append("> optimal PRO 6000 performance for small models.")
    md.append("")
    
    # Results table
    md.append("## Results: Energy Efficiency (Wh per 1M Tokens)")
    md.append("")
    md.append("### Text Generation (TG)")
    md.append("")
    md.append("| GPU | Backend | Model | Size | TG t/s | Avg Power (W) | Max Temp (°C) | TG Wh/MT | Note |")
    md.append("|-----|---------|-------|------|--------|---------------|---------------|----------|------|")
    
    for r in sorted(powered, key=lambda x: (x["gpu"], x["backend"], x["model_size_b"])):
        if r["tg_wh_per_mt"] > 0:
            note = r["note"]
            md.append(f"| {r['gpu']} | {r['backend'].upper()} | {r['model_display']} | {r['model_size_b']:.1f}B | "
                     f"{r['tg_tps']:.1f} | {r['avg_power_w']:.1f} | {r['max_temp_c']:.0f} | "
                     f"**{r['tg_wh_per_mt']:.3f}** | {note} |")
    
    md.append("")
    md.append("### Prompt Processing (PP)")
    md.append("")
    md.append("| GPU | Backend | Model | Size | PP t/s | Avg Power (W) | PP Wh/MT | Note |")
    md.append("|-----|---------|-------|------|--------|---------------|----------|------|")
    
    for r in sorted(powered, key=lambda x: (x["gpu"], x["backend"], x["model_size_b"])):
        if r["pp_wh_per_mt"] > 0:
            md.append(f"| {r['gpu']} | {r['backend'].upper()} | {r['model_display']} | {r['model_size_b']:.1f}B | "
                     f"{r['pp_tps']:.1f} | {r['avg_power_w']:.1f} | "
                     f"**{r['pp_wh_per_mt']:.4f}** | {r['note']} |")
    
    if unpowered:
        md.append("")
        md.append("### Performance Only (No Power Data)")
        md.append("")
        md.append("| GPU | Backend | Model | Size | PP t/s | TG t/s |")
        md.append("|-----|---------|-------|------|--------|--------|")
        for r in sorted(unpowered, key=lambda x: (x["gpu"], x["backend"], x["model_size_b"])):
            md.append(f"| {r['gpu']} | {r['backend'].upper()} | {r['model_display']} | "
                     f"{r['model_size_b']:.1f}B | {r['pp_tps']:.1f} | {r['tg_tps']:.1f} |")
    
    md.append("")
    md.append("## Charts")
    md.append("")
    md.append("### TG Energy Efficiency Across All Configurations")
    md.append("![TG Wh/MT All](../results/psyche-suse/power-analysis/tg_wh_per_mt_all.png)")
    md.append("")
    md.append("### PRO 6000 vs 5070 Ti Comparison")
    md.append("![PRO 6000 vs 5070 Ti](../results/psyche-suse/power-analysis/pro6000_vs_5070ti.png)")
    md.append("")
    md.append("### Energy Cost vs Model Size")
    md.append("![Efficiency vs Size](../results/psyche-suse/power-analysis/efficiency_vs_model_size.png)")
    md.append("")
    
    # Key findings
    md.append("## Key Findings")
    md.append("")
    
    # Compute some stats
    ti_tg = [r for r in powered if "5070 Ti" in r["gpu"] and r["tg_wh_per_mt"] > 0]
    pro_tg = [r for r in powered if "PRO 6000" in r["gpu"] and r["tg_wh_per_mt"] > 0]
    
    if ti_tg:
        best_ti = min(ti_tg, key=lambda r: r["tg_wh_per_mt"])
        worst_ti = max(ti_tg, key=lambda r: r["tg_wh_per_mt"])
        md.append(f"1. **5070 Ti most efficient model**: {best_ti['model_display']} ({best_ti['backend'].upper()}) "
                 f"at {best_ti['tg_wh_per_mt']:.3f} Wh/MT")
        md.append(f"2. **5070 Ti least efficient model**: {worst_ti['model_display']} ({worst_ti['backend'].upper()}) "
                 f"at {worst_ti['tg_wh_per_mt']:.3f} Wh/MT")
    
    if pro_tg:
        best_pro = min(pro_tg, key=lambda r: r["tg_wh_per_mt"])
        md.append(f"3. **PRO 6000 most efficient model**: {best_pro['model_display']} ({best_pro['backend'].upper()}) "
                 f"at {best_pro['tg_wh_per_mt']:.3f} Wh/MT")
    
    # Compare same models
    md.append("")
    md.append("### GPU Comparison (Same Models)")
    md.append("")
    
    ti_by_model = {(r["model_display"], r["backend"]): r for r in ti_tg}
    pro_by_model = {(r["model_display"], r["backend"]): r for r in pro_tg}
    
    for key in sorted(ti_by_model.keys(), key=lambda k: ti_by_model[k]["model_size_b"]):
        if key in pro_by_model:
            ti_r = ti_by_model[key]
            pro_r = pro_by_model[key]
            ratio = pro_r["tg_wh_per_mt"] / ti_r["tg_wh_per_mt"] if ti_r["tg_wh_per_mt"] > 0 else 0
            winner = "5070 Ti" if ti_r["tg_wh_per_mt"] < pro_r["tg_wh_per_mt"] else "PRO 6000"
            md.append(f"- **{key[0]} ({key[1].upper()})**: PRO 6000 {pro_r['tg_wh_per_mt']:.3f} vs "
                     f"5070 Ti {ti_r['tg_wh_per_mt']:.3f} Wh/MT → **{winner}** more efficient "
                     f"({ratio:.2f}x ratio)")
    
    md.append("")
    md.append("### Thermal Throttling Impact")
    md.append("")
    md.append("The Feb 9 PRO 6000 data was collected during a period of thermal throttling, particularly")
    md.append("affecting small model benchmarks. The GPU was running at reduced clocks (~420 MHz SM)")
    md.append("with temperatures at 90°C. This degrades both throughput and energy efficiency since the")
    md.append("GPU still draws significant idle/base power while producing fewer tokens per second.")
    md.append("")
    md.append("### CUDA vs Vulkan on 5070 Ti")
    md.append("")
    
    ti_cuda = {r["model_display"]: r for r in ti_tg if r["backend"] == "cuda13"}
    ti_vulkan = {r["model_display"]: r for r in ti_tg if r["backend"] == "vulkan"}
    
    for model in sorted(ti_cuda.keys(), key=lambda m: ti_cuda[m]["model_size_b"]):
        if model in ti_vulkan:
            c = ti_cuda[model]
            v = ti_vulkan[model]
            winner = "CUDA" if c["tg_wh_per_mt"] < v["tg_wh_per_mt"] else "Vulkan"
            md.append(f"- **{model}**: CUDA {c['tg_wh_per_mt']:.3f} vs Vulkan {v['tg_wh_per_mt']:.3f} Wh/MT "
                     f"→ **{winner}** more efficient")
    
    md.append("")
    md.append("---")
    md.append(f"*Generated by `tools/power_per_token.py`*")
    md.append("")
    
    return "\n".join(md)


def main():
    all_results = []
    
    # 1. PRO 6000 Vulkan (Feb 9) - localscore format with CSV
    feb9_dir = RESULTS / "2026-02-09" / "scaling-test"
    if feb9_dir.exists():
        # Only process vulkan files (cuda13 from this date also available)
        for jf in sorted(feb9_dir.glob("*.json")):
            model, backend = extract_model_backend(jf.name)
            note = "⚠️ throttled" if MODEL_SIZES.get(model, 99) <= 3.8 else ""
            results = process_dataset.__wrapped__(jf, feb9_dir, "feb9", "RTX PRO 6000", False, note) if False else None
        
        # Process whole directory
        r = process_dataset(feb9_dir, "feb9", "RTX PRO 6000", is_raw_format=False, note="")
        # Add throttling notes for small models
        for entry in r:
            if entry["model_size_b"] <= 3.8:
                entry["note"] = "⚠️ throttled"
        all_results.extend(r)
    
    # 2. PRO 6000 CUDA (Feb 11) - raw llama-bench, NO CSVs
    for run_num in [1, 2, 3]:
        run_dir = RESULTS / "2026-02-11" / "scaling-test" / f"run-{run_num}"
        if run_dir.exists():
            r = process_dataset(run_dir, f"feb11-run{run_num}", "RTX PRO 6000", 
                              is_raw_format=True, note="")
            all_results.extend(r)
            break  # Just use run-1 for now since no power data anyway
    
    # 3. 5070 Ti (Feb 12) - localscore format with CSV
    ti_dir = RESULTS / "5070ti" / "run-1"
    if ti_dir.exists():
        r = process_dataset(ti_dir, "feb12", "RTX 5070 Ti", is_raw_format=False, note="")
        all_results.extend(r)
    
    # Print summary table
    print(f"\n{'GPU':<16} {'Backend':<8} {'Model':<20} {'Size':>5} {'TG t/s':>8} {'Avg W':>7} "
          f"{'TG Wh/MT':>9} {'PP Wh/MT':>9} {'Note'}")
    print("-" * 110)
    for r in sorted(all_results, key=lambda x: (x["gpu"], x["backend"], x["model_size_b"])):
        pwr = f"{r['avg_power_w']:.1f}" if r["has_power"] else "N/A"
        tg = f"{r['tg_wh_per_mt']:.3f}" if r["tg_wh_per_mt"] > 0 else "N/A"
        pp = f"{r['pp_wh_per_mt']:.4f}" if r["pp_wh_per_mt"] > 0 else "N/A"
        print(f"{r['gpu']:<16} {r['backend']:<8} {r['model_display']:<20} {r['model_size_b']:>4.1f}B "
              f"{r['tg_tps']:>8.1f} {pwr:>7} {tg:>9} {pp:>9} {r['note']}")
    
    # Generate charts
    print("\nGenerating charts...")
    make_charts(all_results)
    print(f"Charts saved to {OUTPUT_DIR}/")
    
    # Generate markdown
    md_content = generate_markdown(all_results)
    md_path = BASE / "docs" / "power-per-token-analysis.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_content)
    print(f"Report saved to {md_path}")


if __name__ == "__main__":
    main()
