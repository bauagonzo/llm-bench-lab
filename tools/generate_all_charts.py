#!/usr/bin/env python3
"""Generate charts for ALL benchmark JSON files across all results directories.
Creates a PNG next to each JSON that doesn't already have one.

For pairs (vulkan + cuda): comparison chart.
For singles: single-backend chart.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLOR_VULKAN = "#1f6feb"
COLOR_CUDA = "#da3633"
COLOR_BG = "#ffffff"
COLOR_TEXT = "#24292f"
COLOR_GRID = "#d0d7de"
COLOR_SPINE = "#d0d7de"

plt.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_SPINE,
    "axes.labelcolor": COLOR_TEXT,
    "text.color": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "font.family": "sans-serif",
    "font.size": 12,
})


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)


def extract_metrics(data):
    """Extract PP and TG t/s from llama-bench JSON output."""
    results = data if isinstance(data, list) else data.get("results", [])
    # Also check results_summary
    if not results and isinstance(data, dict):
        results = data.get("results_summary", [])
    pp_vals, tg_vals = [], []
    for r in results:
        # Format 1: prompt_tps / gen_tps (localscore-bench)
        if "prompt_tps" in r:
            pp_vals.append(r["prompt_tps"])
        if "gen_tps" in r:
            tg_vals.append(r["gen_tps"])
        # Format 2: n_prompt/n_gen + avg_ts (llama-bench)
        if "avg_ts" in r:
            n_pp = r.get("n_prompt", 0)
            n_tg = r.get("n_gen", 0)
            if n_pp > 0 and n_tg == 0:
                pp_vals.append(r["avg_ts"])
            elif n_tg > 0 and n_pp == 0:
                tg_vals.append(r["avg_ts"])
        # Format 3: test field
        if "test" in r or "type" in r:
            test = r.get("test", r.get("type", ""))
            avg = r.get("tokens_per_second", 0)
            if "pp" in str(test).lower() and avg:
                pp_vals.append(avg)
            elif "tg" in str(test).lower() and avg:
                tg_vals.append(avg)
    pp = sum(pp_vals) / len(pp_vals) if pp_vals else 0
    tg = sum(tg_vals) / len(tg_vals) if tg_vals else 0
    return pp, tg


def generate_single_chart(json_path, backend_label, color, output_path, model_name):
    data = load_json(json_path)
    pp, tg = extract_metrics(data)
    if pp == 0 and tg == 0:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Prompt Processing\n(t/s)", "Token Generation\n(t/s)"], [pp, tg],
                  color=color, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, [pp, tg]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pp, tg)*0.02,
                f"{val:,.0f}", ha="center", va="bottom", fontweight="bold", fontsize=13)

    ax.set_title(f"{model_name} — {backend_label}", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Tokens/sec")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_comparison_chart(vulkan_path, cuda_path, output_path, model_name):
    v_data = load_json(vulkan_path)
    c_data = load_json(cuda_path)
    v_pp, v_tg = extract_metrics(v_data)
    c_pp, c_tg = extract_metrics(c_data)

    if v_pp == 0 and v_tg == 0 and c_pp == 0 and c_tg == 0:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    x = [0, 1]
    w = 0.35
    bars_v = ax.bar([i - w/2 for i in x], [v_pp, v_tg], w,
                    label="Vulkan", color=COLOR_VULKAN, edgecolor="white", linewidth=1.5)
    bars_c = ax.bar([i + w/2 for i in x], [c_pp, c_tg], w,
                    label="CUDA 13.1", color=COLOR_CUDA, edgecolor="white", linewidth=1.5)

    for bars in [bars_v, bars_c]:
        for bar in bars:
            val = bar.get_height()
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + max(v_pp, v_tg, c_pp, c_tg)*0.02,
                        f"{val:,.0f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(["Prompt Processing\n(t/s)", "Token Generation\n(t/s)"])
    ax.set_title(f"{model_name} — Vulkan vs CUDA", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Tokens/sec")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=12)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    repo = Path(__file__).parent.parent
    all_jsons = []
    for d in ["results", "tools/results"]:
        rd = repo / d
        if rd.exists():
            all_jsons.extend(rd.rglob("*.json"))
    all_jsons.sort()
    generated = 0
    skipped = 0

    for jf in all_jsons:
        stem = jf.stem
        parent = jf.parent

        # Determine model name and backend
        if "-vulkan" in stem:
            model_base = stem.replace("-vulkan", "")
            backend = "vulkan"
        elif "-cuda13" in stem:
            model_base = stem.replace("-cuda13", "")
            backend = "cuda13"
        elif "-cuda" in stem:
            model_base = stem.replace("-cuda", "")
            backend = "cuda"
        else:
            continue

        vulkan_json = parent / f"{model_base}-vulkan.json"
        cuda_json = parent / f"{model_base}-cuda13.json"
        comparison_png = parent / f"{model_base}-comparison.png"
        single_png = parent / f"{model_base}-{backend}.png"

        # If both exist and this is the first of the pair, make comparison chart
        if vulkan_json.exists() and cuda_json.exists():
            if not comparison_png.exists() and backend == "vulkan":
                try:
                    if generate_comparison_chart(str(vulkan_json), str(cuda_json),
                                                  str(comparison_png), model_base):
                        generated += 1
                        print(f"✅ {comparison_png.relative_to(repo)}")
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"❌ {model_base}: {e}")
            elif comparison_png.exists():
                skipped += 1
            # Also generate single charts for each
            if not single_png.exists():
                color = COLOR_VULKAN if backend == "vulkan" else COLOR_CUDA
                label = "Vulkan" if backend == "vulkan" else "CUDA 13.1"
                try:
                    if generate_single_chart(str(jf), label, color, str(single_png), model_base):
                        generated += 1
                        print(f"✅ {single_png.relative_to(repo)}")
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"❌ {model_base}: {e}")
            else:
                skipped += 1
        else:
            # Single backend only
            if not single_png.exists():
                color = COLOR_VULKAN if backend == "vulkan" else COLOR_CUDA
                label = "Vulkan" if backend == "vulkan" else "CUDA 13.1"
                try:
                    if generate_single_chart(str(jf), label, color, str(single_png), model_base):
                        generated += 1
                        print(f"✅ {single_png.relative_to(repo)}")
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"❌ {model_base}: {e}")
            else:
                skipped += 1

    print(f"\nDone: {generated} charts generated, {skipped} skipped (already exist or empty)")


if __name__ == "__main__":
    main()
