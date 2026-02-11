#!/usr/bin/env python3
"""Generate pretty bar charts from benchmark JSON files."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Theme
COLOR_VULKAN = "#1f6feb"
COLOR_CUDA   = "#da3633"
COLOR_BG     = "#ffffff"
COLOR_TEXT   = "#24292f"
COLOR_GRID   = "#d0d7de"
COLOR_SPINE  = "#d0d7de"

plt.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor":   COLOR_BG,
    "axes.edgecolor":   COLOR_SPINE,
    "axes.labelcolor":  COLOR_TEXT,
    "text.color":       COLOR_TEXT,
    "xtick.color":      COLOR_TEXT,
    "ytick.color":      COLOR_TEXT,
    "font.family":      "sans-serif",
    "font.size":        12,
})


def load_json(filepath: str) -> dict:
    with open(filepath) as f:
        return json.load(f)


def get_localscore(data: dict) -> int:
    """Derive LocalScore from pp1024+tg16 avg_time_ms.

    The localscore-bench tool computes a composite score.  We approximate it
    with a calibration constant derived from known Vulkan result (9277 @ 40.39 ms).
    """
    K = 9277 * 40.39  # calibration constant
    for result in data.get("results", []):
        if result.get("name") == "pp1024+tg16":
            avg = result.get("avg_time_ms", 0)
            if avg > 0:
                return round(K / avg)
    raise ValueError("No pp1024+tg16 result found")


def generate_chart(vulkan_file, cuda_file, model_name, output_path, title=None):
    vulkan_data = load_json(vulkan_file)
    cuda_data   = load_json(cuda_file)

    vulkan_score = get_localscore(vulkan_data)
    cuda_score   = get_localscore(cuda_data)

    delta_pct = round(((vulkan_score - cuda_score) / cuda_score) * 100, 1)
    winner = "Vulkan" if vulkan_score > cuda_score else "CUDA"

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_pos  = [0, 0.8]          # closer together
    width  = 0.55
    scores = [vulkan_score, cuda_score]
    colors = [COLOR_VULKAN, COLOR_CUDA]

    bars = ax.bar(x_pos, scores, width=width, color=colors, edgecolor="none", alpha=0.92)

    # Score labels above bars
    for bar, score, color in zip(bars, scores, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(scores) * 0.015,
            f"{score:,}",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=color,
        )

    # X-axis: bar labels only, no tick marks
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        ["RTX 6000 Linux\nVulkan", "RTX 6000 Linux\nCUDA 13.1"],
        fontsize=11, fontweight="medium",
    )
    ax.tick_params(axis="x", length=0)   # hide tick marks

    # Y-axis
    ax.set_ylabel("LocalScore (higher is better)", fontsize=12, fontweight="medium")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, max(scores) * 1.18)

    # Title
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=14)

    # Grid + spines
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Winner annotation
    symbol = "▲" if winner == "Vulkan" else "▼"
    color  = COLOR_VULKAN if winner == "Vulkan" else COLOR_CUDA
    ax.text(
        0.5, -0.18,
        f"{symbol}  {winner} wins by {abs(delta_pct)}%",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=11, fontweight="semibold", color=color,
    )

    # No legend (labels under bars are self-explanatory)
    ax.legend_ = None

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=COLOR_BG, dpi=200)
    plt.close(fig)

    print(f"✅  {output_path}")
    print(f"   Vulkan: {vulkan_score:,}  |  CUDA: {cuda_score:,}  |  Δ {delta_pct}%")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate benchmark comparison chart")
    p.add_argument("vulkan_file")
    p.add_argument("cuda_file")
    p.add_argument("--model", "-m", default="Model")
    p.add_argument("--output", "-o", default=None)
    args = p.parse_args()

    if args.output:
        out = Path(args.output)
    else:
        slug = args.model.lower().replace(" ", "_").replace("/", "_")
        out = Path(__file__).parent.parent / "docs" / f"{slug}_chart.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    generate_chart(
        args.vulkan_file, args.cuda_file, args.model,
        str(out), title=f"LocalScore: {args.model}",
    )


if __name__ == "__main__":
    main()
