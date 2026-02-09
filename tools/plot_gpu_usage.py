#!/usr/bin/env python3
"""Plot GPU usage from nvidia-smi CSV logs produced by localscore-bench --monitor-gpu.

Usage:
    python plot_gpu_usage.py gpu_log.csv                     # auto-detect & plot all
    python plot_gpu_usage.py gpu_log.csv -o chart.png        # save to file
    python plot_gpu_usage.py gpu_log.csv --metrics power,temp # specific metrics
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# ---------- Theme ----------
COLOR_BG   = "#ffffff"
COLOR_TEXT  = "#24292f"
COLOR_GRID  = "#d0d7de"
COLOR_SPINE = "#d0d7de"

METRIC_COLORS = {
    "gpu_util":    "#1f6feb",   # blue
    "mem_util":    "#8250df",   # purple
    "power":       "#da3633",   # red
    "temperature": "#bf8700",   # amber
    "mem_used":    "#1a7f37",   # green
    "sm_clock":    "#0969da",   # dark blue
    "mem_clock":   "#6639ba",   # deep purple
}

METRIC_LABELS = {
    "gpu_util":    "GPU Utilization (%)",
    "mem_util":    "Memory Controller (%)",
    "power":       "Power Draw (W)",
    "temperature": "Temperature (°C)",
    "mem_used":    "Memory Used (MiB)",
    "sm_clock":    "SM Clock (MHz)",
    "mem_clock":   "Memory Clock (MHz)",
}

plt.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor":   COLOR_BG,
    "axes.edgecolor":   COLOR_SPINE,
    "axes.labelcolor":  COLOR_TEXT,
    "text.color":       COLOR_TEXT,
    "xtick.color":      COLOR_TEXT,
    "ytick.color":      COLOR_TEXT,
    "font.family":      "sans-serif",
    "font.size":        11,
})


# ---------- Parsing ----------

def parse_nvidia_csv(filepath: str) -> Tuple[List[dict], List[Tuple[float, str]]]:
    """Parse nvidia-smi CSV output.

    Returns (rows, markers) where:
        rows   = list of dicts with parsed numeric fields + 'elapsed_s'
        markers = list of (elapsed_s, label) from MARKER comments
    """
    rows = []
    markers = []
    t0 = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Marker comments from GpuMonitor
            if line.startswith("# MARKER:"):
                match = re.match(r"# MARKER:\s*(.+?)(?:\s*@\s*(.*))?$", line)
                if match and t0 is not None:
                    label = match.group(1).strip()
                    # We don't have a reliable timestamp in the marker,
                    # so we use the last row's elapsed time
                    elapsed = rows[-1]["elapsed_s"] if rows else 0
                    markers.append((elapsed, label))
                continue

            # Skip nvidia-smi header line
            if "timestamp" in line.lower() and "utilization" in line.lower():
                continue
            # Skip lines with header-like content (units row)
            if line.startswith("#"):
                continue

            # Parse CSV data line
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 12:
                continue

            try:
                ts_str = parts[0].strip()
                # nvidia-smi timestamps: "2026/02/09 10:30:15.123"
                try:
                    ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
                except ValueError:
                    ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")

                if t0 is None:
                    t0 = ts

                elapsed = (ts - t0).total_seconds()

                def _num(s):
                    """Parse a numeric value, stripping units like ' %', ' W', ' MiB', ' MHz'."""
                    s = s.strip()
                    # Remove known suffixes
                    for suffix in (" %", " W", " MiB", " MHz", " C"):
                        s = s.replace(suffix, "")
                    s = s.strip()
                    if s in ("[N/A]", "N/A", "[Not Supported]", ""):
                        return None
                    return float(s)

                row = {
                    "elapsed_s":   elapsed,
                    "timestamp":   ts,
                    "gpu_util":    _num(parts[2]),
                    "mem_util":    _num(parts[3]),
                    "mem_used":    _num(parts[4]),
                    "mem_total":   _num(parts[5]),
                    "temperature": _num(parts[6]),
                    "power":       _num(parts[7]),
                    "sm_clock":    _num(parts[8]),
                    "mem_clock":   _num(parts[9]),
                    "pcie_gen":    _num(parts[10]),
                    "pcie_width":  _num(parts[11]),
                }
                rows.append(row)

            except (ValueError, IndexError):
                continue  # skip malformed lines

    return rows, markers


# ---------- Plotting ----------

DEFAULT_METRICS = ["gpu_util", "power", "temperature", "mem_used"]


def plot_gpu_usage(
    rows: List[dict],
    markers: List[Tuple[float, str]],
    metrics: List[str],
    output_path: str,
    title: str = "GPU Usage During Benchmark",
):
    """Generate a multi-panel time-series chart."""
    n_panels = len(metrics)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(12, 2.5 * n_panels + 1),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    elapsed = [r["elapsed_s"] for r in rows]

    for ax, metric in zip(axes, metrics):
        values = [r.get(metric) for r in rows]
        color = METRIC_COLORS.get(metric, "#333333")
        label = METRIC_LABELS.get(metric, metric)

        # Filter out None values
        valid = [(t, v) for t, v in zip(elapsed, values) if v is not None]
        if not valid:
            ax.text(0.5, 0.5, f"No data for {metric}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#999")
            continue

        vt, vv = zip(*valid)
        ax.plot(vt, vv, color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(vt, vv, alpha=0.08, color=color)

        ax.set_ylabel(label, fontsize=10, fontweight="medium")
        ax.yaxis.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add min/max/avg annotation
        vals = list(vv)
        avg_v = sum(vals) / len(vals)
        max_v = max(vals)
        min_v = min(vals)
        ax.axhline(avg_v, color=color, linewidth=0.8, linestyle=":", alpha=0.5)
        ax.text(
            0.99, 0.95,
            f"avg {avg_v:.0f}  max {max_v:.0f}  min {min_v:.0f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=color, alpha=0.7,
        )

    # Draw markers on all panels
    for elapsed_s, label in markers:
        is_start = label.startswith("START")
        is_error = label.startswith("ERROR")
        color = "#da3633" if is_error else "#1a7f37" if is_start else "#bf8700"
        for ax in axes:
            ax.axvline(elapsed_s, color=color, linewidth=0.8, linestyle="--", alpha=0.5)
        # Label on top panel only
        short_label = label.replace("START ", "▶ ").replace("END ", "■ ").replace("ERROR ", "✗ ")
        axes[0].text(
            elapsed_s, axes[0].get_ylim()[1] * 0.98,
            short_label, fontsize=7, rotation=90,
            va="top", ha="right", color=color, alpha=0.7,
        )

    # X-axis label on bottom panel
    axes[-1].set_xlabel("Elapsed Time (s)", fontsize=11)

    # Title
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight", facecolor=COLOR_BG, dpi=200)
    plt.close(fig)

    print(f"✅  {output_path}")
    print(f"   {len(rows)} samples, {len(markers)} markers, {n_panels} panels")


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(
        description="Plot GPU usage from nvidia-smi CSV logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csv_file", help="Path to nvidia-smi CSV log")
    p.add_argument("-o", "--output", default=None, help="Output image path (default: <csv>.png)")
    p.add_argument(
        "--metrics", "-m",
        default="gpu_util,power,temperature,mem_used",
        help="Comma-separated metrics to plot (default: gpu_util,power,temperature,mem_used). "
             "Available: gpu_util, mem_util, power, temperature, mem_used, sm_clock, mem_clock",
    )
    p.add_argument("--title", "-t", default=None, help="Chart title")
    args = p.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output or str(csv_path.with_suffix(".png"))
    metrics = [m.strip() for m in args.metrics.split(",")]
    title = args.title or f"GPU Usage — {csv_path.stem}"

    rows, markers = parse_nvidia_csv(str(csv_path))
    if not rows:
        print(f"Error: no valid data rows in {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(rows)} samples, {len(markers)} markers from {csv_path}")
    plot_gpu_usage(rows, markers, metrics, output, title=title)


if __name__ == "__main__":
    main()
