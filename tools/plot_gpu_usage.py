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
    """Parse nvidia-smi CSV output, auto-detecting column layout from header.

    Returns (rows, markers) where:
        rows   = list of dicts with parsed numeric fields + 'elapsed_s'
        markers = list of (elapsed_s, label) from MARKER comments
    """
    rows = []
    markers = []
    raw_markers = []
    t0 = None
    col_map = None  # Built from header line

    # Header keyword -> field name mapping
    HEADER_PATTERNS = {
        "utilization.gpu": "gpu_util",
        "utilization.memory": "mem_util",
        "memory.used": "mem_used",
        "memory.total": "mem_total",
        "temperature.gpu": "temperature",
        "power.draw": "power",
        "clocks.current.graphics": "sm_clock",
        "clocks.current.sm": "sm_clock",
        "clocks.current.memory": "mem_clock",
        "pcie.link.gen": "pcie_gen",
        "pcie.link.width": "pcie_width",
    }

    def _num(s):
        """Parse a numeric value, stripping units."""
        s = s.strip()
        for suffix in (" %", " W", " MiB", " MHz", " C"):
            if s.endswith(suffix):
                s = s[:-len(suffix)]
        s = s.strip()
        if s in ("[N/A]", "N/A", "[Not Supported]", ""):
            return None
        return float(s)

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Marker comments
            if line.startswith("# MARKER:"):
                match = re.match(r"# MARKER:\s*(.+?)\s*@\s*(\d{4}/\d{2}/\d{2}\s+\S+)", line)
                if match:
                    label = match.group(1).strip()
                    ts_str = match.group(2).strip()
                    try:
                        try:
                            marker_ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
                        except ValueError:
                            marker_ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
                        raw_markers.append((marker_ts, label))
                    except ValueError:
                        pass
                continue

            if line.startswith("#"):
                continue

            # Detect header line and build column map
            if "timestamp" in line.lower() and ("utilization" in line.lower() or "memory" in line.lower()):
                headers = [h.strip().lower() for h in line.split(",")]
                col_map = {"timestamp": 0}
                for i, h in enumerate(headers):
                    for pattern, field in HEADER_PATTERNS.items():
                        if pattern in h:
                            col_map[field] = i
                            break
                continue

            # Parse data line
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue

            try:
                ts_idx = col_map.get("timestamp", 0) if col_map else 0
                ts_str = parts[ts_idx].strip()
                try:
                    ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
                except ValueError:
                    ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")

                if t0 is None:
                    t0 = ts

                elapsed = (ts - t0).total_seconds()

                def _get(field):
                    if col_map is None:
                        return None
                    idx = col_map.get(field)
                    if idx is None or idx >= len(parts):
                        return None
                    return _num(parts[idx])

                row = {
                    "elapsed_s":   elapsed,
                    "timestamp":   ts,
                    "gpu_util":    _get("gpu_util"),
                    "mem_util":    _get("mem_util"),
                    "mem_used":    _get("mem_used"),
                    "mem_total":   _get("mem_total"),
                    "power":       _get("power"),
                    "sm_clock":    _get("sm_clock"),
                    "mem_clock":   _get("mem_clock"),
                    "temperature": _get("temperature"),
                    "pcie_gen":    _get("pcie_gen"),
                    "pcie_width":  _get("pcie_width"),
                }
                rows.append(row)

            except (ValueError, IndexError):
                continue

    # Convert raw marker timestamps to elapsed seconds
    if t0 is not None:
        for marker_ts, label in raw_markers:
            elapsed = (marker_ts - t0).total_seconds()
            markers.append((elapsed, label))

    return rows, markers


# ---------- Plotting ----------

DEFAULT_METRICS = ["gpu_util", "power", "temperature", "mem_used"]


def plot_gpu_usage(
    rows: List[dict],
    markers: List[Tuple[float, str]],
    metrics: List[str],
    output_path: str,
    title: str = "GPU Usage During Benchmark",
    power_limit: float = 250.0,
    power_limit2: float = 600.0,
    mem_total_mib: float = 97887.0,
):
    """Generate a multi-panel time-series chart."""
    # Dynamic Y-axis: will be computed per-metric from actual data (max + 10%)

    n_panels = len(metrics)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(13, 2.8 * n_panels + 1.2),
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

        # Dynamic Y-axis: 0 to max_value * 1.10 (10% headroom)
        vals = list(vv)
        max_v = max(vals)
        min_v = min(vals)
        avg_v = sum(vals) / len(vals)

        y_lo = 0
        y_hi = max_v * 1.10 if max_v > 0 else 1

        # For power: ensure threshold lines are visible if close to data range
        if metric == "power":
            if power_limit and power_limit <= y_hi * 1.2:
                y_hi = max(y_hi, power_limit * 1.08)
            if power_limit2 and power_limit2 <= y_hi * 1.2:
                y_hi = max(y_hi, power_limit2 * 1.08)

        ax.set_ylim(y_lo, y_hi)

        ax.plot(vt, vv, color=color, linewidth=1.4, alpha=0.9)
        ax.fill_between(vt, [y_lo] * len(vt), vv, alpha=0.06, color=color)

        ax.set_ylabel(label, fontsize=10, fontweight="medium")
        ax.yaxis.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Avg line + stats badge (compact, non-overlapping)
        ax.axhline(avg_v, color=color, linewidth=0.8, linestyle=":", alpha=0.4)
        stats_text = f"avg {avg_v:.0f} │ max {max_v:.0f} │ min {min_v:.0f}"
        ax.text(
            0.99, 0.92,
            stats_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=color, alpha=0.75,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.3),
        )

        # Power threshold lines
        if metric == "power":
            if power_limit and power_limit <= y_hi:
                ax.axhline(power_limit, color="#bf8700", linewidth=1.2, linestyle="--", alpha=0.6)
                ax.text(
                    0.01, power_limit, f" TDP {power_limit:.0f}W",
                    transform=ax.get_yaxis_transform(),
                    va="bottom", ha="left",
                    fontsize=8, color="#bf8700", alpha=0.7, fontweight="bold",
                )
            if power_limit2 and power_limit2 <= y_hi:
                ax.axhline(power_limit2, color="#da3633", linewidth=1.2, linestyle="-", alpha=0.6)
                ax.text(
                    0.01, power_limit2, f" MAX {power_limit2:.0f}W",
                    transform=ax.get_yaxis_transform(),
                    va="bottom", ha="left",
                    fontsize=8, color="#da3633", alpha=0.7, fontweight="bold",
                )

    # Draw markers on all panels — deduplicate overlapping labels
    marker_positions = []
    for elapsed_s, label in markers:
        is_start = label.startswith("START")
        is_error = label.startswith("ERROR")
        mcolor = "#da3633" if is_error else "#1a7f37" if is_start else "#bf8700"
        for ax in axes:
            ax.axvline(elapsed_s, color=mcolor, linewidth=0.7, linestyle="--", alpha=0.4)
        marker_positions.append((elapsed_s, label, mcolor))

    # Label markers on top panel, staggering overlaps
    if marker_positions:
        top_ax = axes[0]
        y_top = top_ax.get_ylim()[1]
        # Sort by x position
        marker_positions.sort(key=lambda m: m[0])
        last_x = -999
        stagger = 0
        for elapsed_s, label, mcolor in marker_positions:
            short_label = label.replace("START ", "▶ ").replace("END ", "■ ").replace("ERROR ", "✗ ")
            # Stagger vertically if markers are close together
            if abs(elapsed_s - last_x) < (elapsed[-1] - elapsed[0]) * 0.03:
                stagger = (stagger + 1) % 3
            else:
                stagger = 0
            y_offset = y_top * (0.98 - stagger * 0.15)
            top_ax.text(
                elapsed_s, y_offset,
                short_label, fontsize=6.5, rotation=45,
                va="top", ha="left", color=mcolor, alpha=0.75,
            )
            last_x = elapsed_s

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
    p.add_argument(
        "--power-limit", "-p",
        type=float, default=250.0,
        help="TDP threshold line in watts (default: 250W). Set 0 to disable.",
    )
    p.add_argument(
        "--power-limit2",
        type=float, default=600.0,
        help="Max power threshold line in watts (default: 600W). Set 0 to disable.",
    )
    p.add_argument(
        "--mem-total",
        type=float, default=97887.0,
        help="Total GPU memory in MiB for Y-axis scale (default: 97887 = ~96GB).",
    )
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
        print(f"DEBUG ERROR: no valid data rows", file=sys.stderr)
        print(f"Error: no valid data rows in {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(rows)} samples, {len(markers)} markers from {csv_path}")
    plot_gpu_usage(rows, markers, metrics, output, title=title,
                   power_limit=args.power_limit or None,
                   power_limit2=args.power_limit2 or None,
                   mem_total_mib=args.mem_total)


if __name__ == "__main__":
    main()
