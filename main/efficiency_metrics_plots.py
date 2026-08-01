"""
efficiency_metrics_plots.py — Bar-chart comparison of computational/energy
efficiency metrics across several evaluated models.

Analogous to box_plots.py (which compares accuracy), but for the efficiency
metrics produced by efficiency_metrics.py. For every model listed in
MODEL_DIRECTORIES it reads the machine-readable ``efficiency_metrics.json`` from
that model's efficiency-results folder and draws one vertical bar chart per
metric, with the models along the x-axis.

Workflow
--------
  1. Run efficiency_metrics.py once per model so each result folder under
     efficiency_results/ contains an ``efficiency_metrics.json``.
  2. List those folders in MODEL_DIRECTORIES (relative to efficiency_results),
     with a matching human-readable label per model in XAXIS_LABELS.
  3. python efficiency_metrics_plots.py

One PNG per metric is written to
    report_figures/efficiency_plots/<NAME_OF_EXPERIMENT>/

Configure everything in the CONFIGURATION block below.
"""

from __future__ import annotations

import os
import re
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# CONFIGURATION — edit these to define the comparison
# --------------------------------------------------------------------------- #

# Name of this comparison. Figures land in
# report_figures/efficiency_plots/<NAME_OF_EXPERIMENT>/ so runs stay organised.
NAME_OF_EXPERIMENT = "Energy Matched CNN vs SNN"

# The efficiency-results folders to compare, relative to efficiency_results.
# One per model. Each MUST contain an efficiency_metrics.json — produced by
# efficiency_metrics.py (re-run it on a model if its folder only has the .txt).
# Example three-model comparison (replace the folders with your own once the
# models are trained and measured):
#     "SNN_results/Reference_model_SNN",
#     "Parameter_matched_results/Parameter_Matched_CNN",
#     "Energy_matched_results/Energy_Matched_CNN",
MODEL_DIRECTORIES = [
    "Energy_matched_results/Energy_Matched_CNN",
    "SNN_results/Reference_model_SNN",
]

# The x-axis label shown under each model's bar, in the same order as above.
# Text wrapped in "$ ... $" is rendered as LaTeX/mathtext.
XAXIS_LABELS = [
    "CNN",
    "SNN"
]

# --------------------------------------------------------------------------- #
# Figures to draw. One vertical bar chart per entry.
#   source : key in efficiency_metrics.json to plot (energy keys are in pJ)
#   kind   : "energy" (unit-convertible) or "ops" (dimensionless count)
#   unit   : for energy figures, one of "J" | "mJ" | "nJ" | "pJ"; ignored for ops
#   ylabel : y-axis label (write the unit into it yourself so it stays in sync)
#   title  : figure title; "" -> no title
#   file   : output PNG base name
# All reported values are rounded to the nearest integer.
# --------------------------------------------------------------------------- #
FIGURES = [
    {
        "source": "energy_per_encounter_pj",
        "kind":   "energy",
        "unit":   "mJ",
        "ylabel": "Energy per encounter [mJ]",
        "title":  "", # Average Total Estimated Inference Energy per Encounter
        "file":   "energy_per_encounter",
    },
    {
        "source": "energy_per_step_pj",
        "kind":   "energy",
        "unit":   "nJ",
        "ylabel": "Energy per time step [nJ]",
        "title":  "", # Average Inference Energy per Time Step
        "file":   "energy_per_step",
    },
    {
        "source": "ops_per_encounter",
        "kind":   "ops",
        "unit":   None,
        "ylabel": "Operations per encounter",
        "title":  "", # Average Total Operations per Encounter
        "file":   "ops_per_encounter",
    },
    {
        "source": "ops_per_step",
        "kind":   "ops",
        "unit":   None,
        "ylabel": "Operations per time step (MACs + ACs)",
        "title":  "", # Average Operations per Time Step (MACs + ACs)
        "file":   "ops_per_step",
    },
]

# Energy-unit conversion: how many picojoules per display unit.
UNIT_TO_PJ = {"J": 1e12, "mJ": 1e9, "nJ": 1e3, "pJ": 1.0}

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
EFFICIENCY_DIR = os.path.join(_HERE, os.pardir, "efficiency_results")
OUTPUT_DIR = os.path.join(_HERE, os.pardir, "report_figures", "efficiency_plots",
                          NAME_OF_EXPERIMENT)

# --------------------------------------------------------------------------- #
# Style (kept consistent with box_plots.py / TFM_report_plots.py)
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
    "font.family": "serif",
})

# Categorical palette (fixed order; validated colourblind-safe reference set).
# One colour per model, kept consistent across all figures so a given model
# reads the same in every chart.
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4"]
BAR_EDGE = "#33393f"
GRID_COLOR = "#b9b9b9"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_model_metrics(rel_dir: str) -> dict:
    """Load one model's efficiency_metrics.json."""
    path = os.path.join(EFFICIENCY_DIR, rel_dir, "efficiency_metrics.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"efficiency_metrics.json not found for model:\n  {rel_dir}\n"
            f"Expected at: {os.path.normpath(path)}\n"
            f"Run efficiency_metrics.py on this model first (it writes the JSON "
            f"alongside efficiency_metric_summary.txt)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_values(metrics_list: list[dict], source: str) -> list[float]:
    """Return the base-unit value of ``source`` for each model."""
    values = []
    for rel_dir, m in zip(MODEL_DIRECTORIES, metrics_list):
        if source not in m:
            raise KeyError(
                f"Key '{source}' not found in efficiency_metrics.json of model:\n"
                f"  {rel_dir}\nAvailable keys: {sorted(m.keys())}"
            )
        values.append(float(m[source]))
    return values


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _display_values(fig_cfg: dict, base_values: list[float]) -> list[int]:
    """Convert base values to the figure's display unit and round to integers."""
    if fig_cfg["kind"] == "energy":
        unit = fig_cfg["unit"]
        if unit not in UNIT_TO_PJ:
            raise ValueError(
                f"Unknown energy unit {unit!r} for figure '{fig_cfg['file']}'. "
                f"Choose one of {list(UNIT_TO_PJ)}.")
        return [round(v / UNIT_TO_PJ[unit]) for v in base_values]
    return [round(v) for v in base_values]     # ops: dimensionless count


def plot_metric(fig_cfg: dict, base_values: list[float]) -> None:
    values = _display_values(fig_cfg, base_values)
    n = len(values)
    x = np.arange(n)

    # Width scales with the number of models AND the longest x-label so the
    # (often long) model names do not collide.
    label_room = 0.09 * max((len(s) for s in XAXIS_LABELS), default=8)
    fig, ax = plt.subplots(figsize=(max(6.5, (1.6 + label_room) * n + 1.6), 5.0))
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
    bars = ax.bar(x, values, width=0.62, color=colors, edgecolor=BAR_EDGE,
                  linewidth=1.0, zorder=3)

    # Direct value labels on top of each bar (rounded integer, thousands-grouped).
    vmax = max(values) if values else 0
    for b, v in zip(bars, values):
        ax.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11.5, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(XAXIS_LABELS, fontsize=12)
    ax.set_ylabel(fig_cfg["ylabel"])
    if fig_cfg["title"]:
        ax.set_title(fig_cfg["title"])
    ax.set_ylim(0, vmax * 1.18 if vmax > 0 else 1)
    ax.margins(x=0.08)
    ax.grid(axis="y", color=GRID_COLOR, linestyle=":", linewidth=0.8, alpha=0.7,
            zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{fig_cfg['file']}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {os.path.basename(path)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(MODEL_DIRECTORIES) != len(XAXIS_LABELS):
        raise ValueError(
            f"MODEL_DIRECTORIES ({len(MODEL_DIRECTORIES)}) and XAXIS_LABELS "
            f"({len(XAXIS_LABELS)}) must have the same length.")
    if not MODEL_DIRECTORIES:
        raise ValueError("MODEL_DIRECTORIES is empty — nothing to plot.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics_list = [load_model_metrics(d) for d in MODEL_DIRECTORIES]
    for rel_dir, m in zip(MODEL_DIRECTORIES, metrics_list):
        print(f"loaded  {m.get('family', '?'):>3}  {m.get('model_version', '?'):<20}"
              f"  <-  {rel_dir}")

    for fig_cfg in FIGURES:
        base_values = collect_values(metrics_list, fig_cfg["source"])
        plot_metric(fig_cfg, base_values)

    print(f"\nAll figures written to: {os.path.normpath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
