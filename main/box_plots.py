"""
box_plots.py — Per-metric box plots comparing several model experiments.

For every evaluation metric this script draws a single figure whose y-axis lists
the experiments being compared and whose x-axis shows the box-and-whisker
distribution of that metric across all test cases. The figure title is the
metric name.

Each experiment is identified by the results folder produced at test time, e.g.

    SNN_results/RateOnly_Original_Stateful_Nz16_epoch100_Warmup

which is expected to contain an ``accuracy_metrics.csv`` file (one row per test
case plus a trailing AVERAGE row, which is ignored here). Configure the
comparison in the CONFIGURATION block below:

  * EXPERIMENT_DIRECTORIES : the result folders, one per experiment, relative to
                             the results/ folder.
  * YAXIS_LABELS           : the label shown on the y-axis for each experiment,
                             in the same order. Text wrapped in ``$ ... $`` is
                             rendered as LaTeX/mathtext (e.g. "$N_z=16$").
  * METRICS                : the metrics to plot, as {csv_column: figure_title}.

One PNG per metric is written to OUTPUT_DIR.

Usage:
    python box_plots.py
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# CONFIGURATION — edit these to define the comparison
# --------------------------------------------------------------------------- #

# Name of this comparison. The figures are written to
# report_figures/box_plots/<NAME_OF_EXPERIMENT>/ so results stay organised.
NAME_OF_EXPERIMENT = "Energy Matched CNN vs SNN"

# The result folders to compare, relative to "results" folder. One per experiment.
EXPERIMENT_DIRECTORIES = [
    "Parameter_matched_results/Parameter_Matched_CNN_epoch100_Warmup",
    "SNN_results/RateOnly_Original_Stateful_Nz16_epoch100_Warmup",
    ]

# The y-axis label for each experiment, in the same order as above. Anything
# wrapped in "$ ... $" is rendered as LaTeX/mathtext.
YAXIS_LABELS = ["CNN", "SNN"]

# Gust-sign analysis.
#   GUST_SIGN_ANALYSIS = False -> the experiment comparison described above.
#   GUST_SIGN_ANALYSIS = True  -> ignore EXPERIMENT_DIRECTORIES / YAXIS_LABELS and
#                                instead draw, for the single result folder named
#                                in GUST_SIGN_RESULTS_DIR, a 2x2 grid of the four
#                                metrics. Every subplot is a horizontal box plot
#                                that separates the test cases by the sign of the
#                                gust ratio (G >= 0 on top, G < 0 on the bottom),
#                                read from the "Gust" column of
#                                accuracy_metrics.csv. The subplot titles are the
#                                metric names and the shared y-axis label is
#                                "Gust Sign". Output is written automatically to
#                                report_figures/box_plots/Gust Sign/<folder name>/.
GUST_SIGN_ANALYSIS = False
GUST_SIGN_RESULTS_DIR = (
    "SNN_results/RateOnly_Original_Stateful_Nz16_epoch100_Warmup"
)

# Metrics to plot: {CSV column name: figure title}. Comment out any you do not
# want. Every column here must exist in accuracy_metrics.csv.
#
# The default is the four complementary metrics reported in the thesis
# (Evaluation Metrics section): MAE, RMSE, Relative L2 Error and Cosine
# Similarity. The remaining recognised columns are left commented out below in
# case a broader comparison is needed.
METRICS = {
    "MAE_Cl":                          "MAE",
    "RMSE_Cl":                         "RMSE",
    "RelL2Err_Cl":                     "Relative $L_2$ Error",
    "CosineSim_Cl":                    "Cosine Similarity",
    # "MSE_Cl":                          "MSE",
    # "Peak_Cl_Error":                   "Peak $C_l$ Error",
    # "Peak_Timing_Error":               "Peak Timing Error",
    # "Integrated_Abs_Error":            "Integrated Absolute Error",
    # "Normalized_Integrated_Abs_Error": "Normalized Integrated Absolute Error",
}

# Optional override of the metric-axis label (keyed by CSV column). The figure
# title is unchanged; only the axis label is replaced. Metrics not listed here
# use their title as the axis label. Used to keep symbols consistent with the
# report (e.g. the Relative L2 Error axis is labelled with its symbol).
METRIC_AXIS_LABELS = {
    "RelL2Err_Cl": r"$\varepsilon_{L_2}$",
}

# Row that summarises the file and must be excluded from the distributions.
SUMMARY_ROW_KEY = "AVERAGE"

# Where to write the figures. Each comparison lands in its own subfolder named
# after NAME_OF_EXPERIMENT.
RESULTS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results")
if GUST_SIGN_ANALYSIS:
    # Self-contained output folder derived from the result folder name so no
    # extra configuration is needed.
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir,
                              "report_figures", "box_plots", "Gust Sign",
                              os.path.basename(GUST_SIGN_RESULTS_DIR))
else:
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir,
                              "report_figures", "box_plots", NAME_OF_EXPERIMENT)


# --------------------------------------------------------------------------- #
# Style (kept consistent with TFM_report_plots.py)
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

BOX_FACE = "#cfe0f0"     # light blue box fill
BOX_EDGE = "#1f5fa6"     # blue box / whisker / median edge
POINT_COLOR = "#c0392b"  # red individual-case markers
MEAN_COLOR = "#2e8b57"   # green mean marker

# Gust-sign analysis: individual test-case points are coloured by gust sign
# (matching test.py's breakdown_by_gust_signed plots).
POS_COLOR = "#1f5fa6"    # blue — positive gust (G >= 0, incl. baseline)
NEG_COLOR = "#c0392b"    # red  — negative gust (G < 0)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _slugify(title: str) -> str:
    """Turn a figure title into a safe file name (strip LaTeX/mathtext)."""
    text = title.replace("$", "")
    text = re.sub(r"[\\{}^_]", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text.lower()


def _metrics_csv_path(rel_dir: str) -> str:
    """Path to the accuracy_metrics.csv for one experiment
    (<experiment>/accuracy_metrics.csv)."""
    return os.path.join(RESULTS_DIR, rel_dir, "accuracy_metrics.csv")


def load_experiment_metrics(rel_dir: str) -> pd.DataFrame:
    """Load the per-case metrics for one experiment (AVERAGE row dropped)."""
    csv_path = _metrics_csv_path(rel_dir)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"accuracy_metrics.csv not found for experiment:\n  {rel_dir}\n"
            f"Expected at: {os.path.normpath(csv_path)}"
        )
    df = pd.read_csv(csv_path)
    # Drop the trailing summary row (its Case column equals SUMMARY_ROW_KEY).
    df = df[df["Case"].astype(str) != SUMMARY_ROW_KEY].reset_index(drop=True)
    return df


def collect_values(experiments: list[pd.DataFrame], column: str) -> list[np.ndarray]:
    """Return the finite per-case values of ``column`` for each experiment."""
    series = []
    for rel_dir, df in zip(EXPERIMENT_DIRECTORIES, experiments):
        if column not in df.columns:
            raise KeyError(
                f"Metric column '{column}' not found in accuracy_metrics.csv "
                f"of experiment:\n  {rel_dir}\n"
                f"Available columns: {list(df.columns)}"
            )
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        series.append(values[np.isfinite(values)])
    return series


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _legend_handles() -> list:
    """The three legend entries shared by every box plot (median, mean, case)."""
    return [
        plt.Line2D([], [], color=BOX_EDGE, linewidth=1.8, label="Median"),
        plt.Line2D([], [], marker="D", linestyle="none",
                   markerfacecolor=MEAN_COLOR, markeredgecolor=MEAN_COLOR,
                   markersize=6, label="Mean"),
        plt.Line2D([], [], marker="o", linestyle="none",
                   markerfacecolor=POINT_COLOR, markeredgecolor="none",
                   markersize=6, alpha=0.55, label="Test case"),
    ]


def _add_marker_legend(ax, y_offset=-0.18) -> None:
    """Legend (median, mean, test case) placed below the axes in a single row.

    ``y_offset`` is the axes-fraction position of the legend below the x-axis.
    Shorter figures (a single experiment) need a larger drop so the legend does
    not overlap the x-axis label.
    """
    handles = _legend_handles()
    # Outside, centred below the plot, all entries side by side so no test-case
    # point is ever covered by the legend.
    ax.legend(handles=handles, frameon=False, ncol=len(handles),
              loc="upper center", bbox_to_anchor=(0.5, y_offset),
              columnspacing=2.5, handletextpad=0.6)


def _save_metric_fig(fig, title: str) -> None:
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"boxplot_{_slugify(title)}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {os.path.basename(path)}")


def _draw_metric_boxplot(ax, column: str, title: str,
                         data: list[np.ndarray]) -> None:
    """Draw one metric's horizontal box plot onto ``ax`` (no legend, no save).

    Experiments run along the y-axis and the metric values along the x-axis.
    With a single experiment this is a classic one-box plot with no y label.
    """
    n = len(data)
    axis_label = METRIC_AXIS_LABELS.get(column, title)

    positions = np.arange(1, n + 1)
    ax.boxplot(
        data,
        positions=positions,
        vert=False,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor=MEAN_COLOR,
                       markeredgecolor=MEAN_COLOR, markersize=6),
        medianprops=dict(color=BOX_EDGE, linewidth=1.8),
        boxprops=dict(facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=1.4),
        whiskerprops=dict(color=BOX_EDGE, linewidth=1.4),
        capprops=dict(color=BOX_EDGE, linewidth=1.4),
        flierprops=dict(marker="", markersize=0),  # outliers shown via scatter
    )

    # Overlay the individual test cases as jittered points for transparency.
    rng = np.random.default_rng(0)
    for pos, values in zip(positions, data):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(values, pos + jitter, s=16, color=POINT_COLOR,
                   alpha=0.45, zorder=3, edgecolors="none")

    if n == 1:
        # Single experiment: no category to label on the y-axis.
        ax.set_yticks([])
        ax.margins(y=0.25)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(YAXIS_LABELS)
        ax.margins(y=0.08)
        # Show the first experiment at the top, matching the config order.
        ax.invert_yaxis()

    ax.set_xlabel(axis_label)
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5)


def plot_metric_boxplot(column: str, title: str, data: list[np.ndarray]) -> None:
    """Draw and save a standalone horizontal box plot for one metric.

    Used when comparing several experiments (one figure per metric). The wide
    aspect ratio spreads out the box content; height grows with the number of
    experiments so each box keeps enough vertical room for its jittered points.
    """
    n = len(data)
    fig, ax = plt.subplots(figsize=(8.0, max(2.8, 1.15 * n + 1.6)))
    _draw_metric_boxplot(ax, column, title, data)
    # The single-experiment figure is short, so drop the legend further to clear
    # the x-axis label; taller multi-experiment figures need only a small drop.
    _add_marker_legend(ax, y_offset=-0.5 if n == 1 else -0.18)
    _save_metric_fig(fig, title)


def plot_metrics_grid(experiments: list[pd.DataFrame]) -> None:
    """Single-experiment view: all metrics in one compact grid (2 columns) with
    a single shared legend at the bottom. Saved as one figure."""
    items = list(METRICS.items())
    ncols = 2
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(11.0, 2.7 * nrows + 0.4),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, (column, title) in zip(axes_flat, items):
        _draw_metric_boxplot(ax, column, title, collect_values(experiments, column))

    # Hide any unused cells (e.g. an odd number of metrics).
    for ax in axes_flat[len(items):]:
        ax.set_visible(False)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    # One shared legend, centred at the bottom of the whole figure.
    fig.legend(handles=_legend_handles(), frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               columnspacing=2.5, handletextpad=0.6)

    path = os.path.join(OUTPUT_DIR, "boxplots_combined.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {os.path.basename(path)}")


# --------------------------------------------------------------------------- #
# Gust-sign analysis
# --------------------------------------------------------------------------- #
def _gust_signed_split(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Split one metric's per-case values by the sign of the gust ratio.

    Returns ``(neg_vals, pos_vals)`` where ``pos_vals`` holds the cases with
    G >= 0 (baseline included) and ``neg_vals`` the cases with G < 0. Only finite
    metric/gust pairs are kept.
    """
    if "Gust" not in df.columns:
        raise KeyError(
            "Column 'Gust' not found in accuracy_metrics.csv of experiment:\n"
            f"  {GUST_SIGN_RESULTS_DIR}\nAvailable columns: {list(df.columns)}"
        )
    if column not in df.columns:
        raise KeyError(
            f"Metric column '{column}' not found in accuracy_metrics.csv of "
            f"experiment:\n  {GUST_SIGN_RESULTS_DIR}\n"
            f"Available columns: {list(df.columns)}"
        )
    gust = pd.to_numeric(df["Gust"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values) & np.isfinite(gust)
    values, gust = values[finite], gust[finite]
    return values[gust < 0], values[gust >= 0]


def _gust_signed_legend_handles() -> list:
    """Legend entries for the gust-sign grid (median, mean, and the two signs)."""
    return [
        plt.Line2D([], [], color=BOX_EDGE, linewidth=1.8, label="Median"),
        plt.Line2D([], [], marker="D", linestyle="none",
                   markerfacecolor=MEAN_COLOR, markeredgecolor=MEAN_COLOR,
                   markersize=6, label="Mean"),
        plt.Line2D([], [], marker="o", linestyle="none",
                   markerfacecolor=POS_COLOR, markeredgecolor="none",
                   markersize=6, alpha=0.6, label=r"Positive gust ($G\geq0$)"),
        plt.Line2D([], [], marker="o", linestyle="none",
                   markerfacecolor=NEG_COLOR, markeredgecolor="none",
                   markersize=6, alpha=0.6, label=r"Negative gust ($G<0$)"),
    ]


def _draw_gust_signed_boxplot(ax, column: str, title: str,
                              neg_vals: np.ndarray, pos_vals: np.ndarray) -> None:
    """Draw one metric's gust-sign box plot onto ``ax`` (no legend, no save).

    Two horizontal boxes: G < 0 at the bottom and G >= 0 at the top. The y-axis
    is labelled "Gust Sign" with the two signs as tick labels; the title is the
    metric name.
    """
    axis_label = METRIC_AXIS_LABELS.get(column, title)
    data = [neg_vals, pos_vals]  # position 1 = negative (bottom), 2 = positive (top)

    ax.boxplot(
        data,
        positions=[1, 2],
        vert=False,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor=MEAN_COLOR,
                       markeredgecolor=MEAN_COLOR, markersize=6),
        medianprops=dict(color=BOX_EDGE, linewidth=1.8),
        boxprops=dict(facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=1.4),
        whiskerprops=dict(color=BOX_EDGE, linewidth=1.4),
        capprops=dict(color=BOX_EDGE, linewidth=1.4),
        flierprops=dict(marker="", markersize=0),  # outliers shown via scatter
    )

    # Overlay the individual test cases as jittered points, coloured by sign.
    rng = np.random.default_rng(0)
    for pos_i, values, col in zip([1, 2], data, [NEG_COLOR, POS_COLOR]):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(values, pos_i + jitter, s=16, color=col, alpha=0.5,
                   zorder=3, edgecolors="none")

    ax.set_yticks([1, 2])
    ax.set_yticklabels([r"$G< 0$", r"$G\geq 0$"])
    ax.margins(y=0.25)
    ax.set_xlabel(axis_label)
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5)


def plot_gust_signed_grid(df: pd.DataFrame) -> None:
    """All metrics in one 2-column grid, each split by gust sign, shared legend."""
    items = list(METRICS.items())
    ncols = 2
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(11.0, 2.7 * nrows + 0.4),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, (column, title) in zip(axes_flat, items):
        neg_vals, pos_vals = _gust_signed_split(df, column)
        _draw_gust_signed_boxplot(ax, column, title, neg_vals, pos_vals)

    # Hide any unused cells (e.g. an odd number of metrics).
    for ax in axes_flat[len(items):]:
        ax.set_visible(False)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    # One shared legend, centred at the bottom of the whole figure.
    fig.legend(handles=_gust_signed_legend_handles(), frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               columnspacing=2.5, handletextpad=0.6)

    path = os.path.join(OUTPUT_DIR, "boxplots_gust_signed_combined.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {os.path.basename(path)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if GUST_SIGN_ANALYSIS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = load_experiment_metrics(GUST_SIGN_RESULTS_DIR)
        print(f"loaded {len(df)} cases  <-  {GUST_SIGN_RESULTS_DIR}")
        plot_gust_signed_grid(df)
        print(f"\nFigure written to: {os.path.normpath(OUTPUT_DIR)}")
        return

    if len(EXPERIMENT_DIRECTORIES) != len(YAXIS_LABELS):
        raise ValueError(
            f"EXPERIMENT_DIRECTORIES ({len(EXPERIMENT_DIRECTORIES)}) and "
            f"YAXIS_LABELS ({len(YAXIS_LABELS)}) must have the same length."
        )
    if not EXPERIMENT_DIRECTORIES:
        raise ValueError("EXPERIMENT_DIRECTORIES is empty — nothing to plot.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    experiments = [load_experiment_metrics(d) for d in EXPERIMENT_DIRECTORIES]
    for rel_dir, df in zip(EXPERIMENT_DIRECTORIES, experiments):
        print(f"loaded {len(df)} cases  <-  {rel_dir}")

    # Combined 2x2 grid of all metrics — produced every run. Uses YAXIS_LABELS on
    # the y-axis and the directories in EXPERIMENT_DIRECTORIES.
    plot_metrics_grid(experiments)

    # With several experiments, also save one standalone figure per metric.
    if len(EXPERIMENT_DIRECTORIES) > 1:
        for column, title in METRICS.items():
            data = collect_values(experiments, column)
            plot_metric_boxplot(column, title, data)

    print(f"\nAll figures written to: {os.path.normpath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
