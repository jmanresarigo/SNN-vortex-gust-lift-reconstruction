"""
test.py — Evaluate a trained checkpoint on the held-out test set.

Loads one checkpoint folder (config_snapshot.txt + a .pth weights file) selected
via `model_folder_name` / `MODEL_WEIGHTS_FILE` below, rebuilds the exact
train-time configuration from the snapshot, reconstructs the model (the spiking
SNN or a parameter-/energy-matched CNN baseline, auto-detected from the
snapshot), runs inference on the test cases, and reports Cl-reconstruction
accuracy metrics plus diagnostic figures.

Output — each run creates a timestamped folder grouped by model type
    results/{SNN_results | Parameter_matched_results | Energy_matched_results}/<timestamp>/
containing:
    config_snapshot.txt   the reconstructed test configuration + test-case list
    dataset_split.txt     the train / val / test case listing
    accuracy_metrics.csv  per-case Cl metrics (+ an AVERAGE summary row)
    summary/, latent/, samples/   Cl-evolution plots, latent visualisations,
                          and (decoder models only) per-case animations

Usage:
    python test.py
"""

import csv
import gc
import os
import re
import sys
import warnings
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Consistent, readable styling across every figure. mathtext.fontset='cm'
# renders LaTeX symbols (e.g. $C_l$) in the classic Computer Modern equation
# style so subscripts look like real equations; the enlarged font sizes keep
# labels/legends legible when the figures are embedded in the report.
plt.rcParams.update({
    "font.size":        15,
    "axes.titlesize":   19,
    "axes.labelsize":   17,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  14,
    "figure.titlesize": 22,
    "mathtext.fontset": "cm",
})
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
try:
    from matplotlib.animation import FFMpegWriter
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.ndimage import uniform_filter
from scipy.interpolate import CubicSpline

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'models'))

from pre_encoder import (
    load_and_encode,
    get_device,
    _case_tag,
    write_dataset_split_file,
    _parse_forced_cases,
    load_rd_params,
    list_interpolated_cases,
    _pre_compute_case_split_map,
)





# ==========================================
# TEST TARGET CHECKPOINT
# ==========================================
# Path to one specific checkpoint subfolder produced by a train_*.py script
# (contains config_snapshot.txt, best.pth, final.pth), relative to training/.
# Prefix with the training-type folder it was written into:
#   "SNN_training/<run>"                 — train_SNN.py
#   "Parameter_matched_training/<run>"   — train_parameter_matched.py
#   "Energy_matched_training/<run>"      — train_energy_matched.py
model_folder_name = "Energy_matched_training/CNN_EnergyMatched_Nz16"
MODEL_WEIGHTS = os.path.join(_HERE, '..', 'training', model_folder_name)

# Name of the .pth weights file to load from MODEL_WEIGHTS. When None, defaults
# to 'best.pth' (with a fallback to any '*best*.pth' in the folder). Set this
# explicitly when there is no best.pth — e.g., when the run was trained with
# VAL_RATIO=0, only periodic checkpoints (epoch_5.pth, epoch_10.pth, ...,
# final.pth) are produced and the user must pick one by trial-and-error.
# Example: MODEL_WEIGHTS_FILE = "epoch_50.pth"
MODEL_WEIGHTS_FILE = "epoch_100.pth"

# Number of test samples to generate per-case animations for.
# Set to None to process ALL test cases (can be slow).
MAX_ANIMATION_SAMPLES = 10

# When DATASET_TYPE='interpolated' in the snapshot, also report Cl metrics on
# the predictions downsampled to the original 1190-frame grid. The full-grid
# metrics are always reported by default.
REPORT_AT_ORIGINAL_GRID = True

# ---- Warmup-pass inference ----
# When USE_WARMUP_PASS is True, each RD case is fed to the model TWICE in a row,
# joined by a smooth cubic-spline transition. The first pass charges up the LIF
# membrane potentials (which would otherwise be initialised at zero); the second
# pass is the one used for KPI computation. The interpolated bridge avoids the
# discontinuity that would arise from naively concatenating end-of-case to
# start-of-case. Only the prediction on the second pass is scored.
#
# NOTE: the bridge pass was found to hurt strong-gust cases — the end->start
# spline leaves the membranes in an out-of-distribution state the (cold-trained)
# model handles worse than a plain cold start. Prefer the burn-in approach below
# (USE_WARMUP_PASS = False + WARMUP_EVAL_FRAMES > 0).
USE_WARMUP_PASS = False
# Number of frames in the cubic-spline bridge between the two passes.
WARMUP_INTERP_POINTS = 500

# ---- Burn-in warmup (single pass) ----
# When USE_WARMUP_PASS is False, run the normal single cold-start pass and treat
# the first WARMUP_EVAL_FRAMES timesteps as a burn-in: they are fed to the model
# (so the stateful LIF membranes charge up on the real early flow history) but
# EXCLUDED from every evaluation metric, which are computed on frame
# WARMUP_EVAL_FRAMES onward. This removes the cold-start transient from the score
# without any out-of-distribution bridge. In the Cl evolution plots the burn-in
# segment is drawn in a distinct colour. Set to 0 to score the full trajectory.
# (Overrides the trained cfg.WARMUP_STEPS at test time.)
WARMUP_EVAL_FRAMES = 30

# ---- Early-window (warmup effect) metrics ----
# The global encounter-level metrics average over the full ~6k-frame trajectory,
# so the cold-start region that the warmup pass actually improves is diluted to
# near-nothing. To make the warmup effect measurable, an additional analysis is
# produced over ONLY the first N_WARMUP_METRIC_FRAMES timesteps of each scored
# trajectory, written to results/<run>/warmup_<N>/ (alongside summary/, latent/,
# samples/). Set to None or 0 to disable.
N_WARMUP_METRIC_FRAMES = 300

# ---- Non-spiking CNN baselines ----
# The parameter-matched / energy-matched CNNs (models/parameter_matched.py,
# energy_matched.py) reconstruct Cl from the RAW continuous vorticity field
# frame-by-frame (no spike encoding, no recurrence). They are evaluated with the
# SAME accuracy metrics and the same WARMUP_EVAL_FRAMES burn-in as the SNN, for a
# fair comparison. CNN_FRAME_BATCH controls how many frames are forwarded at once
# during inference (throughput only; does not affect the metrics).
CNN_MODEL_TYPES = ("cnn_param_matched", "cnn_energy_matched")
CNN_FRAME_BATCH = 256


def _is_cnn(cfg):
    return getattr(cfg, 'MODEL_TYPE', '') in CNN_MODEL_TYPES



# ==========================================
# GRID DOMAIN  (Fukami & Taira 2023)
# ==========================================
# Physical domain of the omg_box observation window.
# Source: "vorticity fields over (x,y)/c ∈ [−1.4, 4] × [−1.2, 1.2]"
# Chord c = 1. Leading edge fixed at (0, 0).
GRID_X_MIN, GRID_X_MAX = -1.4, 4.0
GRID_Y_MIN, GRID_Y_MAX = -1.2, 1.2
GRID_EXTENT = [GRID_X_MIN, GRID_X_MAX, GRID_Y_MIN, GRID_Y_MAX]


# ==========================================
# AIRFOIL HELPERS
# ==========================================
def _naca0012_profile(n_points: int = 300):
    """
    NACA 0012 closed profile in body frame (LE at origin, chord = 1 along x).
    Returns (x, y) arrays for the full closed contour (upper then lower surface).
    """
    x = np.linspace(0, 1, n_points)
    t = 0.12
    y_t = 5 * t * (0.2969 * np.sqrt(x)
                   - 0.1260 * x
                   - 0.3516 * x**2
                   + 0.2843 * x**3
                   - 0.1015 * x**4)
    x_profile = np.concatenate([x, x[::-1]])
    y_profile = np.concatenate([y_t, -y_t[::-1]])
    return x_profile, y_profile


def _add_airfoil(ax, aoa_deg: float):
    """
    Overlay a filled black NACA 0012 airfoil on ax at the given angle of
    attack (degrees). Assumes physical coordinates with LE at (0, 0) and
    freestream along +x. Positive AoA = nose up = clockwise rotation of
    the chord (TE moves to negative y).
    """
    aoa = np.radians(aoa_deg)
    x_b, y_b = _naca0012_profile()
    # Rotate body frame → physical frame: clockwise by aoa
    x_phys =  x_b * np.cos(aoa) + y_b * np.sin(aoa)
    y_phys = -x_b * np.sin(aoa) + y_b * np.cos(aoa)
    ax.fill(x_phys, y_phys, color='black', zorder=5, linewidth=0)



# ==========================================
# CHECKPOINT CONFIG PARSER
# ==========================================
def _parse_snapshot_file(snapshot_path):
    """Parse key=value pairs from a checkpoint config snapshot."""
    values = {}
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _build_cfg_from_checkpoint(model_weights):
    """Build test config namespace from a checkpoint folder."""
    ckpt_dir = os.path.abspath(model_weights)
    if not os.path.isdir(ckpt_dir):
        training_root = os.path.abspath(os.path.join(_HERE, '..', 'training'))
        available = []
        if os.path.isdir(training_root):
            for training_type in sorted(os.listdir(training_root)):
                type_dir = os.path.join(training_root, training_type)
                if not os.path.isdir(type_dir):
                    continue
                for run in sorted(os.listdir(type_dir)):
                    if os.path.isdir(os.path.join(type_dir, run)):
                        available.append(f"{training_type}/{run}")
        available_txt = "\n".join(f"  - {d}" for d in available) if available else "  (none)"
        raise FileNotFoundError(
            "MODEL_WEIGHTS must point to an existing checkpoint folder.\n"
            f"Not found: '{ckpt_dir}'\n"
            f"Available checkpoint folders under '{training_root}':\n{available_txt}"
        )

    snapshot_path = os.path.join(ckpt_dir, 'config_snapshot.txt')
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(
            f"config_snapshot.txt not found inside checkpoint folder: '{ckpt_dir}'"
        )

    raw = _parse_snapshot_file(snapshot_path)
    required = [
        'ENCODING', 'TIME_STEPS', 'ANGLES',
        'VAL_RATIO', 'TEST_RATIO', 'N_Z', 'BETA_INIT',
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(
            f"Snapshot '{snapshot_path}' is missing keys: {missing}"
        )

    _available_pth = sorted(
        fn for fn in os.listdir(ckpt_dir) if fn.endswith('.pth')
    )
    if MODEL_WEIGHTS_FILE:
        # Explicit user-supplied weights file — honour exactly, no auto-fallback.
        best_model_path = os.path.join(ckpt_dir, MODEL_WEIGHTS_FILE)
        if not os.path.exists(best_model_path):
            _listing = "\n".join(f"  - {f}" for f in _available_pth) or "  (none)"
            raise FileNotFoundError(
                f"MODEL_WEIGHTS_FILE='{MODEL_WEIGHTS_FILE}' not found in "
                f"'{ckpt_dir}'.\nAvailable .pth files:\n{_listing}"
            )
    else:
        best_model_path = os.path.join(ckpt_dir, 'best.pth')
        if not os.path.exists(best_model_path):
            best_candidates = [
                os.path.join(ckpt_dir, fn)
                for fn in _available_pth
                if 'best' in fn
            ]
            if not best_candidates:
                _listing = "\n".join(f"  - {f}" for f in _available_pth) or "  (none)"
                raise FileNotFoundError(
                    f"No best.pth file found in '{ckpt_dir}'.\n"
                    f"This run probably trained with VAL_RATIO=0 — pick one of "
                    f"the periodic checkpoints below and set MODEL_WEIGHTS_FILE "
                    f"at the top of test.py.\nAvailable .pth files:\n{_listing}"
                )
            best_model_path = sorted(best_candidates)[0]

    # Parse list helper
    def _parse_int_list(s):
        return [int(a.strip()) for a in s.strip('[] ').split(',')]

    def _parse_str_list(s):
        """Parse a literal Python list of strings from a snapshot line."""
        import ast
        try:
            value = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
        return [str(x) for x in value] if isinstance(value, (list, tuple)) else []

    angles = _parse_int_list(raw['ANGLES'])

    # Parse split mode and angle assignments
    split_mode = raw.get('SPLIT_MODE', 'gust_intensity')
    train_angles = _parse_int_list(raw['TRAIN_ANGLES']) if 'TRAIN_ANGLES' in raw else angles
    val_angles   = _parse_int_list(raw['VAL_ANGLES'])   if 'VAL_ANGLES'   in raw else angles
    test_angles  = _parse_int_list(raw['TEST_ANGLES'])  if 'TEST_ANGLES'  in raw else angles

    # Parse forced-case lists (random_case mode); empty / missing → []
    train_forced_cases = _parse_str_list(raw['TRAIN_CASES']) if 'TRAIN_CASES' in raw else []
    val_forced_cases   = _parse_str_list(raw['VAL_CASES'])   if 'VAL_CASES'   in raw else []
    test_forced_cases  = _parse_str_list(raw['TEST_CASES'])  if 'TEST_CASES'  in raw else []

    # When angle mode, derive ANGLES from the union (same as train_SNN.py)
    if split_mode == 'angle':
        angles = sorted(set(train_angles + val_angles + test_angles))

    encoding = raw['ENCODING']
    time_steps = int(raw['TIME_STEPS'])
    warmup_steps = int(raw.get('WARMUP_STEPS', 0))   # 0 for runs trained before warmup support
    stride = int(raw['STRIDE']) if 'STRIDE' in raw else time_steps
    bins = int(raw.get('BINS', 1))  # not present for delta encoding; default 1

    # ---- Datasets root resolution ----
    env = os.environ.get('THESIS_DATA_ROOT')
    if env:
        datasets_root = os.path.abspath(env)
    else:
        base = os.path.join(_HERE, '..')
        datasets_root = next(
            (os.path.normpath(os.path.join(base, n))
             for n in ('Datasets', 'DATASETS', 'datasets')
             if os.path.isdir(os.path.normpath(os.path.join(base, n)))),
            os.path.normpath(base)   # legacy fallback: datasets live under the repo root
        )
    orig_dataset_path = os.path.join(datasets_root, 'Dataset_original')

    # ---- Dataset folder resolution (new: DATASET key; legacy: DATASET_TYPE fields) ----
    if 'DATASET' in raw:
        dataset        = raw['DATASET'].strip()
        dataset_type   = "original" if dataset == "Dataset_original" else "interpolated"
        _n_m           = re.search(r'_N(\d+)$', dataset)
        n_interpolated = int(_n_m.group(1)) if _n_m else 0
        _im_m          = re.search(r'Dataset_interpolated_(.+)_N\d+$', dataset)
        interp_method  = _im_m.group(1) if _im_m else 'cubic_spline'
    else:
        # Legacy snapshot written before DATASET was introduced
        dataset_type   = raw.get('DATASET_TYPE', 'original').strip()
        interp_method  = raw.get('INTERP_METHOD', 'cubic_spline').strip()
        n_interpolated = int(raw.get('N_INTERPOLATED', 0))
        dataset = (f"Dataset_interpolated_{interp_method}_N{n_interpolated}"
                   if dataset_type == 'interpolated' else "Dataset_original")

    dataset_path = os.path.join(datasets_root, dataset)

    if dataset_type == 'interpolated':
        interp_root   = dataset_path
        vorticity_dir = os.path.join(interp_root, 'vorticity')
        lift_dir      = os.path.join(interp_root, 'lift')
        cache_dir     = os.path.join(interp_root, 'cache')
        ds_tag        = f"_ds-{dataset}"
    else:
        interp_root   = None
        vorticity_dir = os.path.join(dataset_path, 'vorticity')
        lift_dir      = os.path.join(dataset_path, 'lift')
        cache_dir     = os.path.join(dataset_path, 'cache')
        ds_tag        = ""

    angles_tag = "_".join(str(a) for a in sorted(angles))
    stride_tag = f"_s{stride}" if stride != time_steps else ""
    warmup_tag = f"_W{warmup_steps}" if warmup_steps > 0 else ""
    norm_pct = int(raw.get('NORM_PERCENTILE', 100))
    gamma_v = float(raw.get('GAMMA', 1.0))
    gain_v = float(raw.get('GAIN', 1.0))
    norm_tag = f"_p{norm_pct}" if norm_pct < 100 else ""
    amp_tag = f"_g{gamma_v}_G{gain_v}" if (gamma_v != 1.0 or gain_v != 1.0) else ""
    enc_str = encoding.replace('+', '_')                             # filesystem-safe
    delta_th_tag = (f"_deltaTh{float(raw.get('THRESHOLD_FACTOR', 1.0))}"
                    if encoding == 'rate+delta' else "")

    # Resolve the model type from the snapshot. The only models are the spiking
    # SNN and the two non-spiking CNN baselines; anything else maps to 'snn'.
    _model_tag = (f"{raw.get('MODEL_TYPE', '')} {raw.get('MODEL', '')} "
                  f"{raw.get('MODEL_VERSION', '')}")
    if 'cnn_param_matched' in _model_tag:
        model_type = 'cnn_param_matched'
    elif 'cnn_energy_matched' in _model_tag:
        model_type = 'cnn_energy_matched'
    else:
        model_type = 'snn'

    cfg = SimpleNamespace(
        TRAINING_DESCRIPTION=raw.get('TRAINING_DESCRIPTION', 'unknown'),
        MODEL_TYPE=model_type,
        DATASET_TYPE=dataset_type,
        INTERP_METHOD=interp_method,
        N_INTERPOLATED=n_interpolated,
        INTERP_ROOT=interp_root,
        ENCODING=encoding,
        THRESHOLD_FACTOR=float(raw.get('THRESHOLD_FACTOR', 1.4)),
        ZERO_FIRST_FRAME=raw.get('ZERO_FIRST_FRAME', 'True') in (True, 'True'),
        BINS=bins,
        NORM_PERCENTILE=int(raw.get('NORM_PERCENTILE', 100)),
        GAMMA=float(raw.get('GAMMA', 1.0)),
        GAIN=float(raw.get('GAIN', 1.0)),
        TIME_STEPS=time_steps,
        WARMUP_STEPS=warmup_steps,
        STRIDE=stride,
        ANGLES=angles,
        SPLIT_MODE=split_mode,
        TRAIN_ANGLES=train_angles,
        VAL_ANGLES=val_angles,
        TEST_ANGLES=test_angles,
        VAL_RATIO=float(raw['VAL_RATIO']),
        TEST_RATIO=float(raw['TEST_RATIO']),
        TRAIN_CASES=train_forced_cases,
        VAL_CASES=val_forced_cases,
        TEST_CASES=test_forced_cases,
        N_Z=int(raw['N_Z']),
        IN_CHANNELS=int(raw.get('IN_CHANNELS', 2)),
        BETA_INIT=float(raw['BETA_INIT']),
        FC_THRESHOLD=float(raw.get('FC_THRESHOLD', 0.5)),
        LOSS_TYPE=raw.get('LOSS_TYPE', 'mse'),
        # Paths
        VORTICITY_DIR=vorticity_dir,
        LIFT_DIR=lift_dir,
        RD_PARAMS_PATH=os.path.join(datasets_root, 'RD_list_FT2023.xlsx'),
        CACHE_PATH=os.path.join(
            cache_dir,
            f"processed_{enc_str}_T{time_steps}{warmup_tag}{stride_tag}_b{bins}{norm_tag}{amp_tag}{delta_th_tag}_a{angles_tag}{ds_tag}.pt",
        ),
        # The original-resolution lift dir is needed for REPORT_AT_ORIGINAL_GRID
        # comparisons even when the model was trained on interpolated data.
        ORIGINAL_LIFT_DIR=os.path.join(orig_dataset_path, 'lift'),
        DATA_ROOT=datasets_root,
        RESULTS_BASE_DIR=os.path.join(_HERE, '..', 'results'),
        MODEL_WEIGHTS=ckpt_dir,
        BEST_MODEL_PATH=best_model_path,
        SOURCE_SNAPSHOT_PATH=snapshot_path,
    )
    return cfg


def _print_config(cfg):
    """Print a readable summary of the test configuration.

    Only shows parameters relevant to the selected model version,
    encoding mode, split strategy, and loss function.
    """
    model_type = cfg.MODEL_TYPE
    enc = cfg.ENCODING
    loss = cfg.LOSS_TYPE
    _model_names = {"snn": "SNN_Model",
                    "cnn_param_matched": "ParameterMatchedCNN",
                    "cnn_energy_matched": "EnergyMatchedCNN"}
    model_name = _model_names.get(model_type, model_type)

    print("=" * 60)
    print("  TEST CONFIGURATION (FROM CHECKPOINT SNAPSHOT)")
    print("=" * 60)
    print(f"  Description:  {cfg.TRAINING_DESCRIPTION}")
    print(f"  Model:        {model_name} (N_z={cfg.N_Z})")

    # --- Encoding ---
    if enc == "rate":
        print(f"  Encoding:     {enc} (bins={cfg.BINS})")
        print(f"  Transfer fn:  p = min(1, {getattr(cfg, 'GAIN', 1.0)} · "
              f"|x|^{getattr(cfg, 'GAMMA', 1.0)})")
    elif enc == "delta":
        print(f"  Encoding:     {enc} "
              f"(threshold_factor={cfg.THRESHOLD_FACTOR}, "
              f"zero_first={cfg.ZERO_FIRST_FRAME})")
    elif enc == "rate+delta":
        print(f"  Encoding:     {enc} (bins={cfg.BINS})")
        print(f"  Rate fn:      p = min(1, {getattr(cfg, 'GAIN', 1.0)} · "
              f"|x|^{getattr(cfg, 'GAMMA', 1.0)})")
        print(f"  Delta:        threshold_factor={cfg.THRESHOLD_FACTOR}, "
              f"zero_first={cfg.ZERO_FIRST_FRAME}")
    else:
        print(f"  Encoding:     {enc}")

    # --- Data ---
    print(f"  Time steps:   {cfg.TIME_STEPS} supervised"
          + (f" (stride={cfg.STRIDE})" if hasattr(cfg, 'STRIDE') else ""))
    _ws = getattr(cfg, 'WARMUP_STEPS', 0)
    if _ws > 0:
        print(f"  Warmup steps: {_ws}  (model input = {_ws + cfg.TIME_STEPS} total, "
              f"first {_ws} outputs discarded)")
    if USE_WARMUP_PASS:
        print(f"  Warmup pass:  ON  (each case fed twice, bridged by "
              f"{WARMUP_INTERP_POINTS}-pt cubic spline; only 2nd pass scored)")
    print(f"  Angles:       {cfg.ANGLES}")

    # --- Split ---
    split_mode = getattr(cfg, 'SPLIT_MODE', 'gust_intensity')
    if split_mode == 'angle':
        print(f"  Split mode:   angle-of-attack")
        print(f"  Train angles: {cfg.TRAIN_ANGLES}")
        print(f"  Val angles:   {cfg.VAL_ANGLES} (ratio={cfg.VAL_RATIO})")
        print(f"  Test angles:  {cfg.TEST_ANGLES} (all cases)")
    elif split_mode == 'random_case':
        print(f"  Split mode:   random_case (stratified by |G|, all angles)")
        print(f"  Val ratio:    {cfg.VAL_RATIO}")
        print(f"  Test ratio:   {cfg.TEST_RATIO}")
    else:
        print(f"  Split:        {1 - cfg.VAL_RATIO - cfg.TEST_RATIO:.0%} train / "
              f"{cfg.VAL_RATIO:.0%} val / {cfg.TEST_RATIO:.0%} test (by |G|)")

    # --- Architecture ---
    arch_parts = [f"N_z={cfg.N_Z}", f"beta={cfg.BETA_INIT}"]
    if model_type == "snn":
        arch_parts.append(f"fc_threshold={getattr(cfg, 'FC_THRESHOLD', 0.5)}")
    print(f"  Architecture: {', '.join(arch_parts)}")

    # --- Loss ---
    print(f"  Loss:         {loss} (MSE between reconstructed Cl and ground-truth Cl)")

    print(f"  Checkpoint:   {cfg.MODEL_WEIGHTS}")
    print("=" * 60)


def _get_config_snapshot(cfg):
    """Serialize test-time configuration to a string.

    Only includes parameters relevant to the selected model version,
    encoding mode, and loss function.
    """
    model_type = cfg.MODEL_TYPE
    enc = cfg.ENCODING
    loss = cfg.LOSS_TYPE

    lines = [
        "TEST CONFIGURATION SNAPSHOT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50,
        f"TRAINING_DESCRIPTION = {cfg.TRAINING_DESCRIPTION}",
    ]
    if getattr(cfg, 'DATASET_TYPE', 'original') == 'interpolated':
        lines += [
            "",
            "DATASET",
            "-" * 30,
            f"DATASET_TYPE         = {cfg.DATASET_TYPE}",
            f"INTERP_METHOD        = {cfg.INTERP_METHOD}",
            f"N_INTERPOLATED       = {cfg.N_INTERPOLATED}",
            f"INTERP_ROOT          = {cfg.INTERP_ROOT}",
            f"REPORT_AT_ORIGINAL_GRID = {REPORT_AT_ORIGINAL_GRID}",
        ]
    else:
        lines += [
            "",
            "DATASET",
            "-" * 30,
            f"DATASET_TYPE         = original",
        ]
    lines += [
        "",
        "ENCODING",
        "-" * 30,
        f"ENCODING             = {enc}",
    ]
    if enc == "rate":
        lines.append(f"BINS                 = {cfg.BINS}")
        lines.append(f"NORM_PERCENTILE      = {getattr(cfg, 'NORM_PERCENTILE', 100)}")
        lines.append(f"GAMMA                = {getattr(cfg, 'GAMMA', 1.0)}")
        lines.append(f"GAIN                 = {getattr(cfg, 'GAIN', 1.0)}")
    elif enc == "delta":
        lines.append(f"NORM_PERCENTILE      = {getattr(cfg, 'NORM_PERCENTILE', 100)}")
        lines.append(f"THRESHOLD_FACTOR     = {cfg.THRESHOLD_FACTOR}")
        lines.append(f"ZERO_FIRST_FRAME     = {cfg.ZERO_FIRST_FRAME}")
    elif enc == "rate+delta":
        lines.append(f"BINS                 = {cfg.BINS}")
        lines.append(f"NORM_PERCENTILE      = {getattr(cfg, 'NORM_PERCENTILE', 100)}")
        lines.append(f"GAMMA                = {getattr(cfg, 'GAMMA', 1.0)}")
        lines.append(f"GAIN                 = {getattr(cfg, 'GAIN', 1.0)}")
        lines.append(f"THRESHOLD_FACTOR     = {cfg.THRESHOLD_FACTOR}")
        lines.append(f"ZERO_FIRST_FRAME     = {cfg.ZERO_FIRST_FRAME}")

    lines += [
        "",
        "DATA",
        "-" * 30,
        f"TIME_STEPS           = {cfg.TIME_STEPS}",
        f"WARMUP_STEPS         = {getattr(cfg, 'WARMUP_STEPS', 0)}",
        f"STRIDE               = {cfg.STRIDE}",
        f"ANGLES               = {cfg.ANGLES}",
        f"VAL_RATIO            = {cfg.VAL_RATIO}",
        f"TEST_RATIO           = {cfg.TEST_RATIO}",
        "",
        "WARMUP PASS",
        "-" * 30,
        f"USE_WARMUP_PASS      = {USE_WARMUP_PASS}",
        f"WARMUP_INTERP_POINTS = {WARMUP_INTERP_POINTS}",
    ]
    if getattr(cfg, 'SPLIT_MODE', 'gust_intensity') == 'random_case':
        lines.append(f"TRAIN_CASES          = {getattr(cfg, 'TRAIN_CASES', [])}")
        lines.append(f"VAL_CASES            = {getattr(cfg, 'VAL_CASES', [])}")
        lines.append(f"TEST_CASES           = {getattr(cfg, 'TEST_CASES', [])}")
    lines += [
        "",
        "ARCHITECTURE",
        "-" * 30,
        f"MODEL_TYPE           = {model_type}",
        f"N_Z                  = {cfg.N_Z}",
        f"IN_CHANNELS          = {cfg.IN_CHANNELS}",
        f"BETA_INIT            = {cfg.BETA_INIT}",
    ]
    if model_type == "snn":
        lines.append(f"FC_THRESHOLD         = {getattr(cfg, 'FC_THRESHOLD', 0.5)}")

    lines += [
        "",
        "LOSS",
        "-" * 30,
        f"LOSS_TYPE            = {loss}  (MSE between reconstructed Cl and ground-truth Cl)",
        "",
        "CHECKPOINT",
        f"MODEL_WEIGHTS        = {cfg.MODEL_WEIGHTS}",
        f"BEST_MODEL_PATH      = {cfg.BEST_MODEL_PATH}",
        f"SOURCE_SNAPSHOT      = {cfg.SOURCE_SNAPSHOT_PATH}",
        f"MAX_ANIM_SAMPLES     = {MAX_ANIMATION_SAMPLES}",
        "=" * 50,
    ]
    return "\n".join(lines)


# ==========================================
# RESULTS DIRECTORY
# ==========================================
_RESULTS_SUBFOLDER = {
    "snn":                "SNN_results",
    "cnn_param_matched":  "Parameter_matched_results",
    "cnn_energy_matched": "Energy_matched_results",
}


def create_results_dir(cfg):
    """Create timestamped results directory with subdirectories.

    Results are grouped by model type — not by N_Z — so runs at any latent
    dimension (Nz8, Nz16, Nz32, ...) for a given architecture land side by side
    under the same results/<model>_results/ folder.
    """
    timestamp = datetime.now().strftime("%B_%d_%Y_%Hh_%Mm")
    subfolder = _RESULTS_SUBFOLDER[cfg.MODEL_TYPE]
    results_dir = os.path.join(cfg.RESULTS_BASE_DIR, subfolder, timestamp)
    for subdir in ['summary', 'latent', 'samples']:
        os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
    return results_dir


# ==========================================
# CASE LABEL HELPERS
# ==========================================
# _case_tag is defined in pre_encoder.py and imported below to be reused for
# the forced-case parser and the dataset_split.txt writer.

def _case_title(angle, case_id, rd_params):
    """Build a plot title string like 'AoA=20, RD5, G=3.2'."""
    g_val = 0.0
    if angle in rd_params and case_id in rd_params[angle]:
        g_val = rd_params[angle][case_id]['G']
    return f"AoA={angle}, {case_id}, G={g_val:.2f}"


# ==========================================
# GROUPING HELPER
# ==========================================
def _group_results_by_case(results, max_frames=None):
    """
    Group per-sequence results by (angle, case_id) and concatenate arrays.

    Each test case may produce multiple sequences. This function merges them
    back into one continuous time series per case for visualisation.

    Args:
        max_frames: cap on total timesteps per case. None = no cap (full case).

    Returns:
        list of dicts, one per unique case, with concatenated arrays:
            pred_omega  (F, H, W)
            target_omega (F, H, W)
            pred_cl     (F,)
            target_cl   (F,)
            latent      (F, N_z)
            per_timestep_mse (F,)
            case_tag, case_title, angle, case_id, mse_omega, r2_omega, ...
    """
    from collections import OrderedDict
    groups = OrderedDict()
    for r in results:
        key = (r['angle'], r['case_id'])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    grouped = []
    for (angle, case_id), seqs in groups.items():
        # Sort by sample_idx to preserve temporal ordering
        seqs = sorted(seqs, key=lambda r: r['sample_idx'])

        # Concatenate arrays — no cap when max_frames is None
        has_omega = seqs[0]['pred_omega'] is not None
        if has_omega:
            pred_omega   = np.concatenate([s['pred_omega'] for s in seqs], axis=0)
            target_omega = np.concatenate([s['target_omega'] for s in seqs], axis=0)
            per_t_mse    = np.concatenate([s['per_timestep_mse'] for s in seqs])
            if max_frames is not None:
                pred_omega   = pred_omega[:max_frames]
                target_omega = target_omega[:max_frames]
                per_t_mse    = per_t_mse[:max_frames]
        pred_cl   = np.concatenate([s['pred_cl'] for s in seqs], axis=0)
        target_cl = np.concatenate([s['target_cl'] for s in seqs], axis=0)
        latent    = np.concatenate([s['latent'] for s in seqs], axis=0)
        if max_frames is not None:
            pred_cl   = pred_cl[:max_frames]
            target_cl = target_cl[:max_frames]
            latent    = latent[:max_frames]

        # Propagate original-grid arrays if present in every sequence
        _orig_preds   = [s.get('pred_cl_orig')   for s in seqs]
        _orig_targets = [s.get('target_cl_orig') for s in seqs]
        if all(a is not None for a in _orig_preds):
            pred_cl_orig   = np.concatenate(_orig_preds,   axis=0)
            target_cl_orig = np.concatenate(_orig_targets, axis=0)
        else:
            pred_cl_orig   = None
            target_cl_orig = None

        # Cl metrics will be recomputed at encounter level after grouping
        avg = lambda key: float(np.mean([s[key] for s in seqs]))
        entry = {
            'angle': angle,
            'case_id': case_id,
            'case_tag': seqs[0]['case_tag'],
            'case_title': seqs[0]['case_title'],
            'gust': seqs[0].get('gust', 0.0),
            'pred_cl': pred_cl,
            'target_cl': target_cl,
            'pred_cl_orig':   pred_cl_orig,
            'target_cl_orig': target_cl_orig,
            'latent': latent,
            # Warmup is a per-case quantity (only the first chunk has cold-start
            # frames); take it from the first sequence of the group.
            'warmup_steps':      int(seqs[0].get('warmup_steps', 0)),
            'warmup_steps_orig': int(seqs[0].get('warmup_steps_orig', 0)),
            # Warmup-pass arrays — only present on the interpolated path; each
            # case yields a single sequence there, so just take seqs[0].
            'pred_cl_warmup':   seqs[0].get('pred_cl_warmup'),
            'pred_cl_interp':   seqs[0].get('pred_cl_interp'),
            'target_cl_warmup': seqs[0].get('target_cl_warmup'),
            'target_cl_interp': seqs[0].get('target_cl_interp'),
        }
        if has_omega:
            entry.update({
                'pred_omega': pred_omega,
                'target_omega': target_omega,
                'per_timestep_mse': per_t_mse,
                'mse_omega': avg('mse_omega'),
                'rel_error_omega': avg('rel_error_omega'),
                'r2_omega': avg('r2_omega'),
            })
        grouped.append(entry)

    return grouped


# ==========================================
# CENTRAL Cl METRIC FUNCTION
# ==========================================
def compute_cl_metrics(pred, gt, eps=1e-8):
    """
    Compute Cl trajectory metrics for one trajectory.
    pred, gt: array-like of shape (T,). Unit timestep spacing assumed.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt,   dtype=np.float64)

    err     = pred - gt
    abs_err = np.abs(err)

    mse  = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(abs_err))

    rel_l2_error = float(np.linalg.norm(err) / (np.linalg.norm(gt) + eps))

    cosine_similarity = float(
        np.dot(pred, gt) / (np.linalg.norm(pred) * np.linalg.norm(gt) + eps)
    )

    pred_c = pred - np.mean(pred)
    gt_c   = gt   - np.mean(gt)
    centered_cosine_similarity = float(
        np.dot(pred_c, gt_c) / (np.linalg.norm(pred_c) * np.linalg.norm(gt_c) + eps)
    )

    pred_peak_idx = int(np.argmax(np.abs(pred)))
    gt_peak_idx   = int(np.argmax(np.abs(gt)))
    pred_peak_val = pred[pred_peak_idx]
    gt_peak_val   = gt[gt_peak_idx]

    peak_cl_error        = float(abs(abs(pred_peak_val) - abs(gt_peak_val)))
    peak_cl_error_signed = float(abs(pred_peak_val - gt_peak_val))
    peak_timing_error    = int(abs(pred_peak_idx - gt_peak_idx))

    integrated_abs_error            = float(np.trapezoid(abs_err))
    integrated_sq_error             = float(np.trapezoid(err ** 2))
    normalized_integrated_abs_error = float(
        integrated_abs_error / (np.trapezoid(np.abs(gt)) + eps)
    )

    return {
        "mse_cl":                          mse,
        "rmse_cl":                         rmse,
        "mae_cl":                          mae,
        "rel_l2_error_cl":                 rel_l2_error,
        "cosine_similarity_cl":            cosine_similarity,
        "centered_cosine_similarity_cl":   centered_cosine_similarity,
        "peak_cl_error":                   peak_cl_error,
        "peak_cl_error_signed":            peak_cl_error_signed,
        "peak_timing_error":               peak_timing_error,
        "integrated_abs_error":            integrated_abs_error,
        "integrated_sq_error":             integrated_sq_error,
        "normalized_integrated_abs_error": normalized_integrated_abs_error,
    }


# ==========================================
# WARMUP-PASS HELPER
# ==========================================
def _build_smooth_transition(end_frames, start_frames, n_interp):
    """
    Build a cubic-spline bridge that smoothly joins the tail of one sequence
    to the head of another. Used by the warmup-pass inference path to avoid a
    discontinuity at the seam where the case is concatenated to itself.

    The spline is fit through `end_frames` (the last M frames before the seam,
    in order) and `start_frames` (the first M frames after the seam, in order).
    Using two anchors on each side gives the spline access to the local
    derivatives, so the bridge inherits the slope at both ends rather than
    arriving with a flat tangent.

    Args:
        end_frames:   array (M_end,   ...) ordered, just before the bridge
        start_frames: array (M_start, ...) ordered, just after the bridge
        n_interp:     number of interpolated samples to generate

    Returns:
        bridge: array (n_interp, ...) of the same dtype as the input
    """
    M_end   = end_frames.shape[0]
    M_start = start_frames.shape[0]
    # Place anchors so the bridge spans indices [1 .. n_interp]; end frames
    # sit at non-positive indices and start frames at indices > n_interp.
    t_anchor = np.concatenate([
        np.arange(-(M_end - 1), 1),                       # ..., -2, -1, 0
        np.arange(n_interp + 1, n_interp + 1 + M_start),  # N+1, N+2, ...
    ])
    y_anchor = np.concatenate([end_frames, start_frames], axis=0)
    cs = CubicSpline(t_anchor, y_anchor, axis=0)
    t_eval = np.arange(1, n_interp + 1)
    return cs(t_eval)


# ==========================================
# INFERENCE — INTERPOLATED DATASET
# ==========================================
def _run_inference_interpolated(model, data, device, cfg):
    """Run full-case inference on each test case of the interpolated dataset.

    Each interpolated case is a continuous sequence of Nt_new = (Nt-1)*(N+1)+1
    timesteps. We load the case .npy with mmap, normalize using the saved
    norm_scale, encode the full case to spikes, and forward through the model
    once per case under torch.no_grad(). Returns a list of result dicts (one
    per test case) compatible with the downstream grouping and KPIs.

    Metrics are reported at full interpolated resolution by default. If
    REPORT_AT_ORIGINAL_GRID is True, the prediction is also subsampled to the
    original 1190-frame grid and a parallel set of metrics is computed against
    the original-grid Cl ground truth read from Dataset/lift/...csv.
    """
    from pre_encoder import (
        list_interpolated_cases,
        compute_norm_scale_streaming,
        compute_delta_threshold_streaming,
        _hybrid_encode_full_case,
        _delta_encode_full_case,
        _rate_encode_full_case,
        _pre_compute_case_split_map,
        load_lift,
    )

    case_paths = data['case_paths']
    case_paths_by_key = {(a, c): (vp, cp) for a, c, vp, cp in case_paths}
    case_split_map = data['case_split_map']
    rd_params = data['rd_params']
    norm_scale = float(data['norm_stats']['norm_scale'])
    threshold = data.get('threshold', None)

    n_interp = int(getattr(cfg, 'N_INTERPOLATED', 0))
    encoding = cfg.ENCODING
    gamma = float(getattr(cfg, 'GAMMA', 1.0))
    gain  = float(getattr(cfg, 'GAIN', 1.0))
    zero_first = bool(getattr(cfg, 'ZERO_FIRST_FRAME', True))
    is_cnn = _is_cnn(cfg)   # non-spiking CNN: raw frames, no encoding/warmup pass

    # Test cases come from the case_split_map
    test_cases = sorted(
        [k for k, v in case_split_map.items() if v == 'test'],
    )
    print(f"\n>>> Running interpolated inference on {len(test_cases)} test cases "
          f"(full T per case)...")
    print(f"  (norm_scale={norm_scale:.4f}, threshold={threshold})")
    if USE_WARMUP_PASS:
        print(f"  (warmup pass ON — each case fed twice, bridged by "
              f"{WARMUP_INTERP_POINTS}-pt cubic spline; KPIs use 2nd pass only)")

    model.eval()
    results = []

    with torch.inference_mode():
        for angle, case_id in test_cases:
            if (angle, case_id) not in case_paths_by_key:
                print(f"  [WARN] Test case ({angle}, {case_id}) not found in case_paths, skipping.")
                continue
            vort_path, cl_path = case_paths_by_key[(angle, case_id)]
            tag = _case_tag(angle, case_id, rd_params)
            title = _case_title(angle, case_id, rd_params)
            g_val = 0.0
            if angle in rd_params and case_id in rd_params[angle]:
                g_val = rd_params[angle][case_id]['G']

            cl_full = np.load(cl_path).astype(np.float32, copy=False)

            if is_cnn:
                # ---- Non-spiking CNN: RAW continuous frames, no normalization,
                # clipping, encoding, or warmup pass. Predict Cl per frame. ----
                vort_mm = np.load(vort_path, mmap_mode='r')
                nt = int(vort_mm.shape[0])
                cl_chunks, z_chunks = [], []
                for i in range(0, nt, CNN_FRAME_BATCH):
                    frames = np.array(vort_mm[i:i + CNN_FRAME_BATCH], dtype=np.float32)
                    xb = torch.from_numpy(frames).unsqueeze(1).to(device)  # (B,1,H,W)
                    zb, clb = model(xb, return_latent=True)
                    cl_chunks.append(clb[:, 0].float().cpu().numpy())
                    z_chunks.append(zb.float().cpu().numpy())
                    del xb, zb, clb
                del vort_mm
                cl_pred_np_full = np.concatenate(cl_chunks)
                z_seq_np_full   = np.concatenate(z_chunks)
                n_bridge = 0
                cl_bridge = None
            else:
                # Load + normalize full case
                vort_mm = np.load(vort_path, mmap_mode='r')
                omega = np.array(vort_mm, dtype=np.float32)
                del vort_mm
                omega *= 1.0 / (norm_scale + 1e-8)
                np.clip(omega, -1.0, 1.0, out=omega)

                # ---- Optional warmup pass: feed the case twice, joined by a smooth
                # cubic-spline bridge. Only the second pass is scored. ----
                Nt_case = omega.shape[0]
                if USE_WARMUP_PASS:
                    n_bridge = int(WARMUP_INTERP_POINTS)
                    omega_bridge = _build_smooth_transition(
                        omega[-2:], omega[:2], n_bridge
                    ).astype(np.float32, copy=False)
                    np.clip(omega_bridge, -1.0, 1.0, out=omega_bridge)
                    cl_bridge = _build_smooth_transition(
                        cl_full[-2:], cl_full[:2], n_bridge
                    ).astype(np.float32, copy=False)
                    omega_enc_input = np.concatenate([omega, omega_bridge, omega], axis=0)
                    del omega_bridge
                else:
                    n_bridge = 0
                    cl_bridge = None
                    omega_enc_input = omega

                # Encode full case (twice + bridge when warmup is on) to spikes
                if encoding == 'rate+delta':
                    spikes_full = _hybrid_encode_full_case(
                        omega_enc_input, threshold, gamma=gamma, gain=gain,
                        zero_first_frame=zero_first)
                elif encoding == 'delta':
                    spikes_full = _delta_encode_full_case(
                        omega_enc_input, threshold, zero_first_frame=zero_first)
                elif encoding == 'rate':
                    spikes_full = _rate_encode_full_case(omega_enc_input, gamma=gamma, gain=gain)
                else:
                    raise ValueError(f"Unsupported ENCODING={encoding!r} for interpolated inference")
                del omega, omega_enc_input

                # Forward: model expects (T, B, C, Ny, Nx) float32
                x_input = spikes_full.unsqueeze(1).to(device, dtype=torch.float32)
                del spikes_full

                z_seq, cl_pred = model(x_input)
                cl_pred = cl_pred.float()
                z_seq = z_seq.float()

                cl_pred_np_full = cl_pred[:, 0, 0].cpu().numpy()
                z_seq_np_full   = z_seq[:, 0].cpu().numpy()
                del x_input, cl_pred, z_seq

            # Split predictions: warmup pass | bridge | scored pass
            if USE_WARMUP_PASS:
                warm_end   = Nt_case
                bridge_end = warm_end + n_bridge
                pred_cl_warmup_np = cl_pred_np_full[:warm_end]
                pred_cl_interp_np = cl_pred_np_full[warm_end:bridge_end]
                cl_pred_np        = cl_pred_np_full[bridge_end:]
                z_warmup_np       = z_seq_np_full[:warm_end]
                z_interp_np       = z_seq_np_full[warm_end:bridge_end]
                z_seq_np          = z_seq_np_full[bridge_end:]
                # The warmup pass already absorbs cold-start artefacts, so the
                # legacy per-case WARMUP_STEPS skip is bypassed.
                W_eval = 0
            else:
                pred_cl_warmup_np = None
                pred_cl_interp_np = None
                z_warmup_np = None
                z_interp_np = None
                cl_pred_np = cl_pred_np_full
                z_seq_np   = z_seq_np_full
                # Burn-in: exclude the first WARMUP_EVAL_FRAMES from scoring
                # (test-time override of the trained cfg.WARMUP_STEPS).
                W_eval = (int(WARMUP_EVAL_FRAMES) if WARMUP_EVAL_FRAMES
                          else int(getattr(cfg, 'WARMUP_STEPS', 0)))

            # FULL-grid Cl GT comes from the interpolated lift .npy
            target_cl_full = cl_full

            if W_eval > 0:
                cl_pred_eval   = cl_pred_np[W_eval:]
                target_cl_eval = target_cl_full[W_eval:]
            else:
                cl_pred_eval   = cl_pred_np
                target_cl_eval = target_cl_full

            full_metrics = compute_cl_metrics(cl_pred_eval, target_cl_eval)

            # Optional ORIGINAL-grid comparison
            orig_metrics = None
            if REPORT_AT_ORIGINAL_GRID and n_interp > 0:
                step = n_interp + 1
                pred_orig = cl_pred_np[::step]
                # Prefer reading the un-interpolated CSV when available; fall
                # back to subsampled interpolated GT (should match within
                # float tolerance for any interpolating method).
                csv_path = os.path.join(
                    cfg.ORIGINAL_LIFT_DIR, f"lift_a{angle}",
                    f"Lift_AoA{angle}_{case_id}.csv",
                )
                try:
                    target_orig = load_lift(csv_path)
                except FileNotFoundError:
                    target_orig = target_cl_full[::step]
                # Length-align (interpolated GT subsampled → exact original length)
                L = min(len(pred_orig), len(target_orig))
                pred_orig = pred_orig[:L]
                target_orig = target_orig[:L]
                # Warmup on the original grid: W_eval interpolated frames cover
                # ceil(W_eval/step) original-grid frames.
                W_orig = -(-W_eval // step) if step > 0 else W_eval
                pred_orig_eval   = pred_orig[W_orig:]   if W_orig > 0 else pred_orig
                target_orig_eval = target_orig[W_orig:] if W_orig > 0 else target_orig
                orig_metrics = compute_cl_metrics(pred_orig_eval, target_orig_eval)

            # Build a single result entry per case (sample_idx=0 — one per case
            # so _group_results_by_case treats each as its own group with one seq).
            results.append({
                'sample_idx': 0,
                'angle': angle,
                'case_id': case_id,
                'case_tag': tag,
                'case_title': title,
                'gust': g_val,
                **full_metrics,
                'pred_cl': cl_pred_np,
                'target_cl': target_cl_full,
                'pred_cl_orig':   pred_orig   if orig_metrics is not None else None,
                'target_cl_orig': target_orig if orig_metrics is not None else None,
                'latent': z_seq_np,
                # Warmup-pass arrays (None when USE_WARMUP_PASS is False)
                'pred_cl_warmup':   pred_cl_warmup_np,
                'pred_cl_interp':   pred_cl_interp_np,
                'target_cl_warmup': target_cl_full if USE_WARMUP_PASS else None,
                'target_cl_interp': cl_bridge,
                'latent_warmup':    z_warmup_np,
                'latent_interp':    z_interp_np,
                'mse_omega': float('nan'),
                'rel_error_omega': float('nan'),
                'r2_omega': float('nan'),
                'per_timestep_mse': np.array([]),
                'pred_omega': None,
                'target_omega': None,
                'metrics_original_grid': orig_metrics,
                'warmup_steps':      W_eval,
                'warmup_steps_orig': (-(-W_eval // (n_interp + 1))
                                      if (REPORT_AT_ORIGINAL_GRID and n_interp > 0)
                                      else W_eval),
            })

            tag_full = f"MSE_Cl: {full_metrics['mse_cl']:.6f} | RMSE_Cl: {full_metrics['rmse_cl']:.4f}"
            if orig_metrics is not None:
                tag_full += (f" | [orig grid] MSE: {orig_metrics['mse_cl']:.6f} "
                             f"RMSE: {orig_metrics['rmse_cl']:.4f}")
            print(f"  {tag:30s} | {tag_full} (T={len(cl_pred_np)})")

    # Summary
    print(f"\n{'='*60}")
    print(f"INTERPOLATED TEST SUMMARY ({len(results)} cases)")
    print(f"{'='*60}")
    avg = lambda key, src=results: float(np.nanmean([r[key] for r in src]))
    print(f"[METRICS @ FULL T] "
          f"MSE: {avg('mse_cl'):.6f}  RMSE: {avg('rmse_cl'):.6f}  "
          f"RelL2Err: {avg('rel_l2_error_cl'):.4f}  "
          f"CosineSim: {avg('cosine_similarity_cl'):.4f}  "
          f"PeakErr: {avg('peak_cl_error'):.4f}")
    if REPORT_AT_ORIGINAL_GRID and any(r['metrics_original_grid'] for r in results):
        keys_orig = list(results[0]['metrics_original_grid'].keys())
        avg_orig = {k: float(np.nanmean(
            [r['metrics_original_grid'][k] for r in results
             if r['metrics_original_grid'] is not None])) for k in keys_orig}
        print(f"[METRICS @ ORIG T] "
              f"MSE: {avg_orig['mse_cl']:.6f}  RMSE: {avg_orig['rmse_cl']:.6f}  "
              f"RelL2Err: {avg_orig['rel_l2_error_cl']:.4f}  "
              f"CosineSim: {avg_orig['cosine_similarity_cl']:.4f}  "
              f"PeakErr: {avg_orig['peak_cl_error']:.4f}")
    print(f"{'='*60}")

    return results


# ==========================================
# INFERENCE
# ==========================================
def run_inference(model, data, device, decoder_free=False, cfg=None):
    """
    Run model inference on all test samples.

    For decoder-free models (v5+), uses **continuous per-case inference**:
    all chunks of a test case are concatenated into one long sequence so the
    SNN membrane states and LSTM hidden state carry across chunk boundaries.
    This eliminates the prediction spikes at t=T, 2T, 3T, ... that arise
    from cold-starting every TIME_STEPS frames.

    For v1-v4 models, runs per-sequence inference (original behaviour).

    Returns a list of result dicts, one per test sequence.
    """
    if cfg is not None and (
        getattr(cfg, 'DATASET_TYPE', 'original') == 'interpolated'
        or getattr(cfg, 'TRAINING_MODE', 'random_window') == 'stateful_tbptt'
    ):
        # Both the interpolated dataset and the original-dataset stateful path
        # use per-case .npy files via case_paths — `_run_inference_interpolated`
        # encodes from these on the fly and is dataset-agnostic.
        return _run_inference_interpolated(model, data, device, cfg)

    spikes = data['spikes']
    omega_seqs = data.get('omega_seqs', None)   # None for decoder-free models (v5+)
    cl_targets = data['targets']
    test_indices = data['split_indices']['test']
    case_labels = data['case_labels']
    rd_params = data['rd_params']

    model.eval()
    results = []

    with torch.inference_mode():
        if decoder_free:
            # ---------------------------------------------------------
            # CONTINUOUS PER-CASE INFERENCE (v5+)
            # ---------------------------------------------------------
            # Group test sequences by case, concatenate chunks, run model
            # once per case.  This gives the LSTM the full temporal
            # evolution without state resets.
            from collections import OrderedDict
            case_groups = OrderedDict()
            for idx in test_indices:
                key = case_labels[idx]
                case_groups.setdefault(key, []).append(idx)

            T_per_seq = cl_targets.shape[1]   # TIME_STEPS (before BINS)
            W = int(getattr(cfg, 'WARMUP_STEPS', 0)) if cfg is not None else 0

            print(f"\n>>> Running continuous inference on {len(case_groups)} "
                  f"test cases ({len(test_indices)} sequences)...")
            print("  (decoder-free — vorticity metrics skipped)")
            print("  (continuous — SNN/LSTM state carried across chunks)")
            if W > 0:
                print(f"  (warmup={W} — first window includes warmup, "
                      f"subsequent windows feed supervised region only)")

            # Stride skip: with overlapping windows, only take every Nth window
            # so concatenation gives non-overlapping sequential coverage.
            # e.g. T=100, stride=50 → stride_skip=2 → windows 0,2,4,... cover
            # frames 0-99, 100-199, 200-299, ... without duplication.
            _time_steps = cfg.TIME_STEPS if cfg is not None else T_per_seq
            _stride_val = getattr(cfg, 'STRIDE', _time_steps) if cfg is not None else _time_steps
            _stride_skip = max(1, _time_steps // _stride_val)

            for (angle, case_id), indices in case_groups.items():
                indices.sort()   # temporal order within case
                # Keep only non-overlapping windows
                indices = indices[::_stride_skip]
                tag = _case_tag(angle, case_id, rd_params)
                title = _case_title(angle, case_id, rd_params)
                g_val = 0.0
                if angle in rd_params and case_id in rd_params[angle]:
                    g_val = rd_params[angle][case_id]['G']

                # Concatenate chunks into one continuous spike sequence.
                # With warmup: each cached window has shape (W+T, ...) where the
                # first W frames overlap with the previous kept window's last W
                # supervised frames. To avoid feeding the model a backward jump
                # in physical time, include warmup only for the FIRST window and
                # feed only the supervised tail (W:) of subsequent windows.
                if W > 0:
                    parts = [spikes[indices[0]]]
                    parts.extend(spikes[i][W:] for i in indices[1:])
                    x_full = torch.cat(parts, dim=0)        # (W + k*T, C, H, W)
                else:
                    x_full = torch.cat([spikes[i] for i in indices], dim=0)
                cl_full = torch.cat([cl_targets[i] for i in indices], dim=0)

                x_input = x_full.unsqueeze(1).to(device, dtype=torch.float32)
                cl_gt_full = cl_full.unsqueeze(-1).unsqueeze(-1).to(
                    device, dtype=torch.float32)

                z_seq, cl_pred = model(x_input)
                z_seq = z_seq.float()
                cl_pred = cl_pred.float()

                # Drop the warmup outputs of the first window (cold-start artifacts).
                # With BINS>1 the model output is bin-multiplied, so scale W accordingly.
                if W > 0:
                    cfg_bins = int(getattr(cfg, 'BINS', 1)) if cfg is not None else 1
                    drop = W * cfg_bins
                    cl_pred = cl_pred[drop:]
                    z_seq   = z_seq[drop:]

                # BINS averaging back to real timestep resolution
                T_gt_total = cl_gt_full.shape[0]
                if (cl_pred.shape[0] > T_gt_total and
                        cl_pred.shape[0] % T_gt_total == 0):
                    bins = cl_pred.shape[0] // T_gt_total
                    cl_pred = cl_pred.view(
                        T_gt_total, bins, *cl_pred.shape[1:]).mean(dim=1)
                    z_seq = z_seq.view(
                        T_gt_total, bins, *z_seq.shape[1:]).mean(dim=1)

                # Case-level metrics for printout
                mse_case  = nn.functional.mse_loss(cl_pred, cl_gt_full).item()
                rmse_case = mse_case ** 0.5

                # Split back into per-sequence windows for downstream code
                for seq_i, idx in enumerate(indices):
                    t0 = seq_i * T_per_seq
                    t1 = t0 + T_per_seq

                    pred_w = cl_pred[t0:t1]
                    gt_w = cl_gt_full[t0:t1]
                    z_w = z_seq[t0:t1]

                    _pred_np = pred_w[:, 0, 0].cpu().numpy()   # (T,)
                    _gt_np   = gt_w[:, 0, 0].cpu().numpy()     # (T,)

                    _cl_metrics = compute_cl_metrics(_pred_np, _gt_np)

                    results.append({
                        'sample_idx': idx,
                        'angle': angle,
                        'case_id': case_id,
                        'case_tag': tag,
                        'case_title': title,
                        'gust': g_val,
                        **_cl_metrics,
                        'pred_cl': _pred_np,
                        'target_cl': _gt_np,
                        'latent': z_w[:, 0].cpu().numpy(),
                        'mse_omega': float('nan'),
                        'rel_error_omega': float('nan'),
                        'r2_omega': float('nan'),
                        'per_timestep_mse': np.array([]),
                        'pred_omega': None,
                        'target_omega': None,
                    })

                print(f"  {tag:30s} | MSE_Cl: {mse_case:.6f} | "
                      f"RMSE_Cl: {rmse_case:.4f} ({len(indices)} seqs)")

                del x_input, cl_gt_full, z_seq, cl_pred, x_full, cl_full

        else:
            # ---------------------------------------------------------
            # PER-SEQUENCE INFERENCE (v1-v4)
            # ---------------------------------------------------------
            W = int(getattr(cfg, 'WARMUP_STEPS', 0)) if cfg is not None else 0
            cfg_bins = int(getattr(cfg, 'BINS', 1)) if cfg is not None else 1

            print(f"\n>>> Running inference on {len(test_indices)} "
                  f"test sequences...")
            if W > 0:
                print(f"  (warmup={W} — first {W} outputs of each window discarded)")
            for idx in test_indices:
                angle, case_id = case_labels[idx]
                tag = _case_tag(angle, case_id, rd_params)
                title = _case_title(angle, case_id, rd_params)

                # spikes[idx] / omega_seqs[idx] have shape (W+T, ...) when W > 0
                x_input = spikes[idx].unsqueeze(1).to(
                    device=device, dtype=torch.float32)
                cl_gt = cl_targets[idx].unsqueeze(1).unsqueeze(2).to(
                    device=device, dtype=torch.float32)
                omega_gt = omega_seqs[idx].unsqueeze(1).unsqueeze(2).to(
                    device=device, dtype=torch.float32)

                recon, z_seq, cl_pred = model(x_input)

                # Drop warmup outputs from model predictions and corresponding
                # warmup frames from omega_gt so that all metrics are computed
                # only on the supervised region.
                if W > 0:
                    drop = W * cfg_bins
                    cl_pred = cl_pred[drop:]
                    z_seq   = z_seq[drop:]
                    recon   = recon[drop:]
                    omega_gt = omega_gt[W:]   # omega has 1 frame per physical step

                # BINS averaging
                T_gt = cl_gt.shape[0]
                if (cl_pred.shape[0] > T_gt and
                        cl_pred.shape[0] % T_gt == 0):
                    bins_per_step = cl_pred.shape[0] // T_gt
                    cl_pred = cl_pred.view(
                        T_gt, bins_per_step, *cl_pred.shape[1:]).mean(dim=1)
                    z_seq = z_seq.view(
                        T_gt, bins_per_step, *z_seq.shape[1:]).mean(dim=1)
                    recon = recon.view(
                        T_gt, bins_per_step, *recon.shape[1:]).mean(dim=1)

                # Vorticity metrics
                mse_omega = nn.functional.mse_loss(recon, omega_gt).item()
                rel_error_omega = (torch.norm(recon - omega_gt) /
                                   torch.norm(omega_gt)).item()
                ss_res = ((recon - omega_gt) ** 2).sum().item()
                ss_tot = ((omega_gt - omega_gt.mean()) ** 2).sum().item()
                r2_omega = 1 - ss_res / max(ss_tot, 1e-8)

                T = recon.shape[0]
                per_t_mse = [nn.functional.mse_loss(
                    recon[t], omega_gt[t]).item() for t in range(T)]

                # Cl metrics
                _pred_np = cl_pred[:, 0, 0].cpu().numpy()   # (T,)
                _gt_np   = cl_gt[:, 0, 0].cpu().numpy()     # (T,)

                _cl_metrics = compute_cl_metrics(_pred_np, _gt_np)

                g_val = 0.0
                if angle in rd_params and case_id in rd_params[angle]:
                    g_val = rd_params[angle][case_id]['G']

                results.append({
                    'sample_idx': idx,
                    'angle': angle,
                    'case_id': case_id,
                    'case_tag': tag,
                    'case_title': title,
                    'gust': g_val,
                    **_cl_metrics,
                    'pred_cl': _pred_np,
                    'target_cl': _gt_np,
                    'latent': z_seq[:, 0].cpu().numpy(),
                    'mse_omega': mse_omega,
                    'rel_error_omega': rel_error_omega,
                    'r2_omega': r2_omega,
                    'per_timestep_mse': np.array(per_t_mse),
                    'pred_omega': recon[:, 0, 0].cpu().numpy(),
                    'target_omega': omega_gt[:, 0, 0].cpu().numpy(),
                })

                del x_input, cl_gt, z_seq, cl_pred, recon, omega_gt

                print(f"  {tag:30s} | MSE_w: {mse_omega:.6f} | "
                      f"R2_w: {r2_omega:.4f} | MSE_Cl: {_cl_metrics['mse_cl']:.6f} | "
                      f"RMSE_Cl: {_cl_metrics['rmse_cl']:.4f}")

    # Summary
    avg = lambda key: np.nanmean([r[key] for r in results])
    print(f"\n{'='*60}")
    print(f"WINDOW-LEVEL TEST SUMMARY ({len(results)} sequences)")
    print(f"{'='*60}")
    if not decoder_free:
        print(f"  Vorticity  — MSE: {avg('mse_omega'):.6f}  "
              f"RelErr: {avg('rel_error_omega'):.4f}  R2: {avg('r2_omega'):.4f}")
    print(f"  Lift (Cl)  — MSE: {avg('mse_cl'):.6f}  RMSE: {avg('rmse_cl'):.6f}  "
          f"RelL2Err: {avg('rel_l2_error_cl'):.4f}  "
          f"CosineSim: {avg('cosine_similarity_cl'):.4f}  "
          f"PeakErr: {avg('peak_cl_error'):.4f}  "
          f"PeakTiming: {avg('peak_timing_error'):.1f} steps")
    print(f"{'='*60}")

    return results


# ==========================================
# 1. PER-CASE ANIMATIONS (MP4 or GIF)
# ==========================================
def create_sample_animations(grouped_results, save_dir):
    """Create GT vs Reconstruction animation for test cases (~200 frames each)."""
    to_animate = grouped_results
    if MAX_ANIMATION_SAMPLES is not None:
        to_animate = grouped_results[:MAX_ANIMATION_SAMPLES]

    print(f"\n>>> Generating animations for {len(to_animate)} test cases "
          f"(~{to_animate[0]['pred_omega'].shape[0]} frames each)...")

    for r in to_animate:
        tag = r['case_tag']
        title = r['case_title']
        pred = r['pred_omega']
        target = r['target_omega']
        T = pred.shape[0]

        case_dir = os.path.join(save_dir, 'samples', tag)
        os.makedirs(case_dir, exist_ok=True)

        vmax = float(max(abs(target.min()), abs(target.max())))
        aoa = r['angle']

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        im_gt = axes[0].imshow(target[0], origin='lower', cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, aspect='auto',
                                extent=GRID_EXTENT)
        _add_airfoil(axes[0], aoa)
        axes[0].set_title("Ground Truth")
        axes[0].set_xlabel("x/c")
        axes[0].set_ylabel("y/c")
        plt.colorbar(im_gt, ax=axes[0], fraction=0.046, pad=0.04)

        im_pr = axes[1].imshow(pred[0], origin='lower', cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, aspect='auto',
                                extent=GRID_EXTENT)
        _add_airfoil(axes[1], aoa)
        axes[1].set_title("Reconstruction")
        axes[1].set_xlabel("x/c")
        axes[1].set_ylabel("y/c")
        plt.colorbar(im_pr, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        def update(frame, im_gt=im_gt, im_pr=im_pr, target=target,
                   pred=pred, title=title, r=r, T=T):
            im_gt.set_data(target[frame])
            im_pr.set_data(pred[frame])
            fig.suptitle(f"{title} | Frame {frame+1}/{T} | "
                         f"MSE_w={r['mse_omega']:.5f} | R2_w={r['r2_omega']:.4f}",
                         fontsize=12)
            return [im_gt, im_pr]

        ani = FuncAnimation(fig, update, frames=T, interval=150, blit=False)

        # Try mp4 first (ffmpeg), fall back to gif
        if HAS_FFMPEG:
            try:
                mp4_path = os.path.join(case_dir, f'{tag}.mp4')
                ani.save(mp4_path, writer=FFMpegWriter(fps=8))
                plt.close(fig)
                print(f"  Saved: {mp4_path}")
                continue
            except Exception:
                pass

        gif_path = os.path.join(case_dir, f'{tag}.gif')
        ani.save(gif_path, writer=PillowWriter(fps=8))
        plt.close(fig)
        print(f"  Saved: {gif_path}")


# ==========================================
# 2. PER-CASE Cl EVOLUTION
# ==========================================
def create_cl_evolution_per_case(grouped_results, save_dir, rd_params=None):
    """
    Plot full Cl(t) evolution (all timesteps) for every test case.

    Each plot shows the complete ground-truth and predicted Cl trajectory
    for the entire RD case duration, not a windowed subset.

    ``rd_params`` (nested dict rd_params[angle][case_id] = {'G','R','y'}) is
    used to build the LaTeX parameter subtitle; when absent, only the gust is
    shown.
    """
    print(f"\n>>> Generating Cl evolution plots for {len(grouped_results)} test cases...")

    # Font sizes — enlarged so the figures stay legible when embedded in the
    # thesis report at reduced size.
    TITLE_FS = 24
    SUB_FS   = 18
    LABEL_FS = 19
    TICK_FS  = 15
    LEG_FS   = 15

    rd_params = rd_params or {}

    for r in grouped_results:
        tag = r['case_tag']
        pred_cl   = r['pred_cl']
        target_cl = r['target_cl']
        T = len(pred_cl)
        gust = r.get('gust', 0.0)
        W = int(r.get('warmup_steps', 0))

        # Identifying parameters for the subtitle. Fall back to the stored gust
        # when rd_params has no entry for this case.
        angle = r.get('angle')
        case_id = r.get('case_id')
        p = rd_params.get(angle, {}).get(case_id, {})
        G_val = p.get('G', gust)
        R_val = p.get('R', None)
        y_val = p.get('y', None)
        subtitle = rf"$\alpha = {angle}$,  $G = {G_val:.2f}$"
        if R_val is not None:
            subtitle += rf",  $R = {R_val:.2f}$"
        if y_val is not None:
            subtitle += rf",  $y = {y_val:.2f}$"

        case_dir = os.path.join(save_dir, 'samples', tag)
        os.makedirs(case_dir, exist_ok=True)

        # Error envelope for visual context
        abs_err = np.abs(pred_cl - target_cl)

        fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                                 gridspec_kw={'height_ratios': [3, 1]},
                                 sharex=True)

        # --- Top panel: Cl trajectories ---
        ax = axes[0]
        timesteps = np.arange(T)
        if W > 0 and W < T:
            # Split each line at the warmup boundary. The +1 / W slicing
            # overlaps at index W so the two segments meet without a visible gap.
            t_warm, t_main = timesteps[:W + 1], timesteps[W:]
            gt_warm,  gt_main  = target_cl[:W + 1], target_cl[W:]
            pr_warm,  pr_main  = pred_cl[:W + 1],   pred_cl[W:]
            # Warmup segment: both lines dashed and faded — visually tentative.
            ax.plot(t_warm, gt_warm, color='royalblue', linewidth=1.2,
                    linestyle='--', alpha=0.45)
            ax.plot(t_warm, pr_warm, color='crimson',   linewidth=1.0,
                    linestyle=':',  alpha=0.45)
            # Post-warmup segment: regular styling.
            ax.plot(t_main, gt_main, color='royalblue', linewidth=1.5,
                    label='Ground Truth')
            ax.plot(t_main, pr_main, color='crimson',   linewidth=1.2,
                    linestyle='--', label='Reconstructed', alpha=0.9)
            # Error envelope only on the supervised region.
            ax.fill_between(t_main, gt_main, pr_main,
                            alpha=0.12, color='crimson', label='Error region')
            # Shade the warm-up region instead of drawing a delimiter line.
            ax.axvspan(0, W, color='gray', alpha=0.12, zorder=0, label='Warm-up')
        else:
            ax.plot(timesteps, target_cl, color='royalblue', linewidth=1.5,
                    label='Ground Truth')
            ax.plot(timesteps, pred_cl, color='crimson', linewidth=1.2,
                    linestyle='--', label='Reconstructed', alpha=0.9)
            ax.fill_between(timesteps, target_cl, pred_cl,
                            alpha=0.12, color='crimson', label='Error region')
        ax.set_ylabel(r"$C_l$", fontsize=LABEL_FS)
        # Figure title + parameter subtitle (subtitle on the top axis so it sits
        # just under the suptitle).
        fig.suptitle(r"$C_l$ Evolution", fontsize=TITLE_FS, fontweight='bold')
        ax.set_title(subtitle, fontsize=SUB_FS)
        ax.tick_params(axis='both', labelsize=TICK_FS)
        ax.legend(loc='upper right', fontsize=LEG_FS)
        ax.grid(True, alpha=0.25)

        # --- Bottom panel: absolute error ---
        ax2 = axes[1]
        if W > 0 and W < T:
            # Faded fill over the warmup region, full opacity afterwards.
            ax2.fill_between(timesteps[:W + 1], 0, abs_err[:W + 1],
                             color='darkorange', alpha=0.2)
            ax2.fill_between(timesteps[W:], 0, abs_err[W:],
                             color='darkorange', alpha=0.6)
            ax2.axvspan(0, W, color='gray', alpha=0.12, zorder=0)
        else:
            ax2.fill_between(timesteps, 0, abs_err, color='darkorange', alpha=0.6)
        ax2.set_ylabel("|Abs. Error|", fontsize=LABEL_FS)
        ax2.set_xlabel("time step", fontsize=LABEL_FS)
        ax2.tick_params(axis='both', labelsize=TICK_FS)
        ax2.grid(True, alpha=0.25)

        fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave headroom for the suptitle
        fig_path = os.path.join(case_dir, f'{tag}_cl_evolution.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved {len(grouped_results)} per-case Cl evolution plots")


# ==========================================
# 3. Cl EVOLUTION — ALL CASES OVERLAY
# ==========================================
def create_cl_evolution_all(grouped_results, save_dir, out_subdir='summary'):
    """
    Grid of Cl(t) subplots — one panel per test case, full trajectory.

    Shows all test cases in a single figure, sorted by gust intensity.
    Each panel has ground truth (blue) and prediction (red dashed).

    ``out_subdir`` selects the destination folder under ``save_dir`` (default
    'summary'); the early-window warmup analysis passes 'warmup_<N>'.
    """
    print("\n>>> Generating Cl evolution summary grid...")

    n = len(grouped_results)
    # Sort by |gust| descending so hardest cases appear first
    sorted_results = sorted(grouped_results,
                            key=lambda r: abs(r.get('gust', 0.0)), reverse=True)

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 3 * nrows),
                             squeeze=False)

    for i, r in enumerate(sorted_results):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        target_cl = r['target_cl']
        pred_cl   = r['pred_cl']
        T = len(target_cl)
        t = np.arange(T)
        gust = r.get('gust', 0.0)
        W = int(r.get('warmup_steps', 0))
        if W > 0 and W < T:
            # Burn-in region [0:W] — fed to the model but excluded from metrics.
            # Drawn in a distinct colour (orange) so it is visually separated
            # from the evaluated region [W:] (blue GT / red prediction).
            t_warm, t_main = t[:W + 1], t[W:]
            ax.plot(t_warm, target_cl[:W + 1], color='darkorange', linewidth=1.4,
                    alpha=0.9)
            ax.plot(t_warm, pred_cl[:W + 1],   color='darkorange', linewidth=1.1,
                    linestyle='--', alpha=0.9)
            ax.plot(t_main, target_cl[W:], color='royalblue', linewidth=1.2,
                    label='GT')
            ax.plot(t_main, pred_cl[W:],   color='crimson',   linewidth=1.0,
                    linestyle='--', label='Pred', alpha=0.85)
            ax.axvline(W, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        else:
            ax.plot(t, target_cl, color='royalblue', linewidth=1.2, label='GT')
            ax.plot(t, pred_cl,   color='crimson',   linewidth=1.0,
                    linestyle='--', label='Pred', alpha=0.85)
        ax.set_title(f"{r['case_tag']}\nG={gust:+.1f}  MSE={r['mse_cl']:.4f}",
                     fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        if row == nrows - 1:
            ax.set_xlabel("Timestep", fontsize=8)
        if col == 0:
            ax.set_ylabel("Cl", fontsize=8)

    # Hide empty subplots
    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)

    # Single shared legend
    handles = [
        plt.Line2D([0], [0], color='royalblue', linewidth=1.5, label='Ground Truth'),
        plt.Line2D([0], [0], color='crimson', linewidth=1.2,
                   linestyle='--', label='Reconstructed'),
    ]
    if any(int(r.get('warmup_steps', 0)) > 0 for r in sorted_results):
        handles.append(plt.Line2D([0], [0], color='darkorange', linewidth=1.4,
                                  label='Burn-in (excluded from metrics)'))
    fig.legend(handles=handles, loc='lower right', fontsize=9, ncol=len(handles))

    fig.suptitle(f"Cl Evolution — All {n} Test Cases  (sorted by |G| descending)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.join(save_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'cl_evolution_all.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved Cl evolution summary grid")


# ==========================================
# 3b. Cl EVOLUTION WITH WARMUP PASS — ALL CASES
# ==========================================
def create_cl_evolution_all_warmup(grouped_results, save_dir, cases_per_fig=12):
    """
    Grid of Cl(t) subplots showing the FULL warmup-pass trajectory for every
    test case: [warmup pass | cubic-spline bridge | scored pass]. Each region
    is separated by a vertical black line so the user can tell at a glance
    where the KPI window starts. Cases are sorted by |G| descending and split
    into figures of `cases_per_fig` cases each — for 24 test cases this gives
    `cl_evolution_all_warmup_1.png` and `cl_evolution_all_warmup_2.png`.

    Skipped silently when the warmup-pass arrays are missing (e.g.
    USE_WARMUP_PASS=False, or non-interpolated inference path).
    """
    cases_with_warmup = [
        r for r in grouped_results if r.get('pred_cl_warmup') is not None
    ]
    if not cases_with_warmup:
        return

    print("\n>>> Generating Cl evolution warmup-pass grids...")

    sorted_results = sorted(
        cases_with_warmup,
        key=lambda r: abs(r.get('gust', 0.0)),
        reverse=True,
    )
    n = len(sorted_results)
    n_figs = (n + cases_per_fig - 1) // cases_per_fig

    for fig_idx in range(n_figs):
        start = fig_idx * cases_per_fig
        end   = min(start + cases_per_fig, n)
        sub   = sorted_results[start:end]

        ncols = 3
        nrows = (len(sub) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 3 * nrows),
                                 squeeze=False)

        for i, r in enumerate(sub):
            row, col = divmod(i, ncols)
            ax = axes[row][col]

            pred_w   = r['pred_cl_warmup']
            pred_i   = r['pred_cl_interp']
            pred_m   = r['pred_cl']
            target_w = r['target_cl_warmup']
            target_i = r['target_cl_interp']
            target_m = r['target_cl']

            Lw, Li, Lm = len(pred_w), len(pred_i), len(pred_m)
            t = np.arange(Lw + Li + Lm)
            warm_end   = Lw
            bridge_end = Lw + Li

            pred_full   = np.concatenate([pred_w, pred_i, pred_m])
            target_full = np.concatenate([target_w, target_i, target_m])

            ax.plot(t, target_full, color='royalblue', linewidth=1.0, label='GT')
            ax.plot(t, pred_full,   color='crimson',   linewidth=0.9,
                    linestyle='--', alpha=0.85, label='Pred')

            # Region separators — solid black so the boundaries are obvious.
            ax.axvline(warm_end,   color='black', linewidth=1.0, alpha=0.8)
            ax.axvline(bridge_end, color='black', linewidth=1.0, alpha=0.8)

            gust = r.get('gust', 0.0)
            ax.set_title(f"{r['case_tag']}\nG={gust:+.1f}  MSE={r['mse_cl']:.4f}",
                         fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)
            if row == nrows - 1:
                ax.set_xlabel("Timestep  (warmup | bridge | KPI)", fontsize=8)
            if col == 0:
                ax.set_ylabel("Cl", fontsize=8)

        # Hide empty subplots
        for j in range(len(sub), nrows * ncols):
            row, col = divmod(j, ncols)
            axes[row][col].set_visible(False)

        handles = [
            plt.Line2D([0], [0], color='royalblue', linewidth=1.5,
                       label='Ground Truth'),
            plt.Line2D([0], [0], color='crimson', linewidth=1.2,
                       linestyle='--', label='Reconstructed'),
            plt.Line2D([0], [0], color='black', linewidth=1.0,
                       label='Region boundary'),
        ]
        fig.legend(handles=handles, loc='lower right', fontsize=9,
                   ncol=len(handles))

        fig.suptitle(
            f"Cl Evolution with Warmup Pass — Cases {start + 1}-{end} of {n}  "
            f"(sorted by |G| descending)",
            fontsize=12, fontweight='bold',
        )
        plt.tight_layout()

        suffix = f"_{fig_idx + 1}" if n_figs > 1 else ""
        out_path = os.path.join(
            save_dir, 'summary', f'cl_evolution_all_warmup{suffix}.png',
        )
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")


# ==========================================
# 4. ACCURACY METRICS (CSV + BAR CHART)
# ==========================================
def write_accuracy_metrics_csv(results, csv_path):
    """Write the per-encounter accuracy_metrics.csv (one row per case + AVERAGE).

    Self-contained: detects vorticity/extra metrics from the results and writes
    the same schema used everywhere else. Shared by the full-trajectory summary
    and the early-window warmup analysis.
    """
    has_omega = not np.isnan(results[0].get('mse_omega', float('nan')))

    extra_keys, extra_labels = [], []
    for key, label in [('mean_ssim', 'SSIM'), ('mean_corr', 'Correlation')]:
        if key in results[0]:
            extra_keys.append(key)
            extra_labels.append(label)

    _cl_cols = [
        'MSE_Cl', 'RMSE_Cl', 'MAE_Cl', 'RelL2Err_Cl',
        'CosineSim_Cl',
        'Peak_Cl_Error', 'Peak_Cl_Error_Signed', 'Peak_Timing_Error',
        'Integrated_Abs_Error', 'Integrated_Sq_Error',
        'Normalized_Integrated_Abs_Error',
    ]
    _cl_keys = [
        'mse_cl', 'rmse_cl', 'mae_cl', 'rel_l2_error_cl',
        'cosine_similarity_cl',
        'peak_cl_error', 'peak_cl_error_signed', 'peak_timing_error',
        'integrated_abs_error', 'integrated_sq_error',
        'normalized_integrated_abs_error',
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if has_omega:
            header = (['Case', 'AoA', 'Case_ID', 'Gust',
                       'MSE_omega', 'RelErr_omega', 'R2_omega']
                      + _cl_cols + extra_labels)
        else:
            header = ['Case', 'AoA', 'Case_ID', 'Gust'] + _cl_cols + extra_labels
        writer.writerow(header)

        for r in results:
            row = [r['case_tag'], r['angle'], r['case_id'],
                   f"{r.get('gust', 0.0):.2f}"]
            if has_omega:
                row += [f"{r['mse_omega']:.6f}", f"{r['rel_error_omega']:.6f}",
                        f"{r['r2_omega']:.6f}"]
            for k in _cl_keys:
                v = r[k]
                row.append(str(int(v)) if k == 'peak_timing_error' else f"{v:.6f}")
            for key in extra_keys:
                row.append(f"{r[key]:.6f}")
            writer.writerow(row)

        avg_row = ['AVERAGE', '-', '-', '-']
        if has_omega:
            for key in ['mse_omega', 'rel_error_omega', 'r2_omega']:
                avg_row.append(f"{np.mean([r[key] for r in results]):.6f}")
        for k in _cl_keys:
            avg_row.append(f"{np.nanmean([r[k] for r in results]):.6f}")
        for key in extra_keys:
            avg_row.append(f"{np.mean([r[key] for r in results]):.6f}")
        writer.writerow(avg_row)

    print(f"  Saved: {csv_path}")


def plot_accuracy_strip_chart(results, out_path, has_omega=False):
    """Draw the 2x3 encounter-level Cl metric strip chart to ``out_path``.

    Each panel shows the per-encounter values of one metric (jittered points)
    with the average marked as a red diamond. ``results`` is a list of per-case
    dicts carrying the metric keys ('rmse_cl', 'cosine_similarity_cl', ...).
    """
    if has_omega:
        strip_metrics = [
            ('mse_omega',       'MSE (vorticity)',            'steelblue'),
            ('rel_error_omega', 'Relative Error (vorticity)', 'darkorange'),
            ('r2_omega',        'R2 (vorticity)',             'forestgreen'),
            ('cosine_similarity_cl', 'Cosine Similarity (Cl)', 'mediumorchid'),
            ('rmse_cl',         'RMSE (Cl)',                  'crimson'),
            ('rel_l2_error_cl', 'Relative L2 Error (Cl)',     'darkorange'),
        ]
    else:
        strip_metrics = [
            ('cosine_similarity_cl',            'Cosine Similarity (Cl)',         'mediumorchid'),
            ('rmse_cl',                         'RMSE (Cl)',                      'crimson'),
            ('rel_l2_error_cl',                 'Relative L2 Error (Cl)',         'darkorange'),
            ('peak_cl_error',                   'Peak Cl Error',                  'steelblue'),
            ('peak_timing_error',               'Peak Timing Error (steps)',      'teal'),
            ('normalized_integrated_abs_error', 'Normalized Integrated Abs Error','saddlebrown'),
        ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes_flat = axes.flatten()

    for ax, (key, label, color) in zip(axes_flat, strip_metrics):
        values = np.array([r[key] for r in results], dtype=float)
        avg_val = float(np.nanmean(values))

        y_jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(values))
        ax.scatter(values, y_jitter, color=color, s=50, alpha=0.7, zorder=3,
                   edgecolors='white', linewidth=0.5)
        ax.scatter([avg_val], [0], color='red', s=120, zorder=5, marker='D',
                   edgecolors='black', linewidth=1.0)
        ax.axvline(x=avg_val, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.set_title(f"{label}  (avg = {avg_val:.4f})")
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_accuracy_metrics(grouped_results, save_dir):
    """Save per-case encounter-level accuracy metrics as CSV, strip chart, and breakdown charts."""
    results = grouped_results  # one entry per gust encounter
    print("\n>>> Computing accuracy metrics...")

    # Check if vorticity metrics are available (v5+ decoder-free models set them to NaN)
    has_omega = not np.isnan(results[0].get('mse_omega', float('nan')))

    # CSV — encounter-level
    write_accuracy_metrics_csv(results, os.path.join(save_dir, 'accuracy_metrics.csv'))

    # ------------------------------------------------------------------
    # 1D strip chart — encounter-level Cl metrics
    # ------------------------------------------------------------------
    plot_accuracy_strip_chart(
        results,
        os.path.join(save_dir, 'summary', 'accuracy_strip_chart.png'),
        has_omega=has_omega)
    print("  Saved accuracy strip chart")

    # ------------------------------------------------------------------
    # Breakdown figures — results is already encounter-level. Each thesis
    # metric (MAE, RMSE, Relative L2 Error, Cosine Similarity) is broken down
    # by gust intensity and by angle of attack in its own figure.
    # ------------------------------------------------------------------
    case_list = [{'angle': r['angle'], 'gust': r.get('gust', 0.0)} for r in results]

    # --- By Gust Intensity: scatter (|G| vs metric) + signed box plots ---
    # For every thesis metric two figures are produced:
    #   1. breakdown_by_gust_<slug>.png        — scatter of the metric against the
    #      continuous |G|, points coloured by gust sign (red = negative rotation,
    #      blue = positive incl. baseline), faint mild/medium/strong bands, and a
    #      smooth qualitative trend.
    #   2. breakdown_by_gust_signed_<slug>.png — horizontal box plot of the metric
    #      split into positive (incl. baseline G=0) vs negative gusts, directly
    #      contrasting the two vortex-rotation senses.
    _BOX_FACE   = "#cfe0f0"    # light blue box fill
    _BOX_EDGE   = "#1f5fa6"    # blue box / whisker / median edge
    _MEAN_COLOR = "#2e8b57"    # green mean marker
    _NEG_COLOR  = "#c0392b"    # red  — negative gust (one rotation sense)
    _POS_COLOR  = "#1f5fa6"    # blue — positive gust (incl. baseline)

    # (result key, figure title, metric-axis label, filename slug).
    _gust_box_metrics = [
        ('mae_cl',               'MAE',                  'MAE',                   'mae'),
        ('rmse_cl',              'RMSE',                 'RMSE',                  'rmse'),
        ('rel_l2_error_cl',      'Relative $L_2$ Error', r'$\varepsilon_{L_2}$',  'rell2'),
        ('cosine_similarity_cl', 'Cosine Similarity',    'Cosine Similarity',     'cossim'),
    ]

    gust_signed = np.array([r.get('gust', 0.0) for r in results], dtype=float)
    gust_abs    = np.abs(gust_signed)
    angle_all   = np.array([r['angle'] for r in results])
    g_max       = float(gust_abs.max()) if len(gust_abs) else 1.0
    x_hi        = max(2.5, g_max) * 1.02

    # Marker shape encodes AoA; fill colour encodes gust sign.
    aoa_unique   = sorted(set(angle_all.tolist()))
    _marker_pool = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'p', 'h']
    aoa_marker   = {a: _marker_pool[i % len(_marker_pool)]
                    for i, a in enumerate(aoa_unique)}

    # Mild/medium/strong bands (edges match the gust-intensity thresholds).
    _bands = [(0.0, 1.0, '#4CAF50', r'Mild ($|G|\leq1$)'),
              (1.0, 2.5, '#FF9800', r'Medium ($1<|G|\leq2.5$)'),
              (2.5, x_hi, '#F44336', r'Strong ($|G|>2.5$)')]

    for key, title, metric_label, slug in _gust_box_metrics:
        vals   = np.array([r[key] for r in results], dtype=float)
        finite = np.isfinite(vals) & np.isfinite(gust_abs)
        x   = gust_abs[finite]
        y   = vals[finite]
        sg  = gust_signed[finite]
        ang = angle_all[finite]

        # ---- Figure 1: scatter of the metric vs continuous |G| ----
        # marker shape = AoA, fill colour = gust sign, x = |G|.
        fig, ax = plt.subplots(figsize=(9.5, 6.0))
        for lo, hi, col, _lab in _bands:
            ax.axvspan(lo, hi, color=col, alpha=0.08, zorder=0)
        for a in aoa_unique:
            m = aoa_marker[a]
            sel_pos = (ang == a) & (sg >= 0)
            sel_neg = (ang == a) & (sg < 0)
            ax.scatter(x[sel_pos], y[sel_pos], s=70, marker=m,
                       facecolor=_POS_COLOR, edgecolors='black', linewidths=0.4,
                       alpha=0.8, zorder=3)
            ax.scatter(x[sel_neg], y[sel_neg], s=70, marker=m,
                       facecolor=_NEG_COLOR, edgecolors='black', linewidths=0.4,
                       alpha=0.8, zorder=3)
        ax.set_xlim(0, x_hi)
        ax.set_xlabel(r'Gust intensity, $|G|$', fontsize=18)
        ax.set_ylabel(metric_label, fontsize=18)
        ax.set_title(f'{title} vs Gust Intensity', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Two legends below the axes: gust sign (left), and the AoA marker key
        # (right). The shaded intensity bands are left unlabelled — the mild /
        # medium / strong regions read clearly from the colours alone.
        sign_handles = [
            Line2D([], [], marker='o', linestyle='none', markersize=11,
                   markerfacecolor=_POS_COLOR, markeredgecolor='black',
                   label=r'Positive gust ($G\geq0$)'),
            Line2D([], [], marker='o', linestyle='none', markersize=11,
                   markerfacecolor=_NEG_COLOR, markeredgecolor='black',
                   label=r'Negative gust ($G<0$)'),
        ]
        aoa_handles = [
            Line2D([], [], marker=aoa_marker[a], linestyle='none', markersize=11,
                   markerfacecolor='gray', markeredgecolor='black', label=f'{a}°')
            for a in aoa_unique
        ]
        leg1 = ax.legend(handles=sign_handles, loc='upper center',
                         bbox_to_anchor=(0.27, -0.13), frameon=False, fontsize=13,
                         title='Gust sign', title_fontsize=14)
        ax.add_artist(leg1)
        ax.legend(handles=aoa_handles, loc='upper center',
                  bbox_to_anchor=(0.78, -0.13), ncol=2, frameon=False, fontsize=13,
                  title=r'Angle of attack, $\alpha$', title_fontsize=14)

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'summary', f'breakdown_by_gust_{slug}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ---- Figure 2: signed box plot (positive incl. baseline vs negative) ----
        neg_vals = y[sg < 0]
        pos_vals = y[sg >= 0]          # baseline (G=0) counted as positive
        data = [neg_vals, pos_vals]    # position 1 = negative (bottom), 2 = positive (top)

        fig, ax = plt.subplots(figsize=(8.0, 3.4))
        ax.boxplot(
            data, positions=[1, 2], vert=False, widths=0.55,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker='D', markerfacecolor=_MEAN_COLOR,
                           markeredgecolor=_MEAN_COLOR, markersize=6),
            medianprops=dict(color=_BOX_EDGE, linewidth=1.8),
            boxprops=dict(facecolor=_BOX_FACE, edgecolor=_BOX_EDGE, linewidth=1.4),
            whiskerprops=dict(color=_BOX_EDGE, linewidth=1.4),
            capprops=dict(color=_BOX_EDGE, linewidth=1.4),
            flierprops=dict(marker='', markersize=0),  # outliers shown via scatter
        )
        rng = np.random.default_rng(0)
        for pos_i, vv, col in zip([1, 2], data, [_NEG_COLOR, _POS_COLOR]):
            if len(vv) == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(vv))
            ax.scatter(vv, pos_i + jitter, s=16, color=col, alpha=0.5,
                       zorder=3, edgecolors='none')
        ax.set_yticks([1, 2])
        ax.set_yticklabels([f'Negative\n($G<0$, n={len(neg_vals)})',
                            f'Positive\n($G\\geq0$, n={len(pos_vals)})'],
                           fontsize=14)
        ax.set_xlabel(metric_label, fontsize=18)
        ax.set_title(f'{title} by Gust Sign', fontsize=20, fontweight='bold')
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'summary',
                                 f'breakdown_by_gust_signed_{slug}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved breakdown by gust intensity "
          f"({len(_gust_box_metrics)} scatter + {len(_gust_box_metrics)} signed box plots)")

    # --- By Angle of Attack: four horizontal box plots (one per metric) ---
    # AoA on the y-axis, metric value on the x-axis. Each box shows the metric
    # distribution across all test cases at that angle. Styling matches
    # box_plots.py (blue box, green mean diamond, red jittered per-case points).
    aoa_order = sorted(set(c['angle'] for c in case_list))

    _BOX_FACE    = "#cfe0f0"   # light blue box fill
    _BOX_EDGE    = "#1f5fa6"   # blue box / whisker / median edge
    _POINT_COLOR = "#c0392b"   # red per-case markers
    _MEAN_COLOR  = "#2e8b57"   # green mean marker

    # (result key, figure title, x-axis label, filename slug) — the four thesis
    # metrics: MAE, RMSE, Relative L2 Error, Cosine Similarity.
    _aoa_box_metrics = [
        ('mae_cl',               'MAE',                  'MAE',                    'mae'),
        ('rmse_cl',              'RMSE',                 'RMSE',                   'rmse'),
        ('rel_l2_error_cl',      'Relative $L_2$ Error', r'$\varepsilon_{L_2}$',   'rell2'),
        ('cosine_similarity_cl', 'Cosine Similarity',    'Cosine Similarity',      'cossim'),
    ]

    # Collect per-case values grouped by AoA for each metric.
    aoa_box = {a: {} for a in aoa_order}
    for a in aoa_order:
        cases_a = [r for r in results if r['angle'] == a]
        for key, *_ in _aoa_box_metrics:
            vals = np.array([r[key] for r in cases_a], dtype=float)
            aoa_box[a][key] = vals[np.isfinite(vals)]

    positions = np.arange(1, len(aoa_order) + 1)

    for key, title, axis_label, slug in _aoa_box_metrics:
        rng = np.random.default_rng(0)  # reproducible jitter, per figure
        data = [aoa_box[a][key] for a in aoa_order]

        fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.7 * len(aoa_order) + 2.0)))
        ax.boxplot(
            data, positions=positions, vert=False, widths=0.55,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker='D', markerfacecolor=_MEAN_COLOR,
                           markeredgecolor=_MEAN_COLOR, markersize=6),
            medianprops=dict(color=_BOX_EDGE, linewidth=1.8),
            boxprops=dict(facecolor=_BOX_FACE, edgecolor=_BOX_EDGE, linewidth=1.4),
            whiskerprops=dict(color=_BOX_EDGE, linewidth=1.4),
            capprops=dict(color=_BOX_EDGE, linewidth=1.4),
            flierprops=dict(marker='', markersize=0),  # outliers shown via scatter
        )

        # Overlay individual test cases as jittered points.
        for pos, vals in zip(positions, data):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(vals, pos + jitter, s=16, color=_POINT_COLOR,
                       alpha=0.45, zorder=3, edgecolors='none')

        ax.set_yticks(positions)
        ax.set_yticklabels([f"{a}" for a in aoa_order], fontsize=15)
        ax.invert_yaxis()  # smallest AoA at the top
        ax.set_xlabel(axis_label, fontsize=18)
        ax.set_ylabel(r"Angle of attack, $\alpha$ [°]", fontsize=18)
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.grid(axis='x', linestyle=':', alpha=0.5)

        handles = [
            plt.Line2D([], [], color=_BOX_EDGE, linewidth=1.8, label='Median'),
            plt.Line2D([], [], marker='D', linestyle='none',
                       markerfacecolor=_MEAN_COLOR, markeredgecolor=_MEAN_COLOR,
                       markersize=6, label='Mean'),
            plt.Line2D([], [], marker='o', linestyle='none',
                       markerfacecolor=_POINT_COLOR, markeredgecolor='none',
                       markersize=6, alpha=0.55, label='Test case'),
        ]
        ax.legend(handles=handles, frameon=False, ncol=3,
                  loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  columnspacing=2.5, handletextpad=0.6)

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'summary', f'breakdown_by_aoa_{slug}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved breakdown by angle of attack ({len(_aoa_box_metrics)} box plots)")


# ==========================================
# 4b. CASE SUMMARY TXT
# ==========================================
def create_case_summary_txt(grouped_results, save_dir, out_subdir='summary'):
    """
    Write encounter-level Cl metrics per case to a text file.

    Metrics are computed from the full concatenated pred_cl/target_cl trajectories
    (encounter-level, not per-window averages).  Output includes averages,
    highlights (best/worst per metric), a per-case table sorted by |G| descending,
    and a gust-bucket breakdown.

    ``out_subdir`` selects the destination folder under ``save_dir`` (default
    'summary'); the early-window warmup analysis passes 'warmup_<N>'.
    """
    print("\n>>> Writing case summary text file...")

    _mf = [
        ('mse_cl',                          'MSE'),
        ('rmse_cl',                         'RMSE'),
        ('mae_cl',                          'MAE'),
        ('rel_l2_error_cl',                 'Relative L2 Error'),
        ('cosine_similarity_cl',            'Cosine Similarity'),
        ('peak_cl_error',                   'Peak Cl Error'),
        ('peak_timing_error',               'Peak Timing Error (ts)'),
        ('integrated_abs_error',            'Integrated Abs Error'),
        ('normalized_integrated_abs_error', 'Normalized Integrated Abs Error'),
    ]
    _metric_fields = [f for f, _ in _mf]
    _metric_labels = {f: lbl for f, lbl in _mf}

    rows = []
    rows_orig = []  # parallel list for original-grid metrics; empty if not applicable
    for r in grouped_results:
        # Skip the warmup region — encounter-level metrics should match the
        # supervised region used at training/validation time.
        W      = int(r.get('warmup_steps', 0))
        W_orig = int(r.get('warmup_steps_orig', 0))
        _pred = r['pred_cl'][W:] if W > 0 else r['pred_cl']
        _gt   = r['target_cl'][W:] if W > 0 else r['target_cl']
        m = compute_cl_metrics(_pred, _gt)
        rows.append({
            'tag':  r['case_tag'],
            'aoa':  r.get('angle', '?'),
            'gust': r.get('gust', 0.0),
            'T':    len(_gt),
            **{f: m[f] for f in _metric_fields},
        })
        if r.get('pred_cl_orig') is not None:
            _pred_o = r['pred_cl_orig'][W_orig:] if W_orig > 0 else r['pred_cl_orig']
            _gt_o   = r['target_cl_orig'][W_orig:] if W_orig > 0 else r['target_cl_orig']
            m_o = compute_cl_metrics(_pred_o, _gt_o)
            rows_orig.append({
                'tag':  r['case_tag'],
                'aoa':  r.get('angle', '?'),
                'gust': r.get('gust', 0.0),
                'T':    len(_gt_o),
                **{f: m_o[f] for f in _metric_fields},
            })

    rows.sort(key=lambda x: abs(x['gust']), reverse=True)

    lines = []
    lines.append("=" * 72)
    lines.append("CASE SUMMARY — ENCOUNTER-LEVEL Cl METRICS")
    lines.append("=" * 72)
    lines.append("")

    # --- Averages ---
    lines.append("AVERAGES (encounter-level)")
    lines.append("-" * 40)
    for f in _metric_fields:
        avg = float(np.mean([row[f] for row in rows]))
        lines.append(f"  {_metric_labels[f]:<36s}: {avg:.6f}")
    lines.append("")

    # --- Medians ---
    lines.append("MEDIANS (encounter-level)")
    lines.append("-" * 40)
    for f in _metric_fields:
        sv  = sorted(row[f] for row in rows)
        n   = len(sv)
        med = sv[n // 2] if n % 2 == 1 else (sv[n // 2 - 1] + sv[n // 2]) / 2.0
        lines.append(f"  {_metric_labels[f]:<36s}: {med:.6f}")
    lines.append("")

    # --- Best / Worst per metric ---
    _higher_is_better = {'cosine_similarity_cl', 'centered_cosine_similarity_cl'}

    lines.append("HIGHLIGHTS")
    lines.append("-" * 40)
    for f in _metric_fields:
        vals = [(row[f], row['tag']) for row in rows]
        if f in _higher_is_better:
            best  = max(vals, key=lambda x: x[0])
            worst = min(vals, key=lambda x: x[0])
        else:
            best  = min(vals, key=lambda x: x[0])
            worst = max(vals, key=lambda x: x[0])
        lines.append(f"  {_metric_labels[f]}")
        lines.append(f"    Best : {best[1]:<32s}  {best[0]:.6f}")
        lines.append(f"    Worst: {worst[1]:<32s}  {worst[0]:.6f}")
    lines.append("")

    # --- Per-case table ---
    hdr = (f"{'Case':<30s} {'AoA':>5} {'G':>7} {'T':>6}"
           f" {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'RelL2':>10}"
           f" {'CosSim':>8} {'PkErr':>10} {'PkTim':>7} {'NormIntAbs':>12}")
    sep = "-" * len(hdr)
    lines.append("PER-CASE TABLE  (sorted by |G| descending)")
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)
    for row in rows:
        lines.append(
            f"{row['tag']:<30s} {row['aoa']:>5} {row['gust']:>+7.2f} {row['T']:>6}"
            f" {row['mse_cl']:>10.5f} {row['rmse_cl']:>10.5f} {row['mae_cl']:>10.5f}"
            f" {row['rel_l2_error_cl']:>10.5f} {row['cosine_similarity_cl']:>8.4f}"
            f" {row['peak_cl_error']:>10.5f} {row['peak_timing_error']:>7d}"
            f" {row['normalized_integrated_abs_error']:>12.5f}"
        )
    lines.append(sep)
    lines.append("")

    # --- Gust bucket breakdown ---
    lines.append("GUST BUCKET BREAKDOWN")
    lines.append("-" * 40)
    buckets = [
        ('Mild   (|G|<=1.0)',   lambda g: abs(g) <= 1.0),
        ('Medium (1<|G|<=2.5)', lambda g: 1.0 < abs(g) <= 2.5),
        ('Strong (|G|>2.5)',    lambda g: abs(g) > 2.5),
    ]
    for bucket_name, bucket_fn in buckets:
        bucket_rows = [r for r in rows if bucket_fn(r['gust'])]
        if not bucket_rows:
            continue
        lines.append(f"  {bucket_name}  (n={len(bucket_rows)})")
        for f in _metric_fields:
            avg = float(np.mean([r[f] for r in bucket_rows]))
            sv  = sorted(r[f] for r in bucket_rows)
            nb  = len(sv)
            med = sv[nb // 2] if nb % 2 == 1 else (sv[nb // 2 - 1] + sv[nb // 2]) / 2.0
            lines.append(f"    {_metric_labels[f]:<36s}: mean={avg:.6f}  median={med:.6f}")
        lines.append("")

    # ---- ORIGINAL GRID section (only when original-grid arrays were propagated) ----
    if rows_orig:
        rows_orig.sort(key=lambda x: abs(x['gust']), reverse=True)
        lines.append("=" * 72)
        lines.append("ORIGINAL GRID — ENCOUNTER-LEVEL Cl METRICS  (pred[::step] vs original CSV)")
        lines.append(f"  T ≈ {rows_orig[0]['T']}  (downsampled from interpolated grid)")
        lines.append("=" * 72)
        lines.append("")

        lines.append("AVERAGES (original grid)")
        lines.append("-" * 40)
        for f in _metric_fields:
            avg_o = float(np.mean([row[f] for row in rows_orig]))
            lines.append(f"  {_metric_labels[f]:<36s}: {avg_o:.6f}")
        lines.append("")

        lines.append("MEDIANS (original grid)")
        lines.append("-" * 40)
        for f in _metric_fields:
            sv  = sorted(row[f] for row in rows_orig)
            n   = len(sv)
            med = sv[n // 2] if n % 2 == 1 else (sv[n // 2 - 1] + sv[n // 2]) / 2.0
            lines.append(f"  {_metric_labels[f]:<36s}: {med:.6f}")
        lines.append("")

        lines.append("HIGHLIGHTS (original grid)")
        lines.append("-" * 40)
        for f in _metric_fields:
            vals = [(row[f], row['tag']) for row in rows_orig]
            if f in _higher_is_better:
                best  = max(vals, key=lambda x: x[0])
                worst = min(vals, key=lambda x: x[0])
            else:
                best  = min(vals, key=lambda x: x[0])
                worst = max(vals, key=lambda x: x[0])
            lines.append(f"  {_metric_labels[f]}")
            lines.append(f"    Best : {best[1]:<32s}  {best[0]:.6f}")
            lines.append(f"    Worst: {worst[1]:<32s}  {worst[0]:.6f}")
        lines.append("")

        lines.append("PER-CASE TABLE (original grid, sorted by |G| descending)")
        lines.append(sep)
        lines.append(hdr)
        lines.append(sep)
        for row in rows_orig:
            lines.append(
                f"{row['tag']:<30s} {row['aoa']:>5} {row['gust']:>+7.2f} {row['T']:>6}"
                f" {row['mse_cl']:>10.5f} {row['rmse_cl']:>10.5f} {row['mae_cl']:>10.5f}"
                f" {row['rel_l2_error_cl']:>10.5f} {row['cosine_similarity_cl']:>8.4f}"
                f" {row['peak_cl_error']:>10.5f} {row['peak_timing_error']:>7d}"
                f" {row['normalized_integrated_abs_error']:>12.5f}"
            )
        lines.append(sep)
        lines.append("")

        lines.append("GUST BUCKET BREAKDOWN (original grid)")
        lines.append("-" * 40)
        for bucket_name, bucket_fn in buckets:
            bucket_rows_o = [r for r in rows_orig if bucket_fn(r['gust'])]
            if not bucket_rows_o:
                continue
            lines.append(f"  {bucket_name}  (n={len(bucket_rows_o)})")
            for f in _metric_fields:
                avg_o = float(np.mean([r[f] for r in bucket_rows_o]))
                sv    = sorted(r[f] for r in bucket_rows_o)
                nb    = len(sv)
                med   = sv[nb // 2] if nb % 2 == 1 else (sv[nb // 2 - 1] + sv[nb // 2]) / 2.0
                lines.append(f"    {_metric_labels[f]:<36s}: mean={avg_o:.6f}  median={med:.6f}")
            lines.append("")

    summary_dir = os.path.join(save_dir, out_subdir)
    os.makedirs(summary_dir, exist_ok=True)
    txt_path = os.path.join(summary_dir, 'case_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')

    print(f"  Saved: {txt_path}")


# ==========================================
# 4b. EARLY-WINDOW (WARMUP EFFECT) METRICS
# ==========================================
def create_warmup_window_analysis(grouped_results, save_dir, n_frames):
    """
    Cold-start (warmup effect) analysis restricted to the first ``n_frames``
    timesteps of each scored trajectory.

    The global encounter-level metrics average over the whole ~6k-frame
    trajectory, so the cold-start region that the warmup pass improves is
    diluted to near-nothing. Re-scoring only the first ``n_frames`` frames
    isolates that region and makes the warmup effect measurable / comparable
    across runs (warmup ON vs OFF).

    Writes into results/<run>/warmup_<n_frames>/:
        accuracy_metrics.csv, accuracy_strip_chart.png, case_summary.txt,
        cl_evolution_all.png
    """
    if n_frames is None or n_frames <= 0:
        return

    subdir = f"warmup_{int(n_frames)}"
    print(f"\n>>> Generating early-window warmup analysis "
          f"(first {int(n_frames)} frames) -> {subdir}/ ...")
    out_dir = os.path.join(save_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Build a windowed copy of the results: truncate the Cl trajectories to the
    # first n_frames and recompute every metric on that window. warmup_steps is
    # forced to 0 so the whole window (including the cold start) is scored, and
    # the original-grid arrays are dropped to keep this file focused.
    windowed = []
    for r in grouped_results:
        pred = np.asarray(r['pred_cl'])
        gt   = np.asarray(r['target_cl'])
        N = min(int(n_frames), len(pred), len(gt))
        if N <= 1:
            continue
        rw = dict(r)
        rw['pred_cl']            = pred[:N]
        rw['target_cl']          = gt[:N]
        rw['warmup_steps']       = 0
        rw['warmup_steps_orig']  = 0
        rw['pred_cl_orig']       = None
        rw['target_cl_orig']     = None
        rw.update(compute_cl_metrics(rw['pred_cl'], rw['target_cl']))
        windowed.append(rw)

    if not windowed:
        print(f"  [WARN] No case has > 1 frame in the first {int(n_frames)} — skipping.")
        return

    has_omega = not np.isnan(windowed[0].get('mse_omega', float('nan')))
    write_accuracy_metrics_csv(windowed, os.path.join(out_dir, 'accuracy_metrics.csv'))
    plot_accuracy_strip_chart(
        windowed, os.path.join(out_dir, 'accuracy_strip_chart.png'),
        has_omega=has_omega)
    create_cl_evolution_all(windowed, save_dir, out_subdir=subdir)
    create_case_summary_txt(windowed, save_dir, out_subdir=subdir)
    print(f"  Saved early-window warmup analysis to {out_dir}")


# ==========================================
# 4c. TRAINING / VALIDATION LOSS CURVES
# ==========================================
def plot_loss_curves(cfg, save_dir):
    """
    Plot train and validation loss curves from the checkpoint's loss_curves.pt.

    The loss_curves.pt file is saved by train_SNN.py at the end of training and
    contains {'train_losses': [...], 'val_losses': [...]}, one value per epoch.
    """
    loss_path = os.path.join(cfg.MODEL_WEIGHTS, 'loss_curves.pt')
    if not os.path.exists(loss_path):
        print(f"\n>>> loss_curves.pt not found in checkpoint — skipping loss plot")
        return

    print("\n>>> Generating loss curves plot...")
    curves = torch.load(loss_path, map_location='cpu', weights_only=True)
    train_losses = curves['train_losses']
    val_losses = curves['val_losses']
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, color='steelblue', linewidth=1.5,
            label='Train loss')
    ax.plot(epochs, val_losses, color='crimson', linewidth=1.5,
            label='Validation loss')

    # Mark best validation epoch
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = min(val_losses)
    ax.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.6,
               label=f'Best val epoch ({best_epoch})')
    ax.scatter([best_epoch], [best_val], color='green', s=80, zorder=5,
              marker='*', edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(f'Training & Validation Loss  (best val = {best_val:.6f} @ epoch {best_epoch})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(train_losses))

    # Log scale if dynamic range is large
    if max(train_losses) / (min(train_losses) + 1e-10) > 20:
        ax.set_yscale('log')

    plt.tight_layout()
    summary_dir = os.path.join(save_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    fig.savefig(os.path.join(summary_dir, 'loss_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved loss curves plot")


# ==========================================
# 5. TEMPORAL ERROR EVOLUTION
# ==========================================
def temporal_error_analysis(grouped_results, save_dir):
    """Plot MSE evolution across timesteps (up to 5 cases + mean)."""
    print("\n>>> Generating temporal error analysis...")

    to_plot = grouped_results[:5]

    # Compute mean across ALL grouped cases (not just first 5)
    min_len = min(len(r['per_timestep_mse']) for r in grouped_results)
    all_per_t_mse = np.stack([r['per_timestep_mse'][:min_len] for r in grouped_results])
    mean_mse = all_per_t_mse.mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, r in enumerate(to_plot):
        timesteps = np.arange(len(r['per_timestep_mse']))
        color = plt.cm.tab10(i)
        ax.plot(timesteps, r['per_timestep_mse'], color=color, linewidth=1.2,
                label=r['case_tag'])

    # Mean across all cases
    ax.plot(np.arange(min_len), mean_mse, 'k--', linewidth=2.0,
            label=f'Mean (all {len(grouped_results)} cases)')

    ax.set_xlabel("Timestep")
    ax.set_ylabel("MSE (vorticity)")
    ax.set_title("Temporal Error Evolution (MSE per Timestep)")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'summary', 'temporal_error_evolution.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved temporal error plot")


# ==========================================
# 6. PER-PIXEL ERROR MAPS
# ==========================================
def per_pixel_error_maps(results, save_dir):
    """Time-averaged spatial MAE map for vorticity reconstruction."""
    print("\n>>> Generating per-pixel error maps...")

    all_errors = []
    for r in results:
        err = np.abs(r['pred_omega'] - r['target_omega'])
        all_errors.append(err.mean(axis=0))

    avg_err = np.mean(all_errors, axis=0)

    unique_aoas = sorted(set(r['angle'] for r in results))

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(avg_err, origin='lower', cmap='hot', aspect='auto',
                   extent=GRID_EXTENT)
    for aoa in unique_aoas:
        _add_airfoil(ax, aoa)
    ax.set_title(f"Per-Pixel MAE (averaged over {len(results)} test cases, "
                 f"all timesteps)")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MAE")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'summary', 'per_pixel_error_maps.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved per-pixel error maps")


# ==========================================
# 7. TEMPORAL CROSS-CORRELATION
# ==========================================
def temporal_cross_correlation(grouped_results, save_dir):
    """Spatial Pearson correlation between pred and GT per timestep."""
    print("\n>>> Generating temporal cross-correlation...")

    # Compute correlation for all grouped cases
    all_corr = []
    for r in grouped_results:
        pred = r['pred_omega']
        target = r['target_omega']
        T = pred.shape[0]

        corr_t = []
        for t in range(T):
            p_flat = pred[t].flatten()
            g_flat = target[t].flatten()
            c = np.corrcoef(p_flat, g_flat)[0, 1]
            corr_t.append(c if np.isfinite(c) else 0.0)

        all_corr.append(np.array(corr_t))
        r['mean_corr'] = float(np.mean(corr_t))

    # Mean across all cases (truncated to shortest)
    min_len = min(len(c) for c in all_corr)
    mean_corr = np.mean([c[:min_len] for c in all_corr], axis=0)

    # Plot up to 5 individual cases + mean
    to_plot = grouped_results[:5]

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, r in enumerate(to_plot):
        timesteps = np.arange(len(all_corr[i]))
        color = plt.cm.tab10(i)
        ax.plot(timesteps, all_corr[i], color=color, linewidth=1.2,
                label=r['case_tag'])

    ax.plot(np.arange(min_len), mean_corr, 'k--', linewidth=2.0,
            label=f'Mean (all {len(grouped_results)} cases)')

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Spatial Pearson Correlation")
    ax.set_title("Temporal Cross-Correlation (Pred vs GT Vorticity)")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'summary', 'temporal_cross_correlation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved temporal cross-correlation plot")


# ==========================================
# 8. SSIM (STRUCTURAL SIMILARITY INDEX)
# ==========================================
def _compute_ssim(img1, img2, win_size=11):
    """Compute SSIM between two 2D arrays using uniform_filter."""
    C1 = (0.01 * (img1.max() - img1.min() + 1e-8)) ** 2
    C2 = (0.03 * (img1.max() - img1.min() + 1e-8)) ** 2

    mu1 = uniform_filter(img1, size=win_size)
    mu2 = uniform_filter(img2, size=win_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = uniform_filter(img1 ** 2, size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=win_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=win_size) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def ssim_analysis(grouped_results, save_dir):
    """Compute SSIM between pred and GT vorticity per timestep."""
    print("\n>>> Generating SSIM analysis...")

    # Compute per-timestep SSIM for all grouped cases
    all_ssim = []
    for r in grouped_results:
        pred = r['pred_omega']
        target = r['target_omega']
        T = pred.shape[0]

        sample_ssim = []
        for t in range(T):
            sample_ssim.append(_compute_ssim(target[t], pred[t]))
        all_ssim.append(np.array(sample_ssim))
        r['mean_ssim'] = float(np.mean(sample_ssim))

    # Mean across all cases (truncated to shortest)
    min_len = min(len(s) for s in all_ssim)
    mean_ssim = np.mean([s[:min_len] for s in all_ssim], axis=0)

    # Plot up to 5 cases + mean
    to_plot = grouped_results[:5]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: temporal SSIM (up to 5 cases + mean)
    for i, r in enumerate(to_plot):
        timesteps = np.arange(len(all_ssim[i]))
        color = plt.cm.tab10(i)
        axes[0].plot(timesteps, all_ssim[i], color=color, linewidth=1.2,
                     label=r['case_tag'])

    axes[0].plot(np.arange(min_len), mean_ssim, 'k--', linewidth=2.0,
                 label=f'Mean (all {len(grouped_results)} cases)')
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("SSIM")
    axes[0].set_title("SSIM Over Time")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, loc='best')
    axes[0].grid(True, alpha=0.3)

    # Right: 1D strip plot of per-case mean SSIM
    sample_means = np.array([r['mean_ssim'] for r in grouped_results])
    avg_ssim = float(np.mean(sample_means))
    y_jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(sample_means))

    axes[1].scatter(sample_means, y_jitter, color='teal', s=50, alpha=0.7,
                    zorder=3, edgecolors='white', linewidth=0.5)
    axes[1].scatter([avg_ssim], [0], color='red', s=120, zorder=5, marker='D',
                    edgecolors='black', linewidth=1.0)
    axes[1].axvline(x=avg_ssim, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].set_xlabel("Mean SSIM")
    axes[1].set_yticks([])
    axes[1].set_title(f"SSIM per Test Case  (avg = {avg_ssim:.4f})")
    axes[1].grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'summary', 'ssim_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved SSIM analysis")


# ==========================================
# 9. LATENT SPACE TRAJECTORIES
# ==========================================
def visualize_latent_space(results, cfg, save_dir):
    """Visualize latent trajectories in 2D/3D and PCA variance."""
    print("\n>>> Generating latent space visualizations...")

    latent_dir = os.path.join(save_dir, 'latent')
    n_z = cfg.N_Z

    # The SNN latent is a continuous membrane potential (not a spike train), so
    # the latent trajectories can be read as smooth latent coordinates.
    is_spike_latent = False
    _title_note = ""

    all_latent = []
    for r in results:
        all_latent.append(r['latent'])

    latent_concat = np.concatenate(all_latent, axis=0)

    # --- Plot 1: Latent trajectories ---
    if n_z > 3:
        # t-SNE projection for high-dimensional latent spaces
        # Build per-case arrays and labels
        case_latents = []
        case_labels_list = []
        for r in results:
            case_latents.append(r['latent'])
            case_labels_list.extend([r['case_tag']] * len(r['latent']))
        latent_all = np.concatenate(case_latents, axis=0)
        case_labels_arr = np.array(case_labels_list)
        unique_cases = list(dict.fromkeys(case_labels_list))  # preserve order

        perplexity = min(30, max(5, len(latent_all) // 4))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        z_2d = tsne.fit_transform(latent_all)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_cases), 10)))
        for i, case in enumerate(unique_cases):
            mask = case_labels_arr == case
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=[colors[i % len(colors)]], s=12, alpha=0.5,
                       label=case)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.set_title(f"t-SNE Projection of N_z={n_z} Latent Space" + _title_note)
        if len(unique_cases) <= 15:
            ax.legend(fontsize=7, markerscale=2, loc='best')
        ax.grid(True, alpha=0.3)
    elif n_z == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for r in results:
            latent = r['latent']
            ax.plot(latent[:, 0], latent[:, 1], latent[:, 2],
                    alpha=0.6, linewidth=1, label=r['case_tag'])
            ax.scatter(latent[0, 0], latent[0, 1], latent[0, 2],
                       marker='o', s=30, zorder=5)
        ax.set_xlabel("z_1")
        ax.set_ylabel("z_2")
        ax.set_zlabel("z_3")
        ax.set_title("Latent Space Trajectories (z_1, z_2, z_3)" + _title_note)
        if len(results) <= 10:
            ax.legend(fontsize=7)
    elif n_z == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in results:
            latent = r['latent']
            ax.plot(latent[:, 0], latent[:, 1], alpha=0.6, linewidth=1,
                    label=r['case_tag'])
            ax.scatter(latent[0, 0], latent[0, 1], marker='o', s=30, zorder=5)
        ax.set_xlabel("z_1")
        ax.set_ylabel("z_2")
        ax.set_title("Latent Space Trajectories" + _title_note)
        if len(results) <= 10:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        for r in results:
            latent = r['latent']
            ax.plot(np.arange(len(latent)), latent[:, 0], alpha=0.6,
                    linewidth=1, label=r['case_tag'])
        ax.set_xlabel("Timestep")
        ax.set_ylabel("z_1 (spike activity)" if is_spike_latent else "z_1")
        ax.set_title("Latent Variable Over Time" + _title_note)
        if len(results) <= 10:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(latent_dir, 'latent_space_trajectories.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 2: PCA explained variance ---
    if n_z >= 2:
        n_components = min(n_z, latent_concat.shape[0])
        pca = PCA(n_components=n_components)
        pca.fit(latent_concat)
        cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(cumvar) + 1), cumvar, 'b-o', markersize=5)
        ax.set_xlabel("Number of PCA Components")
        ax.set_ylabel("Cumulative Explained Variance (%)")
        ax.set_title("PCA of Latent Space")
        ax.set_xticks(range(1, len(cumvar) + 1))
        ax.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='95%')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(latent_dir, 'latent_pca_variance.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print("  Saved latent space plots")


# ==========================================
# 10. LATENT-Cl CORRELATION
# ==========================================
def latent_cl_correlation(results, save_dir):
    """Correlate each latent dimension z_i with Cl(t)."""
    print("\n>>> Generating latent-Cl correlation...")

    all_latent = []
    all_cl = []
    for r in results:
        all_latent.append(r['latent'])
        all_cl.append(r['target_cl'])

    latent_all = np.concatenate(all_latent, axis=0)
    cl_all = np.concatenate(all_cl)
    n_z = latent_all.shape[1]

    corr = np.array([np.corrcoef(latent_all[:, i], cl_all)[0, 1]
                      for i in range(n_z)])
    corr = np.nan_to_num(corr)

    fig, ax = plt.subplots(figsize=(8, max(3, n_z * 0.5)))
    colors = ['steelblue' if c >= 0 else 'salmon' for c in corr]
    ax.barh(range(n_z), corr, color=colors)
    ax.set_yticks(range(n_z))
    ax.set_yticklabels([f"z_{i+1}" for i in range(n_z)])
    ax.set_xlabel("Pearson Correlation with Cl")
    ax.set_title("Latent Dimension — Lift Correlation")
    ax.set_xlim(-1, 1)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    for i, c in enumerate(corr):
        ax.text(c + 0.02 * np.sign(c), i, f"{c:.3f}", va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'latent', 'latent_cl_correlation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved latent-Cl correlation")


# ==========================================
# MAIN
# ==========================================
def main():
    cfg = _build_cfg_from_checkpoint(MODEL_WEIGHTS)

    print("=" * 60)
    print(f"  TEST + KPIs — GustAutoencoder (N_z={cfg.N_Z})")
    print("=" * 60)
    _print_config(cfg)

    device = get_device()
    print(f"\nDevice: {device}")

    # ---- Load data ----
    print("\n>>> Loading data...")
    if _is_cnn(cfg):
        # CNN baselines read RAW frames — no spike encoding. Build a lightweight
        # data dict with the SAME split as the SNN (pinned TEST_CASES). norm_scale
        # is unused (raw input); kept as 1.0 so the shared inference code runs.
        _rd = load_rd_params(cfg.RD_PARAMS_PATH, angles=cfg.ANGLES)
        _split_map = _pre_compute_case_split_map(cfg, _rd)
        # One "sample" per case for the CNN, so split_indices/case_labels index
        # directly into the sorted test-case list (used only for the snapshot's
        # test-case listing).
        _test_keys = sorted(k for k, v in _split_map.items() if v == 'test')
        data = {
            'case_paths':     list_interpolated_cases(cfg.INTERP_ROOT, cfg.ANGLES),
            'case_split_map': _split_map,
            'rd_params':      _rd,
            'norm_stats':     {'norm_scale': 1.0},
            'threshold':      None,
            'case_labels':    _test_keys,
            'split_indices':  {'test': list(range(len(_test_keys)))},
        }
    else:
        data = load_and_encode(cfg)

    # ---- Load model ----
    best_model_path = cfg.BEST_MODEL_PATH
    print(f"\n>>> Loading model from '{best_model_path}'...")
    if cfg.MODEL_TYPE == "cnn_param_matched":
        from parameter_matched import ParameterMatchedCNN
        model = ParameterMatchedCNN(
            n_z=cfg.N_Z, in_channels=cfg.IN_CHANNELS).to(device)
    elif cfg.MODEL_TYPE == "cnn_energy_matched":
        from energy_matched import EnergyMatchedCNN
        model = EnergyMatchedCNN(
            n_z=cfg.N_Z, in_channels=cfg.IN_CHANNELS).to(device)
    else:
        from SNN import SNN_Model
        model = SNN_Model(
            n_z=cfg.N_Z,
            in_channels=cfg.IN_CHANNELS,
            beta_init=cfg.BETA_INIT,
            fc_threshold=float(getattr(cfg, 'FC_THRESHOLD', 0.5)),
            gradient_checkpointing=False,
        ).to(device)
    state_dict = torch.load(best_model_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [COMPAT] Missing keys in checkpoint: {missing}")
        # If z_ln is absent the checkpoint was trained without LayerNorm on z.
        # LayerNorm(weight=1, bias=0) is NOT an identity — it still normalises
        # the input, corrupting a latent space that was never trained with it.
        # Replace with a true identity so inference matches the original model.
        if any('z_ln' in k for k in missing) and hasattr(model, 'encoder'):
            import torch.nn as nn
            model.encoder.z_ln = nn.Identity()
            print("  [COMPAT] z_ln absent from checkpoint — replaced with Identity()")
    if unexpected:
        print(f"  [COMPAT] Unexpected keys (ignored): {unexpected}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {model.__class__.__name__} (N_z={cfg.N_Z}) | "
          f"Parameters: {num_params:,}")

    # ---- Create results directory ----
    results_dir = create_results_dir(cfg)
    print(f"Results will be saved to: {results_dir}/\n")

    # ---- Save config snapshot ----
    snapshot_path = os.path.join(results_dir, 'config_snapshot.txt')
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        f.write(_get_config_snapshot(cfg))

    # Append test set case listing
    test_indices = data['split_indices']['test']
    case_labels = data['case_labels']
    rd_params = data['rd_params']

    # ---- Save dataset_split.txt (train / val / test case listing) ----
    _forced_keys = set()
    if getattr(cfg, 'SPLIT_MODE', 'gust_intensity') == 'random_case':
        for _forced_list in (cfg.TRAIN_CASES, cfg.VAL_CASES, cfg.TEST_CASES):
            _forced_keys.update(
                _parse_forced_cases(_forced_list, rd_params, '_')
            )
    split_listing_path = os.path.join(results_dir, 'dataset_split.txt')
    write_dataset_split_file(
        data['case_split_map'], rd_params, _forced_keys, split_listing_path,
    )
    print(f"Dataset split listing saved to: {split_listing_path}")
    with open(snapshot_path, 'a', encoding='utf-8') as f:
        f.write("\n\nTEST SET CASES\n")
        f.write("=" * 50 + "\n")
        seen_cases = set()
        for idx in test_indices:
            angle, case_id = case_labels[idx]
            key = (angle, case_id)
            if key not in seen_cases:
                seen_cases.add(key)
                tag = _case_tag(angle, case_id, rd_params)
                title = _case_title(angle, case_id, rd_params)
                f.write(f"  {tag:30s}  ({title})\n")
        f.write(f"\nTotal test sequences: {len(test_indices)}\n")
        f.write(f"Unique test cases: {len(seen_cases)}\n")

    print(f"Config snapshot saved to: {snapshot_path}")

    # ---- Phase 1: Inference ----
    # CNN baselines have no vorticity decoder either → treat as decoder-free so
    # the omega/SSIM analyses are skipped (as for the SNN encoder-only models).
    decoder_free = True
    results = run_inference(model, data, device, decoder_free=decoder_free, cfg=cfg)

    # Free model and data from device/RAM after inference
    del model, data
    gc.collect()
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Group results by case for temporal visualizations ----
    # max_frames=None → full case length (all timesteps)
    grouped_results = _group_results_by_case(results, max_frames=None)

    # Recompute Cl metrics at encounter level from full concatenated trajectories.
    # When WARMUP_STEPS > 0, exclude the cold-start window so the encounter-level
    # numbers match the loss-time supervised region.
    for r in grouped_results:
        W = int(r.get('warmup_steps', 0))
        if W > 0:
            r.update(compute_cl_metrics(r['pred_cl'][W:], r['target_cl'][W:]))
        else:
            r.update(compute_cl_metrics(r['pred_cl'], r['target_cl']))

    _avg_enc = lambda key: float(np.nanmean([r[key] for r in grouped_results]))
    print(f"\n{'='*60}")
    print(f"ENCOUNTER-LEVEL TEST SUMMARY ({len(grouped_results)} cases)")
    print(f"{'='*60}")
    print(f"  CosineSim: {_avg_enc('cosine_similarity_cl'):.4f}  "
          f"RMSE: {_avg_enc('rmse_cl'):.6f}  "
          f"RelL2Err: {_avg_enc('rel_l2_error_cl'):.4f}")
    print(f"  PeakErr:   {_avg_enc('peak_cl_error'):.4f}  "
          f"PeakTiming: {_avg_enc('peak_timing_error'):.1f} steps  "
          f"NormIntAbsErr: {_avg_enc('normalized_integrated_abs_error'):.4f}")
    print(f"{'='*60}")

    # ---- Phase 2: KPIs ----
    # Vorticity-dependent visualizations are skipped for decoder-free models (v5+)
    if not decoder_free:
        # 1. Per-case animations (grouped — ~200 frames each)
        create_sample_animations(grouped_results, results_dir)

    # 2. Per-case Cl evolution (grouped)
    create_cl_evolution_per_case(grouped_results, results_dir, rd_params=rd_params)

    # 3. Cl evolution overlay (grouped) — KPI region only
    create_cl_evolution_all(grouped_results, results_dir)

    # 3b. Cl evolution overlay with warmup pass (warmup | bridge | KPI), split
    #     into multiple figures of 12 cases each. Skipped when warmup pass off.
    create_cl_evolution_all_warmup(grouped_results, results_dir, cases_per_fig=12)

    if not decoder_free:
        # 4. Temporal error evolution (grouped)
        temporal_error_analysis(grouped_results, results_dir)

        # 5. Per-pixel error maps (ungrouped — per-sequence snapshots)
        per_pixel_error_maps(results, results_dir)

        # 6. Temporal cross-correlation (grouped)
        temporal_cross_correlation(grouped_results, results_dir)

        # 7. SSIM analysis (grouped)
        ssim_analysis(grouped_results, results_dir)

    # 8. Case summary txt (encounter-level metrics)
    create_case_summary_txt(grouped_results, results_dir)

    # 8b. Early-window (warmup effect) analysis — metrics/plots restricted to
    #     the first N_WARMUP_METRIC_FRAMES frames, written to warmup_<N>/.
    create_warmup_window_analysis(grouped_results, results_dir, N_WARMUP_METRIC_FRAMES)

    # 9. Training / validation loss curves
    plot_loss_curves(cfg, results_dir)

    # 10. Latent space trajectories + PCA (ungrouped — all sequences)
    visualize_latent_space(results, cfg, results_dir)

    # 10. Latent-Cl correlation (ungrouped)
    latent_cl_correlation(results, results_dir)

    # Free large per-sample arrays now that all spatial plots are done
    for r in results:
        r.pop('pred_omega', None)
        r.pop('target_omega', None)
        r.pop('latent', None)

    # 11. Accuracy metrics — encounter-level (after KPIs have added their scalars)
    compute_accuracy_metrics(grouped_results, results_dir)

    del grouped_results
    gc.collect()

    # ---- Done ----
    print(f"\n{'='*60}")
    print(f"All KPIs complete! Results saved to:")
    print(f"  {results_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
