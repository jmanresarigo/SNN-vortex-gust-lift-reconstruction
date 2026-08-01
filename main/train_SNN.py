"""
train_SNN.py — Train the spiking Cl encoder (models/SNN.py, SNN_Model).

Reads the spike-encoded vorticity produced by pre_encoder.py and trains the SNN
to predict the lift coefficient Cl directly (decoder-free), optimising the MSE
between the reconstructed Cl and the ground-truth Cl. Two training modes are
available (TRAINING_MODE): "random_window" (independent windows) and
"stateful_tbptt" (chronological chunks carrying LIF membrane state). Mixed
precision is bfloat16 on CUDA. All run settings are edited in the CONFIG blocks
below.

Output — each run creates a timestamped folder
    training/SNN_training/SNN_Nz{N_Z}_<timestamp>/
containing:
    config_snapshot.txt   full configuration for reproducibility
    dataset_split.txt     the train / val / test case listing
    best.pth              best-val checkpoint (only when VAL_RATIO > 0)
    epoch_<N>.pth         periodic checkpoints (no-val mode) / final.pth
    loss_curves.pt        {'train_losses': [...], 'val_losses': [...]}

Usage:
    python train_SNN.py
"""

import gc
import os
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

# Add the models/ directory to the path so we can import from it
sys.path.insert(0, os.path.join(_HERE, '..', 'models'))

from pre_encoder import (
    load_and_encode,
    get_case_nt,
    load_encoded_chunk,
    write_dataset_split_file,
    _parse_forced_cases,
)
from loss import combined_loss


# ==========================================
# TRAINING DESCRIPTION
# ==========================================
TRAINING_DESCRIPTION = "SNN Rate_Encoding Nz=16 T100 P99 gamma=0.6 gain=1.0 Dataset_interpolated_cubic_spline_N4 selected_test_cases (No val set) Stateful TBPTT Training"

# ==========================================
# FINE TUNING / RESUME
# ==========================================
# Continue (resume) training from a previously saved checkpoint — e.g. when a
# run was interrupted and you want to pick it up "as if it never stopped".
#
#   FINE_TUNE_ENABLED  : master switch. False = normal fresh run.
#   FINE_TUNE_FOLDER   : checkpoint subfolder under SNN_training that holds the
#                        weights (e.g. "SNN_Nz16_July_17_2026_00h_34m").
#   FINE_TUNE_WEIGHTS  : the weights file inside that folder to start from
#                        (e.g. "epoch_40.pth").
#   FINE_TUNE_START_EPOCH : epochs already completed at that checkpoint. Leave
#                        None to auto-parse from a name like "epoch_40.pth"
#                        (→ 40); set explicitly if the filename has no number.
#
# When enabled, training RESUMES IN THE SAME FOLDER: the model weights are
# loaded, the cosine-LR schedule is fast-forwarded to the resume epoch, the
# prior loss curve is carried over, and periodic checkpoints continue their
# numbering (epoch_50, epoch_60, …) up to EPOCHS. The original config_snapshot
# is left untouched.
#
# IMPORTANT: keep every other setting in this file identical to the original
# run (N_Z, DATASET, split, EPOCHS, LR, …). Only the model WEIGHTS are stored in
# a .pth — the optimizer (Adam moment) state is NOT, so it restarts cold; this
# is expected and usually recovers within a few epochs.
FINE_TUNE_ENABLED     = False
FINE_TUNE_FOLDER      = "SNN_Nz16_July_20_2026_16h_44m"
FINE_TUNE_WEIGHTS     = "epoch_70.pth"
FINE_TUNE_START_EPOCH = None

# ==========================================
# ENCODING
# ==========================================
# Rate encoding always uses robust normalization (P{NORM_PERCENTILE}) followed
# by the amplified transfer function p = min(1, GAIN·|x|^GAMMA). Delta encoding
# uses the SAME robust normalization percentile before thresholding, so the
# encoding comparison (rate vs delta vs hybrid) is fully controlled.
ENCODING         = "rate"       # "rate" | "delta" | "latency" | "rate+delta"
THRESHOLD_FACTOR = 1.4          # only used when ENCODING = "delta" or "rate+delta"
ZERO_FIRST_FRAME = True         # zero first spike frame to suppress padding artifact (delta / rate+delta)
BINS             = 1            # only used when ENCODING = "rate" or "rate+delta"

# Robust normalization percentile shared by rate and delta encoding (100 = absmax).
NORM_PERCENTILE  = 99

# Rate transfer-function parameters (applied for "rate" and "rate+delta").
GAMMA            = 0.6          # power-law compression (< 1 boosts weak signals)
GAIN             = 1.0          # firing rate amplifier (> 1 increases activity)

# ==========================================
# DATA
# ==========================================
TIME_STEPS      = 100               # timesteps per sequence
TRAINING_MODE   = "stateful_tbptt"  # "random_window" | "stateful_tbptt"
                                    # stateful_tbptt: process each case as ordered chunks,
                                    # carrying LIF membrane state across chunks.
                                    # Note: random_window on Windows + original dataset
                                    # hits a shared-memory limit with NUM_WORKERS > 0;
                                    # use NUM_WORKERS = 0 there, or this stateful path.

# ==========================================
# DATASET
# ==========================================
DATASET = "Dataset_original" # "Dataset_original" | "Dataset_interpolated_cubic_spline_N4"

# NOTE: TIME_STEPS is always interpreted in the active dataset's units. For an
# interpolated dataset, TIME_STEPS=100 covers a shorter physical duration than
# for the original dataset.

# ==========================================
# DATASET SPLIT
# ==========================================
# The split is always performed by "random_case": cases are stratified by gust
# intensity |G| across all selected angles and split randomly, respecting
# VAL_RATIO / TEST_RATIO and any explicit case overrides below.
ANGLES     = [20, 30, 40, 50, 60]  # angles of attack to include

VAL_RATIO  = 0.00
TEST_RATIO = 0.15

# Explicit case overrides. Force specific RD cases into a given split. Format:
#   "AoA{angle}_RD{idx}_G_{value}"  (use 'n' prefix for negative G)
# Example: ["AoA20_RD19_G_n2.6", "AoA20_RD12_G_2.8"]
# Empty lists fall back to pure random_case selection. Cases listed here are
# pinned to the chosen split; the remaining cases are then distributed by the
# stratified random logic, respecting VAL_RATIO / TEST_RATIO.
TRAIN_CASES = []
VAL_CASES = []
TEST_CASES = [
    "AoA20_RD5_G_n2.2",
    "AoA20_RD8_G_0.8",
    "AoA20_RD17_G_2.8",
    "AoA20_RD29_G_1.2",
    "AoA30_RD2_G_3.4",
    "AoA30_RD4_G_n1.2",
    "AoA30_RD11_G_1.2",
    "AoA30_RD13_G_n3.6",
    "AoA30_RD22_G_n0.4",
    "AoA40_RD1_G_n2.2",
    "AoA40_RD6_G_1.8",
    "AoA40_RD12_G_n2.0",
    "AoA40_RD16_G_4.0",
    "AoA40_RD19_G_0.8",
    "AoA50_RD4_G_n3.0",
    "AoA50_RD7_G_0.4",
    "AoA50_RD8_G_n2.0",
    "AoA50_RD18_G_2.2",
    "AoA50_RD21_G_n4.0",
    "AoA60_RD6_G_n3.0",
    "AoA60_RD9_G_n3.0",
    "AoA60_RD10_G_n0.4",
    "AoA60_RD24_G_n1.8",
    "AoA60_RD26_G_2.0",
]

# ==========================================
# ARCHITECTURE
# ==========================================
# The model is the spiking Cl encoder defined in models/SNN.py (SNN_Model):
# two convolutional spiking blocks followed by a direct latent + Cl readout.
N_Z           = 16              # latent dimension (3, 8, 16, 32, ...)
IN_CHANNELS   = 4 if ENCODING == 'rate+delta' else 2   # 4 for hybrid, 2 otherwise
BETA_INIT     = 0.9             # initial LIF membrane decay factor
FC_THRESHOLD  = 0.5             # FC LIF threshold (0.5 → ~30% firing with LN)

# ==========================================
# MONITORING
# ==========================================
MONITOR_ENCODER = True          # print z_mean/z_std every epoch to detect dead encoder

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
# Mixed precision is always bfloat16 on CUDA (safe for SNN surrogate gradients —
# 8-bit exponent, no gradient scaling needed). On CPU/MPS it runs in float32.
BATCH_SIZE   = 10
EPOCHS       = 100
LR           = 1e-4 
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
NUM_WORKERS        = 4          # DataLoader workers (random_window mode)
PREFETCH_FACTOR    = 2          # DataLoader prefetch factor (random_window mode)
PERSISTENT_WORKERS = True       # DataLoader persistent workers (random_window mode)
STATEFUL_PREFETCH_WORKERS = NUM_WORKERS       # ThreadPoolExecutor threads (stateful_tbptt mode)
STATEFUL_PREFETCH_FACTOR  = PREFETCH_FACTOR   # prefetch window = workers × factor
KEEP_ONLY_BEST_AND_FINAL = True # delete periodic checkpoints to save disk space

# No-validation mode (VAL_RATIO == 0): when validation is disabled, fold the
# would-be-val cases into the training set and save a periodic snapshot every
# SAVE_EVERY_N_EPOCHS epochs (plus the final model). best.pth is not saved
# because there is no val loss to compare against; pick the best epoch by
# running test.py on the periodic checkpoints. Early stopping is ignored in
# this mode.
SAVE_EVERY_N_EPOCHS = 10

# Early stopping (applies only when VAL_RATIO > 0)
EARLY_STOP        = True   # stop training if val loss stagnates
EARLY_STOP_EPOCHS = 15     # patience: stop after this many epochs with no val improvement

# ==========================================
# LOSS FUNCTION
# ==========================================
# The training objective is the MSE between the reconstructed Cl and the
# ground-truth Cl (see loss.py). There are no other loss terms.
LOSS_TYPE = "MSE"


# ==========================================
# PATHS
# ==========================================
import re as _re

def _find_datasets_root():
    env = os.environ.get('THESIS_DATA_ROOT')
    if env:
        return os.path.abspath(env)
    base = os.path.join(_HERE, '..')
    for name in ('Datasets', 'DATASETS', 'datasets'):
        candidate = os.path.normpath(os.path.join(base, name))
        if os.path.isdir(candidate):
            return candidate
    # Legacy fallback: datasets live directly under the repo root
    return os.path.normpath(base)

DATASETS_ROOT    = _find_datasets_root()
DATASET_PATH     = os.path.join(DATASETS_ROOT, DATASET)
RD_PARAMS_PATH   = os.path.join(DATASETS_ROOT, 'RD_list_FT2023.xlsx')
# Results of running this script are saved under training/SNN_training/.
CHECKPOINTS_ROOT = os.path.join(_HERE, '..', 'training', 'SNN_training')

# Derived from DATASET folder name — do not edit these directly
DATASET_TYPE   = "original" if DATASET == "Dataset_original" else "interpolated"
_n_match       = _re.search(r'_N(\d+)$', DATASET)
N_INTERPOLATED = int(_n_match.group(1)) if _n_match else 0
INTERP_ROOT    = DATASET_PATH if DATASET_TYPE == "interpolated" else None
VORTICITY_DIR  = os.path.join(DATASET_PATH, 'vorticity')
LIFT_DIR       = os.path.join(DATASET_PATH, 'lift')
CACHE_DIR      = os.path.join(DATASET_PATH, 'cache')
_ds_tag        = f"_ds-{DATASET}" if DATASET_TYPE == "interpolated" else ""

# ==========================================
# DERIVED VALUES
# ==========================================
_angles_tag    = "_".join(str(a) for a in sorted(ANGLES))
_enc_str       = ENCODING.replace('+', '_')                # filesystem-safe (rate+delta → rate_delta)
_norm_tag      = f"_p{NORM_PERCENTILE}" if NORM_PERCENTILE < 100 else ""
_amp_tag       = f"_g{GAMMA}_G{GAIN}" if (GAMMA != 1.0 or GAIN != 1.0) else ""
_delta_th_tag  = f"_deltaTh{THRESHOLD_FACTOR}" if ENCODING in ('delta', 'rate+delta') else ""
CACHE_PATH = os.path.join(CACHE_DIR,
                           f"processed_{_enc_str}_T{TIME_STEPS}_b{BINS}{_norm_tag}{_amp_tag}{_delta_th_tag}_a{_angles_tag}{_ds_tag}.pt")


# ==========================================
# UTILITIES
# ==========================================
def _resolve_fine_tune():
    """Resolve the fine-tuning/resume checkpoint from the FINE TUNING config.

    Returns (folder_path, weights_path, start_epoch) when FINE_TUNE_ENABLED,
    otherwise None. Raises with a clear message if the folder/file is missing or
    the start epoch cannot be determined.
    """
    if not FINE_TUNE_ENABLED:
        return None

    folder  = os.path.join(CHECKPOINTS_ROOT, FINE_TUNE_FOLDER)
    weights = os.path.join(folder, FINE_TUNE_WEIGHTS)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"[FINE_TUNE] Checkpoint folder not found:\n  {os.path.normpath(folder)}\n"
            f"Check FINE_TUNE_FOLDER.")
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"[FINE_TUNE] Weights file not found:\n  {os.path.normpath(weights)}\n"
            f"Check FINE_TUNE_WEIGHTS.")

    if FINE_TUNE_START_EPOCH is not None:
        start_epoch = int(FINE_TUNE_START_EPOCH)
    else:
        m = _re.search(r'epoch_(\d+)', os.path.basename(FINE_TUNE_WEIGHTS))
        if not m:
            raise ValueError(
                f"[FINE_TUNE] Could not parse the epoch number from "
                f"'{FINE_TUNE_WEIGHTS}'. Set FINE_TUNE_START_EPOCH explicitly.")
        start_epoch = int(m.group(1))

    if start_epoch < 0:
        raise ValueError(f"[FINE_TUNE] start_epoch must be >= 0, got {start_epoch}.")
    if start_epoch >= EPOCHS:
        raise ValueError(
            f"[FINE_TUNE] start_epoch ({start_epoch}) >= EPOCHS ({EPOCHS}) — "
            f"nothing left to train. Increase EPOCHS to continue.")
    return folder, weights, start_epoch


def _build_model():
    """Instantiate the spiking Cl encoder (models/SNN.py)."""
    from SNN import SNN_Model
    return SNN_Model(
        n_z=N_Z,
        in_channels=IN_CHANNELS,
        beta_init=BETA_INIT,
        fc_threshold=FC_THRESHOLD,
    )


def get_device():
    """Select best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fmt_time(seconds):
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_config():
    """Print a readable summary of the training configuration."""
    print("=" * 60)
    print("  TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Description:  {TRAINING_DESCRIPTION}")
    print(f"  Model:        SNN_Model (SNN.py, N_z={N_Z})")
    print(f"  Dataset:      {DATASET}")
    print(f"  Dataset path: {DATASET_PATH}")

    # --- Encoding ---
    if ENCODING == "rate":
        print(f"  Encoding:     {ENCODING} (bins={BINS})")
        print(f"  Norm:         P{NORM_PERCENTILE}" +
              (" (robust, clipped)" if NORM_PERCENTILE < 100 else " (absmax)"))
        print(f"  Transfer fn:  p = min(1, {GAIN} · |x|^{GAMMA})")
    elif ENCODING == "delta":
        print(f"  Encoding:     {ENCODING} "
              f"(threshold_factor={THRESHOLD_FACTOR}, zero_first={ZERO_FIRST_FRAME})")
        print(f"  Norm:         P{NORM_PERCENTILE}" +
              (" (robust, clipped)" if NORM_PERCENTILE < 100 else " (absmax)"))
    elif ENCODING == "rate+delta":
        print(f"  Encoding:     {ENCODING} (bins={BINS})")
        print(f"  Norm:         P{NORM_PERCENTILE}" +
              (" (robust, clipped)" if NORM_PERCENTILE < 100 else " (absmax)"))
        print(f"  Rate fn:      p = min(1, {GAIN} · |x|^{GAMMA})")
        print(f"  Delta:        threshold_factor={THRESHOLD_FACTOR}, zero_first={ZERO_FIRST_FRAME}")
    else:
        print(f"  Encoding:     {ENCODING}")

    # --- Data ---
    _ch_desc = ("(rate_pos, rate_neg, delta_pos, delta_neg)"
                if ENCODING == "rate+delta" else "(ω+, ω-)")
    print(f"  Input:        {IN_CHANNELS} spike channels {_ch_desc}")
    print(f"  Time steps:   {TIME_STEPS} supervised (no overlap)")
    print(f"  Angles:       {ANGLES}")

    # --- Split ---
    print(f"  Split mode:   random_case (stratified by |G|, all angles)")
    print(f"  Val ratio:    {VAL_RATIO}")
    print(f"  Test ratio:   {TEST_RATIO}")
    if TRAIN_CASES:
        print(f"  Forced train: {TRAIN_CASES}")
    if VAL_CASES:
        print(f"  Forced val:   {VAL_CASES}")
    if TEST_CASES:
        print(f"  Forced test:  {TEST_CASES}")

    # --- Architecture ---
    print(f"  Architecture: N_z={N_Z}, beta={BETA_INIT}, fc_threshold={FC_THRESHOLD}")

    # --- Training ---
    print(f"  Training mode:{TRAINING_MODE}")
    if TRAINING_MODE == "stateful_tbptt":
        print(f"  Stateful:     {BATCH_SIZE} cases/batch, chronological chunks, state carry ON")
        print(f"  Chunk length: {TIME_STEPS}  (state detached after each chunk)")
        print(f"  Case order:   shuffled each epoch")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Epochs:       {EPOCHS}")
    if VAL_RATIO == 0:
        print(f"  Validation:   DISABLED (VAL_RATIO=0; val cases folded into train)")
        print(f"  Save every:   {SAVE_EVERY_N_EPOCHS} epochs  (+ final; no best.pth)")
        print(f"  Early stop:   N/A (no val loss to track)")
    elif EARLY_STOP:
        print(f"  Early stop:   ON  (patience={EARLY_STOP_EPOCHS} epochs)")
    else:
        print(f"  Early stop:   OFF")
    print(f"  Learning rate:{LR}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Grad clip:    {GRAD_CLIP}")
    print(f"  Precision:    bfloat16 (CUDA autocast) / float32 (CPU/MPS)")

    # --- Loss ---
    print(f"  Loss:         {LOSS_TYPE} (MSE between reconstructed Cl and ground-truth Cl)")

    # --- Misc ---
    print(f"  Monitor:      {'ON' if MONITOR_ENCODER else 'OFF'}")
    print(f"  Cache:        {os.path.basename(CACHE_PATH)}")
    print(f"  Output dir:   {CHECKPOINTS_ROOT}")
    print(f"  Datasets root:{DATASETS_ROOT}")
    print("=" * 60)


def _get_config_snapshot():
    """Return a string with every configuration value for the current run.

    The snapshot lists all user-selectable options so a run is fully
    reproducible from the saved config_snapshot.txt alone.
    """
    lines = [
        "CONFIGURATION SNAPSHOT",
        "=" * 50,
        f"TRAINING_DESCRIPTION    = {TRAINING_DESCRIPTION}",
        "",
        "FINE_TUNE",
        "-" * 30,
        f"FINE_TUNE_ENABLED       = {FINE_TUNE_ENABLED}",
        f"FINE_TUNE_FOLDER        = {FINE_TUNE_FOLDER}",
        f"FINE_TUNE_WEIGHTS       = {FINE_TUNE_WEIGHTS}",
        f"FINE_TUNE_START_EPOCH   = {FINE_TUNE_START_EPOCH}",
        "",
        "ENCODING",
        "-" * 30,
        f"ENCODING                = {ENCODING}",
        f"NORM_PERCENTILE         = {NORM_PERCENTILE}",
        f"BINS                    = {BINS}",
        f"GAMMA                   = {GAMMA}",
        f"GAIN                    = {GAIN}",
        f"THRESHOLD_FACTOR        = {THRESHOLD_FACTOR}",
        f"ZERO_FIRST_FRAME        = {ZERO_FIRST_FRAME}",
        "",
        "DATASET",
        "-" * 30,
        f"DATASET                 = {DATASET}",
        f"DATASET_TYPE            = {DATASET_TYPE}",
        f"N_INTERPOLATED          = {N_INTERPOLATED}",
    ]
    if DATASET_TYPE == "interpolated":
        lines.append(f"PHYSICAL_DURATION_PER_WINDOW = "
                     f"{TIME_STEPS / (N_INTERPOLATED + 1):.4f}  (in original-grid units)")

    lines += [
        "",
        "DATA",
        "-" * 30,
        f"TRAINING_MODE           = {TRAINING_MODE}",
        f"TIME_STEPS              = {TIME_STEPS}",
    ]
    if TRAINING_MODE == "stateful_tbptt":
        lines += [
            f"CHUNK_LEN               = {TIME_STEPS}",
            f"CASES_PER_BATCH         = {BATCH_SIZE}",
            f"STATE_CARRY             = True",
            f"STATE_DETACH_EVERY_CHUNK = True",
            f"CASE_SHUFFLE            = True",
            f"CHUNK_SHUFFLE           = False",
            f"DROP_LAST_CHUNK         = True",
            f"DROP_INCOMPLETE_BATCH   = True",
        ]

    lines += [
        f"ANGLES                  = {ANGLES}",
        "",
        "SPLIT",
        "-" * 30,
        f"SPLIT_MODE              = random_case",
        f"VAL_RATIO               = {VAL_RATIO}",
        f"TEST_RATIO              = {TEST_RATIO}",
        f"TRAIN_CASES             = {TRAIN_CASES}",
        f"VAL_CASES               = {VAL_CASES}",
        f"TEST_CASES              = {TEST_CASES}",
        "",
        "ARCHITECTURE",
        "-" * 30,
        f"MODEL                   = SNN_Model (SNN.py)",
        f"N_Z                     = {N_Z}",
        f"IN_CHANNELS             = {IN_CHANNELS}",
        f"BETA_INIT               = {BETA_INIT}",
        f"FC_THRESHOLD            = {FC_THRESHOLD}",
        "",
        "TRAINING",
        "-" * 30,
        f"BATCH_SIZE              = {BATCH_SIZE}",
        f"EPOCHS                  = {EPOCHS}",
        f"NO_VAL_MODE             = {VAL_RATIO == 0}",
        f"SAVE_EVERY_N_EPOCHS     = {SAVE_EVERY_N_EPOCHS}",
        f"EARLY_STOP              = {EARLY_STOP and VAL_RATIO > 0}",
        f"EARLY_STOP_EPOCHS       = {EARLY_STOP_EPOCHS}",
        f"LR                      = {LR}",
        f"WEIGHT_DECAY            = {WEIGHT_DECAY}",
        f"GRAD_CLIP               = {GRAD_CLIP}",
        f"PRECISION               = bfloat16 (CUDA autocast) / float32 (CPU/MPS)",
        f"NUM_WORKERS             = {NUM_WORKERS}",
        f"PREFETCH_FACTOR         = {PREFETCH_FACTOR}",
        f"PERSISTENT_WORKERS      = {PERSISTENT_WORKERS}",
        f"STATEFUL_PREFETCH_WORKERS = {STATEFUL_PREFETCH_WORKERS}",
        f"STATEFUL_PREFETCH_FACTOR  = {STATEFUL_PREFETCH_FACTOR}",
        f"KEEP_ONLY_BEST_AND_FINAL = {KEEP_ONLY_BEST_AND_FINAL}",
        f"MONITOR_ENCODER         = {MONITOR_ENCODER}",
        "",
        "LOSS",
        "-" * 30,
        f"LOSS_TYPE               = {LOSS_TYPE}  (MSE between reconstructed Cl and ground-truth Cl)",
        "=" * 50,
    ]
    return "\n".join(lines)


def _print_model_training_defaults(model):
    """Print model-specific recommended defaults if provided by the model class."""
    defaults = getattr(model.__class__, "TRAINING_DEFAULTS", None)
    if not defaults:
        return

    current = {
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "GRAD_CLIP": GRAD_CLIP,
        "BETA_INIT": BETA_INIT,
        "EPOCHS": EPOCHS,
    }

    print("\nModel-specific training defaults:")
    for key, recommended in defaults.items():
        cur = current.get(key, None)
        if cur is None:
            print(f"  {key}: recommended={recommended} (not set in SNN_train.py)")
            continue
        marker = "OK" if cur == recommended else "DIFF"
        print(f"  {key}: current={cur} | recommended={recommended} [{marker}]")


def _save_state_dict_atomic(state_dict, target_path):
    """Atomically overwrite a model state_dict on disk."""
    tmp_path = target_path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, target_path)


def _get_autocast_context(device):
    """Return the bfloat16 autocast context for CUDA training (float32 elsewhere)."""
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast('cuda', dtype=torch.bfloat16)


# ==========================================
# STATEFUL TBPTT PREFETCH HELPERS
# ==========================================
def _load_stateful_batch_chunk(case_paths, case_batch, chunk_idx,
                               norm_scale, threshold, cfg):
    """Load and encode chunk `chunk_idx` for a batch of case indices.

    The supervised frame window for case entry k is
    `[chunk_idx*T, (chunk_idx+1)*T)` where T = cfg.TIME_STEPS (non-overlapping
    chunks, one trajectory per case).

    Called from worker threads inside _prefetch_stateful_chunks.

    Returns:
        x_batch:  (T, B, C, H, W) contiguous CPU float32 tensor
        cl_batch: (T, B, 1)       contiguous CPU float32 tensor
    """
    T = cfg.TIME_STEPS
    x_list, cl_list = [], []
    for ci in case_batch:
        start = chunk_idx * T
        end   = start + T
        x_c, cl_c = load_encoded_chunk(
            case_paths[ci], start, end, norm_scale, threshold, cfg)
        x_list.append(x_c)   # (T, C, H, W)
        cl_list.append(cl_c)  # (T, 1)
    x_batch  = torch.stack(x_list,  dim=0).permute(1, 0, 2, 3, 4).contiguous()
    cl_batch = torch.stack(cl_list, dim=0).permute(1, 0, 2).contiguous()
    return x_batch, cl_batch


def _prefetch_stateful_chunks(case_paths, case_batch, n_chunks,
                               norm_scale, threshold, cfg,
                               num_workers, prefetch_factor):
    """Yield stateful chunks in chronological order with async prefetching.

    `case_batch` is a list of case indices. All entries share the same
    `chunk_idx` sequence so the loop iterates 0..n_chunks-1.

    Submits up to num_workers * prefetch_factor futures ahead of the current
    yield position so chunk loading/encoding overlaps GPU training. Order is
    strictly preserved — chunk k is always yielded before chunk k+1.

    Yields:
        (x_batch_cpu, cl_batch_cpu) — CPU tensors, same shapes as
        _load_stateful_batch_chunk returns.
    """
    max_prefetch = max(1, num_workers * prefetch_factor)

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures     = {}
        next_submit = 0
        next_yield  = 0

        # Fill the initial prefetch window
        while next_submit < min(n_chunks, max_prefetch):
            futures[next_submit] = executor.submit(
                _load_stateful_batch_chunk,
                case_paths, case_batch, next_submit,
                norm_scale, threshold, cfg,
            )
            next_submit += 1

        while next_yield < n_chunks:
            x_batch, cl_batch = futures.pop(next_yield).result()

            # Keep the window full
            if next_submit < n_chunks:
                futures[next_submit] = executor.submit(
                    _load_stateful_batch_chunk,
                    case_paths, case_batch, next_submit,
                    norm_scale, threshold, cfg,
                )
                next_submit += 1

            yield x_batch, cl_batch
            next_yield += 1


# ==========================================
# STATEFUL TBPTT TRAINING LOOP
# ==========================================
def _run_stateful_tbptt(
    model, optimizer, scheduler,
    case_paths, train_case_idxs, val_case_idxs,
    norm_scale, threshold, cfg, device,
    best_model_path, final_model_path, ckpt_dir,
    no_val=False,
    start_epoch=0,
    init_train_losses=None,
    init_val_losses=None,
):
    """Full epoch loop for stateful TBPTT with batched case processing.

    Each epoch:
      - Shuffles training cases then groups them into batches of BATCH_SIZE.
      - For each batch, all B cases share one state tensor (B independent
        membrane trajectories), and their chunks are processed together in
        chronological order (non-overlapping TIME_STEPS chunks). State is
        carried across chunks and detached after each optimizer step.
      - The last (partial) training batch is dropped so every optimizer step
        sees exactly BATCH_SIZE cases; the dropped case rotates each epoch due
        to shuffling.
      - Validation uses the same batched structure but with inference_mode;
        the last incomplete batch is kept.
      - For batches whose cases differ in length, chunks are capped at the
        shortest case to keep the batch aligned.
    """
    best_val_loss = float('inf')
    epochs_since_best = 0
    early_stopped = False
    train_losses  = list(init_train_losses) if init_train_losses else []
    val_losses    = list(init_val_losses) if init_val_losses else []

    # Per-case chunk count (non-overlapping TIME_STEPS chunks), summed across
    # training cases.
    train_chunks_total = sum(
        max(0, get_case_nt(case_paths[i]) // TIME_STEPS)
        for i in train_case_idxs
    )
    val_chunks_total = sum(
        max(0, get_case_nt(case_paths[i]) // TIME_STEPS)
        for i in val_case_idxs
    )
    # Partial last batches are dropped (drop_last=True) so every optimizer step
    # sees exactly BATCH_SIZE cases.
    n_train_batches = len(train_case_idxs) // BATCH_SIZE
    _resume_note = f" (resuming at epoch {start_epoch + 1})" if start_epoch > 0 else ""
    print(f"\n>>> Starting stateful TBPTT training for {EPOCHS} epochs{_resume_note} "
          f"({len(train_case_idxs)} train cases → ~{n_train_batches} batches of {BATCH_SIZE}, "
          f"{len(val_case_idxs)} val cases, "
          f"{TIME_STEPS}-step chunks)...")
    print(f"[STATEFUL] Train case-chunks total: {train_chunks_total}")
    print(f"[STATEFUL] Val case-chunks total:   {val_chunks_total}")
    print(f"[STATEFUL] Cases per batch:         {BATCH_SIZE}")
    print(f"[STATEFUL] Prefetch workers:        {STATEFUL_PREFETCH_WORKERS}")
    print(f"[STATEFUL] Prefetch factor:         {STATEFUL_PREFETCH_FACTOR}")

    t_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        t_epoch_start = time.time()
        # ---- Training ----
        model.train()
        epoch_train_loss = 0.0
        n_train_chunks   = 0
        train_comp_sum   = {}

        # Shuffle cases, then group into drop_last batches of BATCH_SIZE.
        units = list(train_case_idxs)
        random.shuffle(units)
        n_full = len(units) // BATCH_SIZE   # drop_last: skip partial tail
        case_batches = [
            units[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            for i in range(n_full)
        ]
        random.shuffle(case_batches)
        n_work_units = sum(len(b) for b in case_batches)
        print(f"  [Epoch {epoch+1}] Train: {n_work_units} cases (drop_last), "
              f"{len(case_batches)} batches "
              f"(batch size: {BATCH_SIZE})")

        for case_batch in case_batches:
            B = len(case_batch)

            # Use the minimum chunk count across cases so all entries in the
            # batch stay aligned; tail chunks of longer cases are dropped.
            n_chunks = min(
                get_case_nt(case_paths[ci]) // TIME_STEPS
                for ci in case_batch
            )
            if n_chunks == 0:
                continue

            state = model.init_state(batch_size=B, device=device)

            for chunk_idx, (x_cpu, cl_cpu) in enumerate(_prefetch_stateful_chunks(
                case_paths, case_batch, n_chunks,
                norm_scale, threshold, cfg,
                num_workers=STATEFUL_PREFETCH_WORKERS,
                prefetch_factor=STATEFUL_PREFETCH_FACTOR,
            )):
                if device.type == "cuda":
                    x_cpu  = x_cpu.pin_memory()
                    cl_cpu = cl_cpu.pin_memory()

                x_batch  = x_cpu.to(device=device,  dtype=torch.float32, non_blocking=True)
                cl_batch = cl_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with _get_autocast_context(device):
                    z_seq, cl_pred, state = model(
                        x_batch, state=state, return_state=True)

                loss, components = combined_loss(cl_pred.float(), cl_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()

                state = model.detach_state(state)

                epoch_train_loss += loss.item()
                n_train_chunks   += 1
                for k, v in components.items():
                    train_comp_sum[k] = train_comp_sum.get(k, 0.0) + v

        avg_train            = epoch_train_loss / max(n_train_chunks, 1)
        avg_train_components = {k: v / max(n_train_chunks, 1) for k, v in train_comp_sum.items()}
        train_losses.append(avg_train)

        # ---- Validation ----
        if no_val:
            avg_val = float('nan')
            avg_val_components = {}
            last_z_mean = last_z_std = last_z_absmax = 0.0
        else:
            model.eval()
            epoch_val_loss = 0.0
            n_val_chunks   = 0
            val_comp_sum   = {}
            last_z_mean = last_z_std = last_z_absmax = 0.0

            # Validation batches are built in order (no shuffling) so state is
            # carried chronologically within each case trajectory.
            val_batches = [
                val_case_idxs[i : i + BATCH_SIZE]
                for i in range(0, len(val_case_idxs), BATCH_SIZE)
            ]

            with torch.inference_mode():
                for case_batch in val_batches:
                    B = len(case_batch)
                    n_chunks = min(
                        get_case_nt(case_paths[ci]) // TIME_STEPS
                        for ci in case_batch
                    )
                    if n_chunks == 0:
                        continue

                    state = model.init_state(batch_size=B, device=device)

                    for chunk_idx, (x_cpu, cl_cpu) in enumerate(_prefetch_stateful_chunks(
                        case_paths, case_batch, n_chunks,
                        norm_scale, threshold, cfg,
                        num_workers=NUM_WORKERS,
                        prefetch_factor=PREFETCH_FACTOR,
                    )):
                        if device.type == "cuda":
                            x_cpu  = x_cpu.pin_memory()
                            cl_cpu = cl_cpu.pin_memory()

                        x_batch  = x_cpu.to(device=device,  dtype=torch.float32, non_blocking=True)
                        cl_batch = cl_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

                        with _get_autocast_context(device):
                            z_seq, cl_pred, state = model(
                                x_batch, state=state, return_state=True)

                        loss, components = combined_loss(cl_pred.float(), cl_batch)

                        epoch_val_loss += loss.item()
                        n_val_chunks   += 1
                        for k, v in components.items():
                            val_comp_sum[k] = val_comp_sum.get(k, 0.0) + v

                        if MONITOR_ENCODER:
                            last_z_mean   = z_seq.mean().item()
                            last_z_std    = z_seq.std().item()
                            last_z_absmax = z_seq.abs().max().item()

                        state = model.detach_state(state)

            avg_val            = epoch_val_loss / max(n_val_chunks, 1)
            avg_val_components = {k: v / max(n_val_chunks, 1) for k, v in val_comp_sum.items()}
        val_losses.append(avg_val)
        scheduler.step()

        # ---- Logging ----
        lr_now        = optimizer.param_groups[0]['lr']
        elapsed       = time.time() - t_start
        epoch_elapsed = time.time() - t_epoch_start
        _val_str = "  no-val" if no_val else f"Val: {avg_val:.6f}"
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {avg_train:.6f} | {_val_str} | "
              f"LR: {lr_now:.2e} | "
              f"Epoch: {_fmt_time(epoch_elapsed)} | Total: {_fmt_time(elapsed)}")

        if MONITOR_ENCODER and not no_val:
            print(f"  z: mean={last_z_mean:+.4f} std={last_z_std:.4f} "
                  f"|max|={last_z_absmax:.4f}")

        train_comp_str = " | ".join(
            f"{k}: {v:.6f}" for k, v in avg_train_components.items()
        )
        print(f"  Train components: {train_comp_str}")
        if not no_val:
            val_comp_str = " | ".join(
                f"{k}: {v:.6f}" for k, v in avg_val_components.items()
            )
            print(f"  Val   components: {val_comp_str}")
        if device.type == "cuda" and (epoch == 0 or (epoch + 1) % 25 == 0):
            alloc = torch.cuda.memory_allocated() / 1024**2
            res   = torch.cuda.memory_reserved()  / 1024**2
            print(f"  VRAM: {alloc:.0f} MB allocated / {res:.0f} MB reserved")

        # ---- Save best / periodic checkpoint ----
        if no_val:
            if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
                ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> Periodic checkpoint saved: epoch_{epoch+1}.pth")
        else:
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_since_best = 0
                _save_state_dict_atomic(model.state_dict(), best_model_path)
                print(f"  -> Best model saved (val={best_val_loss:.6f})")
            else:
                epochs_since_best += 1
                if EARLY_STOP:
                    print(f"  -> No improvement for {epochs_since_best}/"
                          f"{EARLY_STOP_EPOCHS} epochs (best val={best_val_loss:.6f})")

            if (epoch + 1) % 25 == 0 and not KEEP_ONLY_BEST_AND_FINAL:
                ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), ckpt_path)

        torch.save({'train_losses': train_losses, 'val_losses': val_losses},
                   os.path.join(ckpt_dir, 'loss_curves.pt'))

        # ---- Early stopping ----
        if not no_val and EARLY_STOP and epochs_since_best >= EARLY_STOP_EPOCHS:
            print(f"\n[EARLY STOP] Val loss has not improved for "
                  f"{EARLY_STOP_EPOCHS} epochs. Stopping at epoch {epoch+1}/"
                  f"{EPOCHS} (best val={best_val_loss:.6f}).")
            early_stopped = True
            break

    # ---- Final model ----
    torch.save(model.state_dict(), final_model_path)
    total_time = time.time() - t_start
    finish_reason = "early-stopped" if early_stopped else "complete"
    print(f"\nTraining {finish_reason} in {total_time/60:.1f} min "
          f"({len(train_losses)}/{EPOCHS} epochs)")
    if no_val:
        print(f"Validation disabled — periodic checkpoints saved every "
              f"{SAVE_EVERY_N_EPOCHS} epochs in {ckpt_dir}/")
    else:
        print(f"Best val loss: {best_val_loss:.6f}")
        print(f"Best model:    {best_model_path}")
    print(f"Final model:   {final_model_path}")


# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================
def train():
    # ==========================================
    # 1. SETUP
    # ==========================================
    _print_config()
    device = get_device()
    print(f"\nDevice: {device}")

    # CUDA performance hints
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # auto-tune conv kernels
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {mem:.1f} GB")

    # Fine-tuning / resume: continue in the SAME folder as the source checkpoint;
    # a fresh run gets a new timestamped folder.
    fine_tune = _resolve_fine_tune()
    if fine_tune is not None:
        ckpt_dir, ft_weights, start_epoch = fine_tune
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[FINE_TUNE] Resuming from "
              f"{os.path.join(FINE_TUNE_FOLDER, FINE_TUNE_WEIGHTS)} — "
              f"continuing epochs {start_epoch + 1}..{EPOCHS} in the same folder.")
        # Preserve the original config_snapshot.txt; do not overwrite it.
    else:
        # Create timestamped checkpoint subfolder
        timestamp = datetime.now().strftime("%B_%d_%Y_%Hh_%Mm")
        ckpt_dir = os.path.join(CHECKPOINTS_ROOT, f"SNN_Nz{N_Z}_{timestamp}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ft_weights, start_epoch = None, 0
        # Save config snapshot
        with open(os.path.join(ckpt_dir, 'config_snapshot.txt'), 'w') as f:
            f.write(_get_config_snapshot())
    print(f"Checkpoint dir: {ckpt_dir}/")

    # Checkpoint file paths
    best_model_path  = os.path.join(ckpt_dir, 'best.pth')
    final_model_path = os.path.join(ckpt_dir, 'final.pth')

    # Build cfg namespace for pre_encoder. SPLIT_MODE is fixed to random_case.
    cfg = SimpleNamespace(
        TRAINING_MODE=TRAINING_MODE,
        DATASET=DATASET,
        DATASET_TYPE=DATASET_TYPE,
        N_INTERPOLATED=N_INTERPOLATED,
        INTERP_ROOT=INTERP_ROOT,
        VORTICITY_DIR=VORTICITY_DIR,
        LIFT_DIR=LIFT_DIR,
        RD_PARAMS_PATH=RD_PARAMS_PATH,
        CACHE_PATH=CACHE_PATH,
        ANGLES=ANGLES,
        TIME_STEPS=TIME_STEPS,
        ENCODING=ENCODING,
        BINS=BINS,
        THRESHOLD_FACTOR=THRESHOLD_FACTOR,
        ZERO_FIRST_FRAME=ZERO_FIRST_FRAME,
        NORM_PERCENTILE=NORM_PERCENTILE,
        GAMMA=GAMMA,
        GAIN=GAIN,
        SPLIT_MODE="random_case",
        VAL_RATIO=VAL_RATIO,
        TEST_RATIO=TEST_RATIO,
        TRAIN_CASES=TRAIN_CASES,
        VAL_CASES=VAL_CASES,
        TEST_CASES=TEST_CASES,
        LOSS_TYPE=LOSS_TYPE,
    )

    # ==========================================
    # 2. LOAD DATA
    # ==========================================
    print("\n>>> Loading data...")
    data = load_and_encode(cfg)

    # ---- No-validation mode: fold would-be-val cases into the training set ----
    # pre_encoder's stratified split uses max(1, round(val_ratio * n)) per
    # stratum, so VAL_RATIO=0 still produces ~1 val case per stratum. Patch the
    # outputs here so downstream code (dataset_split.txt, train/val Subsets,
    # stateful case_idxs) all see val as part of train.
    no_val = (VAL_RATIO == 0.0)
    if no_val:
        n_val_before = sum(1 for v in data['case_split_map'].values() if v == 'val')
        if n_val_before > 0:
            print(f"[NO_VAL] VAL_RATIO=0 — folding {n_val_before} val cases "
                  f"into training set; validation phase disabled.")
            data['case_split_map'] = {
                k: ('train' if v == 'val' else v)
                for k, v in data['case_split_map'].items()
            }
            if 'split_indices' in data and data['split_indices'] is not None:
                _si = data['split_indices']
                if _si.get('val'):
                    _si['train'] = sorted(list(_si['train']) + list(_si['val']))
                    _si['val'] = []

    # Write a human-readable dataset_split.txt alongside the config snapshot
    # so the per-split case listing is auditable without re-running the
    # pre-encoder.
    _forced_keys = set()
    for _forced_list in (TRAIN_CASES, VAL_CASES, TEST_CASES):
        _forced_keys.update(
            _parse_forced_cases(_forced_list, data['rd_params'], '_')
        )
    write_dataset_split_file(
        data['case_split_map'],
        data['rd_params'],
        _forced_keys,
        os.path.join(ckpt_dir, 'dataset_split.txt'),
    )
    print(f"Dataset split listing saved to: "
          f"{os.path.join(ckpt_dir, 'dataset_split.txt')}")

    # pre_encoder returns datasets with (spikes, cl_targets). The model is
    # decoder-free: it predicts Cl directly, so no vorticity reconstruction
    # target is needed. Keep compact dtypes (uint8 spikes) to save RAM; cast to
    # float32 per-batch in the training loop.
    if TRAINING_MODE == "stateful_tbptt":
        # Stateful path is case-level — works for both interpolated and original
        # datasets (the latter via per-case .npy files mirrored from .mat/.csv
        # inside pre_encoder).
        _case_paths      = data['case_paths']
        _norm_scale      = data['norm_stats']['norm_scale']
        _threshold       = data['threshold']
        _case_split_map  = data['case_split_map']
        del data
        gc.collect()
        _train_case_idxs = [
            i for i, (a, c, _, _) in enumerate(_case_paths)
            if _case_split_map.get((a, c)) == 'train'
        ]
        _val_case_idxs = [
            i for i, (a, c, _, _) in enumerate(_case_paths)
            if _case_split_map.get((a, c)) == 'val'
        ]
        if no_val:
            # case_split_map was already patched upstream, but be defensive.
            _train_case_idxs = sorted(set(_train_case_idxs + _val_case_idxs))
            _val_case_idxs = []
        print(f"[STATEFUL] Train cases: {len(_train_case_idxs)} | "
              f"Val cases: {len(_val_case_idxs)}"
              + ("  (validation disabled — VAL_RATIO=0)" if no_val else ""))
        train_ds = val_ds = None   # unused in this mode
        split_indices = None
        spikes = None
    elif DATASET_TYPE == "interpolated":
        # Streaming random_window path: train_dataset/val_dataset are
        # Subset instances over a StreamingSpikeDataset; no full spike tensors
        # live in RAM.
        train_ds = data['train_dataset']
        val_ds   = data['val_dataset']
        split_indices = data['split_indices']
        # No-val mode: merge val_ds indices into train_ds when they share an
        # underlying StreamingSpikeDataset (the usual case). If the structure
        # is unexpected, fall back to dropping val_ds — val sequences are then
        # not used, but training still runs.
        if no_val and val_ds is not None and len(val_ds) > 0:
            if (isinstance(train_ds, Subset) and isinstance(val_ds, Subset)
                    and train_ds.dataset is val_ds.dataset):
                merged_idx = sorted(list(train_ds.indices) + list(val_ds.indices))
                train_ds = Subset(train_ds.dataset, merged_idx)
            else:
                print("[NO_VAL] WARN: val_ds structure does not allow merging; "
                      "val sequences will be skipped, not folded into train.")
            val_ds = None
        spikes = None
        del data
        gc.collect()
    else:
        spikes     = data['spikes']                             # (N, T, C, Ny, Nx) uint8
        cl_targets = data['targets'].unsqueeze(-1)              # (N, T, 1) float32
        split_indices = data['split_indices']

        # Free everything we don't need — the data dict keeps refs to all tensors
        del data
        gc.collect()

        full_dataset = TensorDataset(spikes, cl_targets)
        train_ds = Subset(full_dataset, split_indices['train'])
        val_ds   = Subset(full_dataset, split_indices['val'])

    if TRAINING_MODE != "stateful_tbptt":
        pin = device.type == "cuda"
        _loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=pin)
        if NUM_WORKERS > 0:
            _loader_kwargs.update(
                persistent_workers=PERSISTENT_WORKERS,
                prefetch_factor=PREFETCH_FACTOR,
            )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True,
                                  **_loader_kwargs)
        if no_val:
            val_loader = None
        else:
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                    shuffle=False, drop_last=False,
                                    **_loader_kwargs)

        _val_len_str = "disabled" if no_val else f"{len(val_ds)} sequences"
        print(f"Train: {len(train_ds)} sequences | Val: {_val_len_str}")
        if DATASET_TYPE == "interpolated":
            x_probe, cl_probe = train_ds[0]
            print(f"Spikes:  per-item {tuple(x_probe.shape)} (streamed)")
            print(f"Cl:      per-item {tuple(cl_probe.shape)} (streamed)")
            assert x_probe.shape[0] == TIME_STEPS, (
                f"Expected spike seq length {TIME_STEPS}, got {x_probe.shape[0]}")
            assert cl_probe.shape[0] == TIME_STEPS, (
                f"Expected Cl target length {TIME_STEPS}, got {cl_probe.shape[0]}")
            del x_probe, cl_probe
        else:
            print(f"Spikes:  {tuple(spikes.shape)}")
            print(f"Cl:      {tuple(cl_targets.shape)}")
            assert spikes.shape[1] == TIME_STEPS, (
                f"Expected spike seq length {TIME_STEPS}, got {spikes.shape[1]}")
            assert cl_targets.shape[1] == TIME_STEPS, (
                f"Expected Cl target length {TIME_STEPS}, got {cl_targets.shape[1]}")
        print(f"[SANITY] x_seq: T={TIME_STEPS} ✓  cl: T={TIME_STEPS} ✓")

    # ==========================================
    # 3. MODEL
    # ==========================================
    model = _build_model().to(device)
    _print_model_training_defaults(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model.__class__.__name__} (N_z={N_Z}) | "
          f"Parameters: {num_params:,}")

    # ---- Fine-tune / resume: load the source weights into the fresh model ----
    if ft_weights is not None:
        state_dict = torch.load(ft_weights, map_location=device)
        # Checkpoints here are raw model.state_dict() dumps; tolerate a wrapped
        # {'state_dict': ...} just in case.
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"[FINE_TUNE] Weights in '{FINE_TUNE_WEIGHTS}' do not match the "
                f"current model architecture. Make sure N_Z, IN_CHANNELS and the "
                f"other architecture settings match the original run.\n"
                f"Original error: {e}")
        print(f"[FINE_TUNE] Loaded weights from {FINE_TUNE_WEIGHTS} "
              f"(resuming at epoch {start_epoch + 1}).")

    # ==========================================
    # 4. OPTIMIZER & SCHEDULER
    # ==========================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # Resume: fast-forward the cosine LR schedule to the resume epoch (the loop
    # steps the scheduler once per completed epoch) and carry over the prior loss
    # history so the saved loss_curves.pt stays continuous.
    init_train_losses, init_val_losses = [], []
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
        _lc_path = os.path.join(ckpt_dir, 'loss_curves.pt')
        if os.path.isfile(_lc_path):
            _prev = torch.load(_lc_path, map_location='cpu')
            init_train_losses = list(_prev.get('train_losses', []))[:start_epoch]
            init_val_losses   = list(_prev.get('val_losses', []))[:start_epoch]
            print(f"[FINE_TUNE] Carried over {len(init_train_losses)} prior "
                  f"train-loss entries; LR fast-forwarded to "
                  f"{optimizer.param_groups[0]['lr']:.2e}.")

    # ==========================================
    # 5. TRAINING LOOP
    # ==========================================
    if TRAINING_MODE == "stateful_tbptt":
        _run_stateful_tbptt(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            case_paths=_case_paths,
            train_case_idxs=_train_case_idxs,
            val_case_idxs=_val_case_idxs,
            norm_scale=_norm_scale,
            threshold=_threshold,
            cfg=cfg,
            device=device,
            best_model_path=best_model_path,
            final_model_path=final_model_path,
            ckpt_dir=ckpt_dir,
            no_val=no_val,
            start_epoch=start_epoch,
            init_train_losses=init_train_losses,
            init_val_losses=init_val_losses,
        )
        return

    best_val_loss = float('inf')
    epochs_since_best = 0
    early_stopped = False
    train_losses  = list(init_train_losses)
    val_losses    = list(init_val_losses)

    if start_epoch > 0:
        print(f"\n>>> Resuming training: epochs {start_epoch + 1}..{EPOCHS}\n")
    else:
        print(f"\n>>> Starting training for {EPOCHS} epochs...\n")
    t_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        t_epoch_start = time.time()
        # ---- Training ----
        model.train()
        epoch_train_loss = 0.0
        n_train          = 0
        train_comp_sum   = {}

        for batch in train_loader:
            x_batch, cl_batch = batch

            # Permute (B, T, ...) → (T, B, ...) and cast to float32 on device
            x_batch  = x_batch.permute(1, 0, 2, 3, 4).to(device=device, dtype=torch.float32, non_blocking=True)
            cl_batch = cl_batch.permute(1, 0, 2).to(device=device, dtype=torch.float32, non_blocking=True)

            # When BINS > 1, spike sequence is T*BINS long but Cl targets are T
            # long. Repeat each Cl value BINS times: (T, B, 1) → (T*BINS, B, 1)
            if BINS > 1:
                cl_batch = cl_batch.repeat_interleave(BINS, dim=0)

            optimizer.zero_grad(set_to_none=True)

            with _get_autocast_context(device):
                z_seq, cl_pred = model(x_batch)

            loss, components = combined_loss(cl_pred.float(), cl_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train += 1
            for k, v in components.items():
                train_comp_sum[k] = train_comp_sum.get(k, 0.0) + v

        avg_train            = epoch_train_loss / max(n_train, 1)
        avg_train_components = {k: v / max(n_train, 1) for k, v in train_comp_sum.items()}
        train_losses.append(avg_train)

        # ---- Validation ----
        if no_val:
            avg_val = float('nan')
            avg_val_components = {}
        else:
            model.eval()
            epoch_val_loss = 0.0
            n_val          = 0
            val_comp_sum   = {}

            with torch.inference_mode():
                for batch in val_loader:
                    x_batch, cl_batch = batch

                    x_batch  = x_batch.permute(1, 0, 2, 3, 4).to(device=device, dtype=torch.float32, non_blocking=True)
                    cl_batch = cl_batch.permute(1, 0, 2).to(device=device, dtype=torch.float32, non_blocking=True)

                    if BINS > 1:
                        cl_batch = cl_batch.repeat_interleave(BINS, dim=0)

                    with _get_autocast_context(device):
                        z_seq, cl_pred = model(x_batch)

                    loss, components = combined_loss(cl_pred.float(), cl_batch)
                    epoch_val_loss += loss.item()
                    n_val += 1
                    for k, v in components.items():
                        val_comp_sum[k] = val_comp_sum.get(k, 0.0) + v
                    if MONITOR_ENCODER:
                        last_z_mean = z_seq.mean().item()
                        last_z_std  = z_seq.std().item()
                        last_z_absmax = z_seq.abs().max().item()

            avg_val            = epoch_val_loss / max(n_val, 1)
            avg_val_components = {k: v / max(n_val, 1) for k, v in val_comp_sum.items()}
        val_losses.append(avg_val)
        scheduler.step()

        # ---- Logging ----
        lr_now        = optimizer.param_groups[0]['lr']
        elapsed       = time.time() - t_start
        epoch_elapsed = time.time() - t_epoch_start
        _val_str = "  no-val" if no_val else f"Val: {avg_val:.6f}"
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {avg_train:.6f} | {_val_str} | "
              f"LR: {lr_now:.2e} | "
              f"Epoch: {_fmt_time(epoch_elapsed)} | Total: {_fmt_time(elapsed)}")

        if MONITOR_ENCODER and not no_val:
            print(f"  z: mean={last_z_mean:+.4f} std={last_z_std:.4f} "
                  f"|max|={last_z_absmax:.4f}")

        train_comp_str = " | ".join(
            f"{k}: {v:.6f}" for k, v in avg_train_components.items()
        )
        print(f"  Train components: {train_comp_str}")
        if not no_val:
            val_comp_str = " | ".join(
                f"{k}: {v:.6f}" for k, v in avg_val_components.items()
            )
            print(f"  Val   components: {val_comp_str}")
        if device.type == "cuda" and (epoch == 0 or (epoch + 1) % 25 == 0):
            alloc = torch.cuda.memory_allocated() / 1024**2
            res   = torch.cuda.memory_reserved() / 1024**2
            print(f"  VRAM: {alloc:.0f} MB allocated / {res:.0f} MB reserved")

        # ---- Save best / periodic checkpoint ----
        if no_val:
            # No val loss to compare → save every SAVE_EVERY_N_EPOCHS epochs.
            if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
                ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> Periodic checkpoint saved: epoch_{epoch+1}.pth")
        else:
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_since_best = 0
                _save_state_dict_atomic(model.state_dict(), best_model_path)
                print(f"  -> Best model saved (val={best_val_loss:.6f})")
            else:
                epochs_since_best += 1
                if EARLY_STOP:
                    print(f"  -> No improvement for {epochs_since_best}/"
                          f"{EARLY_STOP_EPOCHS} epochs (best val={best_val_loss:.6f})")

            if (epoch + 1) % 25 == 0 and not KEEP_ONLY_BEST_AND_FINAL:
                ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), ckpt_path)

        # ---- Save loss curves after every epoch (allows test.py to run at any point) ----
        torch.save({'train_losses': train_losses, 'val_losses': val_losses},
                   os.path.join(ckpt_dir, 'loss_curves.pt'))

        # ---- Early stopping ----
        if not no_val and EARLY_STOP and epochs_since_best >= EARLY_STOP_EPOCHS:
            print(f"\n[EARLY STOP] Val loss has not improved for "
                  f"{EARLY_STOP_EPOCHS} epochs. Stopping at epoch {epoch+1}/"
                  f"{EPOCHS} (best val={best_val_loss:.6f}).")
            early_stopped = True
            break

    # ==========================================
    # 6. SAVE FINAL MODEL
    # ==========================================
    torch.save(model.state_dict(), final_model_path)
    total_time = time.time() - t_start
    finish_reason = "early-stopped" if early_stopped else "complete"
    print(f"\nTraining {finish_reason} in {total_time/60:.1f} min "
          f"({len(train_losses)}/{EPOCHS} epochs)")
    if no_val:
        print(f"Validation disabled — periodic checkpoints saved every "
              f"{SAVE_EVERY_N_EPOCHS} epochs in {ckpt_dir}/")
    else:
        print(f"Best val loss: {best_val_loss:.6f}")
        print(f"Best model:    {best_model_path}")
    print(f"Final model:   {final_model_path}")


if __name__ == "__main__":
    train()
