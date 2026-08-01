"""
train_parameter_matched.py — Trainer for the parameter-matched non-spiking CNN
(models/parameter_matched.py, ParameterMatchedCNN).

Unlike train_SNN.py (spiking, stateful TBPTT over full encounters), this is a
plain frame-level regressor: each continuous vorticity snapshot is an independent
training sample whose target is that frame's instantaneous Cl. Frames are shuffled
across the training encounters, so large batches and full data-parallelism are
available (no membrane state, no sequence unrolling).

To keep the comparison fair, it reuses the reference SNN's:
  * encounter-level train/test split (same DATASET, ANGLES, SPLIT_MODE, pinned
    TEST_CASES, VAL_RATIO=0, TEST_RATIO) — test frames are never seen;
  * Lift-MSE loss, AdamW, initial LR, cosine LR schedule, epoch count, bf16 AMP.
The batch size is larger (no temporal state to store). SEEDS controls how many
independent runs are trained (currently 1, for a quick approximate result; add
more seeds later for a mean +/- std comparison if higher confidence is needed).

Each seed writes checkpoints + an efficiency-compatible config_snapshot.txt to
    training/Parameter_matched_training/CNN_ParamMatched_Nz{N_Z}_seed{S}_<timestamp>/
which efficiency_metrics.py consumes directly (MODEL_TYPE="cnn").

Usage:
    python train_parameter_matched.py
"""

import os
import sys
import time
import random
import bisect
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'models'))

from pre_encoder import (
    load_rd_params,
    list_interpolated_cases,
    get_case_nt,
    _pre_compute_case_split_map,
)
from parameter_matched import ParameterMatchedCNN


# ==========================================
# CONFIGURATION  (data/split mirror the reference SNN baseline)
# ==========================================
TRAINING_DESCRIPTION = ("Parameter-matched non-spiking CNN (SNN-analog) — raw "
                        "continuous vorticity, per-frame Cl MSE, no recurrence")

# --- Architecture ---
N_Z          = 16
IN_CHANNELS  = 1                       # one signed continuous omega channel

# --- Dataset / split (identical to the SNN so the comparison is controlled) ---
DATASET      = "Dataset_interpolated_cubic_spline_N4"
ANGLES       = [20, 30, 40, 50, 60]
SPLIT_MODE   = "random_case"
VAL_RATIO    = 0.0
TEST_RATIO   = 0.15
TRAIN_CASES  = []
VAL_CASES    = []
TEST_CASES   = [
    "AoA20_RD5_G_n2.2", "AoA20_RD8_G_0.8", "AoA20_RD17_G_2.8", "AoA20_RD29_G_1.2",
    "AoA30_RD2_G_3.4", "AoA30_RD4_G_n1.2", "AoA30_RD11_G_1.2", "AoA30_RD13_G_n3.6",
    "AoA30_RD22_G_n0.4", "AoA40_RD1_G_n2.2", "AoA40_RD6_G_1.8", "AoA40_RD12_G_n2.0",
    "AoA40_RD16_G_4.0", "AoA40_RD19_G_0.8", "AoA50_RD4_G_n3.0", "AoA50_RD7_G_0.4",
    "AoA50_RD8_G_n2.0", "AoA50_RD18_G_2.2", "AoA50_RD21_G_n4.0", "AoA60_RD6_G_n3.0",
    "AoA60_RD9_G_n3.0", "AoA60_RD10_G_n0.4", "AoA60_RD24_G_n1.8", "AoA60_RD26_G_2.0",
]

# --- Training hyperparameters (match the SNN except batch size) ---
SEEDS        = [0]                     # Amount of times the training is done. More seeds means more robustness in the results but longer overall training time. 
BATCH_SIZE   = 1024                    # larger: no temporal state / unrolling
                                        # (4x the original 256 — 4.9/24 GB used
                                        # at 256, so this leaves headroom; bump
                                        # further if nvidia-smi still shows room)
EPOCHS       = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
USE_AMP      = True
AMP_DTYPE    = "bfloat16"
NUM_WORKERS  = 4
SAVE_EVERY_N_EPOCHS = 10

# Smoke-test knobs (leave as-is for a real run). MAX_BATCHES_PER_EPOCH caps the
# batches per epoch; None = full epoch.
MAX_BATCHES_PER_EPOCH = None


# ==========================================
# PATHS  (same resolution logic as train_SNN.py / test.py)
# ==========================================
def _find_datasets_root():
    env = os.environ.get('THESIS_DATA_ROOT')
    if env:
        return os.path.abspath(env)
    base = os.path.join(_HERE, '..')
    for name in ('Datasets', 'DATASETS', 'datasets'):
        cand = os.path.normpath(os.path.join(base, name))
        if os.path.isdir(cand):
            return cand
    return os.path.normpath(base)


DATASETS_ROOT    = _find_datasets_root()
DATASET_PATH     = os.path.join(DATASETS_ROOT, DATASET)
INTERP_ROOT      = DATASET_PATH
RD_PARAMS_PATH   = os.path.join(DATASETS_ROOT, 'RD_list_FT2023.xlsx')
CHECKPOINTS_ROOT = os.path.join(_HERE, '..', 'training', 'Parameter_matched_training')


# ==========================================
# FRAME-LEVEL DATASET
# ==========================================
class FrameDataset(Dataset):
    """Every frame of the training encounters is an independent (omega, Cl) pair.

    A global index maps to (case, frame) via cumulative per-case frame counts.
    Vorticity/lift arrays are memory-mapped and opened lazily per worker so no
    case is ever fully loaded into RAM. Vorticity is fed RAW (no scaling/clip).
    """

    def __init__(self, case_paths):
        # case_paths: list of (angle, case_id, vort_npy, cl_npy)
        self.cases = case_paths
        self.nts = [get_case_nt((a, c, vp, cp)) for (a, c, vp, cp) in case_paths]
        self.cum = np.cumsum([0] + self.nts).tolist()   # len = n_cases + 1
        self.total = self.cum[-1]
        self._vort_mm = {}      # case_idx -> mmap (per-worker cache)
        self._cl_arr = {}       # case_idx -> np array

    def __len__(self):
        return self.total

    def _open(self, ci):
        if ci not in self._vort_mm:
            _, _, vp, cp = self.cases[ci]
            self._vort_mm[ci] = np.load(vp, mmap_mode='r')
            self._cl_arr[ci] = np.load(cp).astype(np.float32).ravel()
        return self._vort_mm[ci], self._cl_arr[ci]

    def __getitem__(self, idx):
        # Locate the case containing global frame idx.
        ci = bisect.bisect_right(self.cum, idx) - 1
        frame = idx - self.cum[ci]
        vort_mm, cl_arr = self._open(ci)
        omega = np.array(vort_mm[frame], dtype=np.float32)      # (H, W) raw
        x = torch.from_numpy(omega).unsqueeze(0)               # (1, H, W)
        y = torch.tensor([cl_arr[frame]], dtype=torch.float32)  # (1,)
        return x, y


# ==========================================
# SPLIT + DATA
# ==========================================
def build_split():
    """Resolve train/test encounters exactly like the SNN (pinned TEST_CASES)."""
    cfg = SimpleNamespace(
        SPLIT_MODE=SPLIT_MODE, VAL_RATIO=VAL_RATIO, TEST_RATIO=TEST_RATIO,
        TRAIN_CASES=TRAIN_CASES, VAL_CASES=VAL_CASES, TEST_CASES=TEST_CASES,
        ANGLES=ANGLES,
    )
    rd_params = load_rd_params(RD_PARAMS_PATH, angles=ANGLES)
    split = _pre_compute_case_split_map(cfg, rd_params)
    case_paths = list_interpolated_cases(INTERP_ROOT, ANGLES)
    by_key = {(a, c): (a, c, vp, cp) for a, c, vp, cp in case_paths}

    # VAL_RATIO=0 -> fold any 'val' into train (matches the SNN's no-val mode).
    train_keys = [k for k, v in split.items() if v != 'test']
    test_keys = [k for k, v in split.items() if v == 'test']
    train_paths = [by_key[k] for k in train_keys if k in by_key]
    print(f"[SPLIT] {len(train_paths)} train / {len(test_keys)} test encounters "
          f"(test frames excluded from training).")
    return train_paths, test_keys


# ==========================================
# CONFIG SNAPSHOT  (efficiency_metrics-compatible)
# ==========================================
def write_config_snapshot(path, seed):
    lines = [
        "CONFIGURATION SNAPSHOT",
        "=" * 50,
        f"TRAINING_DESCRIPTION    = {TRAINING_DESCRIPTION}",
        "",
        "MODEL",
        "-" * 30,
        "MODEL_TYPE              = cnn_param_matched",
        f"N_Z                     = {N_Z}",
        f"IN_CHANNELS             = {IN_CHANNELS}",
        "BETA_INIT               = 0.0",
        # ENCODING=none: efficiency_metrics feeds raw frames for the CNN, but the
        # config parser expects the key to be present.
        "ENCODING                = none",
        "",
        "DATASET",
        "-" * 30,
        f"DATASET                 = {DATASET}",
        "TIME_STEPS              = 1",
        "WARMUP_STEPS            = 0",
        "STRIDE                  = 1",
        f"ANGLES                  = {ANGLES}",
        "",
        "SPLIT",
        "-" * 30,
        f"SPLIT_MODE              = {SPLIT_MODE}",
        f"VAL_RATIO               = {VAL_RATIO}",
        f"TEST_RATIO              = {TEST_RATIO}",
        f"TRAIN_CASES             = {TRAIN_CASES}",
        f"VAL_CASES               = {VAL_CASES}",
        f"TEST_CASES              = {TEST_CASES}",
        "",
        "TRAINING",
        "-" * 30,
        f"SEED                    = {seed}",
        f"BATCH_SIZE              = {BATCH_SIZE}",
        f"EPOCHS                  = {EPOCHS}",
        f"LR                      = {LR}",
        f"WEIGHT_DECAY            = {WEIGHT_DECAY}",
        f"GRAD_CLIP               = {GRAD_CLIP}",
        f"USE_AMP                 = {USE_AMP}",
        f"AMP_DTYPE               = {AMP_DTYPE}",
        "LOSS_TYPE               = mse",
        "=" * 50,
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


# ==========================================
# TRAIN ONE SEED
# ==========================================
def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_ctx(device):
    if USE_AMP and device.type == 'cuda':
        dtype = torch.bfloat16 if AMP_DTYPE == 'bfloat16' else torch.float16
        return torch.autocast('cuda', dtype=dtype)
    from contextlib import nullcontext
    return nullcontext()


def train_one_seed(seed, train_paths, device, epochs=EPOCHS, max_batches=None):
    _seed_everything(seed)

    timestamp = datetime.now().strftime("%B_%d_%Y_%Hh_%Mm")
    ckpt_dir = os.path.join(
        CHECKPOINTS_ROOT, f"CNN_ParamMatched_Nz{N_Z}_seed{seed}_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    write_config_snapshot(os.path.join(ckpt_dir, 'config_snapshot.txt'), seed)
    print(f"\n=== seed {seed} -> {ckpt_dir} ===")

    dataset = FrameDataset(train_paths)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0), drop_last=False,
    )
    n_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    print(f"[DATA] {len(dataset):,} training frames -> ~{n_batches} batches/epoch "
          f"(batch {BATCH_SIZE})")

    model = ParameterMatchedCNN(n_z=N_Z, in_channels=IN_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] ParameterMatchedCNN params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    use_scaler = USE_AMP and device.type == 'cuda' and AMP_DTYPE == 'float16'
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    loss_fn = nn.MSELoss()

    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for bi, (x, y) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_ctx(device):
                pred = model(x)
                loss = loss_fn(pred, y)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            running += float(loss.item()) * x.size(0)
            seen += x.size(0)
        scheduler.step()

        avg = running / max(1, seen)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  epoch {epoch + 1:3d}/{epochs} | MSE {avg:.6f} | LR {lr_now:.2e} | "
              f"{time.time() - t_start:.0f}s")

        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pth'))

    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'final.pth'))
    print(f"[DONE] seed {seed} in {(time.time() - t_start) / 60:.1f} min -> {ckpt_dir}")
    return ckpt_dir


# ==========================================
# MAIN
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    train_paths, _test_keys = build_split()

    ckpt_dirs = []
    for seed in SEEDS:
        ckpt_dirs.append(train_one_seed(
            seed, train_paths, device,
            epochs=EPOCHS, max_batches=MAX_BATCHES_PER_EPOCH))

    print("\nAll seeds trained:")
    for d in ckpt_dirs:
        print(f"  {d}")


if __name__ == "__main__":
    main()
