"""
train_energy_matched.py — Trainer for the energy-matched non-spiking CNN
(models/energy_matched.py, EnergyMatchedCNN).

Identical training protocol to the parameter-matched CNN trainer
(train_parameter_matched.py) — same frame-level regression, encounter-level
train/test split (test frames never seen), Lift-MSE loss, AdamW, initial LR,
cosine schedule, epoch count, bf16 AMP, large batch, same SEEDS list (currently
1 seed, for a quick approximate result). The ONLY difference is the model: a
reduced-width dense CNN whose estimated per-step inference energy matches the
reference SNN's budget (see energy_matched.py).

The generic data pipeline (frame dataset, split, seeding, AMP context) is reused
from train_parameter_matched so the two runs share identical training conditions.

Each seed writes checkpoints + an efficiency-compatible config_snapshot.txt to
    training/Energy_matched_training/CNN_EnergyMatched_Nz{N_Z}_seed{S}_<timestamp>/
which efficiency_metrics.py / test.py consume directly (MODEL_TYPE="cnn").

Usage:
    python train_energy_matched.py
"""

import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'models'))

# Reuse the generic, model-agnostic training machinery + shared config so both
# CNN baselines are trained under identical conditions.
from train_parameter_matched import (
    FrameDataset, build_split, _seed_everything, _amp_ctx,
    N_Z, IN_CHANNELS, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP,
    USE_AMP, AMP_DTYPE, NUM_WORKERS, SAVE_EVERY_N_EPOCHS, MAX_BATCHES_PER_EPOCH,
    SEEDS, DATASET, ANGLES, SPLIT_MODE, VAL_RATIO, TEST_RATIO,
    TRAIN_CASES, VAL_CASES, TEST_CASES,
)
from energy_matched import EnergyMatchedCNN, DEFAULT_C1, DEFAULT_C2

# Results of running this script are saved under training/Energy_matched_training/
# (a sibling of Parameter_matched_training/ and SNN_training/, not shared with them).
CHECKPOINTS_ROOT = os.path.join(_HERE, '..', 'training', 'Energy_matched_training')

TRAINING_DESCRIPTION = ("Energy-matched non-spiking CNN — raw continuous "
                        "vorticity, per-frame Cl MSE, widths reduced to ~SNN "
                        f"per-step energy budget (c1={DEFAULT_C1}, c2={DEFAULT_C2})")
C1 = DEFAULT_C1
C2 = DEFAULT_C2

NUM_WORKERS  = 12
# Batches each worker fetches ahead of the GPU (only used when NUM_WORKERS > 0;
# default is 2). Higher keeps the GPU better fed on this I/O-bound frame loader,
# but costs RAM: ~NUM_WORKERS * PREFETCH_FACTOR batches are buffered, and one
# batch is BATCH_SIZE * 120 * 240 * 4 bytes (~118 MB at batch 1024). Lower it if
# host RAM gets tight.
PREFETCH_FACTOR = 4
BATCH_SIZE = 2048

# ==========================================
# CONFIG SNAPSHOT  (efficiency_metrics / test compatible)
# ==========================================
def write_config_snapshot(path, seed):
    lines = [
        "CONFIGURATION SNAPSHOT",
        "=" * 50,
        f"TRAINING_DESCRIPTION    = {TRAINING_DESCRIPTION}",
        "",
        "MODEL",
        "-" * 30,
        "MODEL_TYPE              = cnn_energy_matched",
        f"N_Z                     = {N_Z}",
        f"IN_CHANNELS             = {IN_CHANNELS}",
        f"C1                      = {C1}",
        f"C2                      = {C2}",
        "BETA_INIT               = 0.0",
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
def train_one_seed(seed, train_paths, device, epochs=EPOCHS, max_batches=None):
    _seed_everything(seed)

    timestamp = datetime.now().strftime("%B_%d_%Y_%Hh_%Mm")
    ckpt_dir = os.path.join(
        CHECKPOINTS_ROOT, f"CNN_EnergyMatched_Nz{N_Z}_seed{seed}_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    write_config_snapshot(os.path.join(ckpt_dir, 'config_snapshot.txt'), seed)
    print(f"\n=== seed {seed} -> {ckpt_dir} ===")

    dataset = FrameDataset(train_paths)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0), drop_last=False,
        # prefetch_factor is only valid with worker processes.
        prefetch_factor=(PREFETCH_FACTOR if NUM_WORKERS > 0 else None),
    )
    n_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    print(f"[DATA] {len(dataset):,} training frames -> ~{n_batches} batches/epoch "
          f"(batch {BATCH_SIZE})")

    model = EnergyMatchedCNN(c1=C1, c2=C2, n_z=N_Z, in_channels=IN_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] EnergyMatchedCNN(c1={C1}, c2={C2}) params: {n_params:,}")

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
