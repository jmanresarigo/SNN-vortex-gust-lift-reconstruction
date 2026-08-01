"""
pre_encoder.py — Spike-encoding pipeline.

Pipeline:
  1. Load raw .mat vorticity fields ω(x, y, t) for each simulation case
  2. Load corresponding lift coefficient Cl(t) from CSV files
  3. Normalize vorticity fields
  4. Reshape into sequences of TIME_STEPS frames
  5. Encode vorticity into spikes (rate / delta / latency coding)
  6. Prepare lift targets aligned with spike sequences
  7. Cache everything to a .pt file
  8. Provide train / val / test splits

Encoding mode and parameters are set externally via a config namespace
(passed from train_SNN.py).
"""

import gzip
import os
import re
import gc
import glob
import numpy as np
import torch
import snntorch.spikegen as spikegen
from torch.utils.data import TensorDataset, Subset
import scipy.io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==========================================
# DEVICE DETECTION
# ==========================================
def get_device():
    """Detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==========================================
# RAW DATA LOADING
# ==========================================
def load_vorticity(mat_path):
    """
    Load a single vorticity .mat file.

    Returns:
        omega: (Nt, Ny, Nx) vorticity field time series
    """
    data = scipy.io.loadmat(mat_path)
    omega = data['omg_box']  # (Nt, Ny, Nx)
    return omega.astype(np.float32)


def load_lift(csv_path):
    """
    Load a single lift coefficient CSV file.

    Returns:
        cl: (Nt,) lift coefficient time series
    """
    df = pd.read_csv(csv_path, header=None)
    cl = df.values.flatten().astype(np.float32)
    return cl


def load_rd_params(rd_params_path, angles=None):
    """
    Load gust parameters (G, R, y) from RD_list_FT2023.xlsx.

    The xlsx has one sheet per angle (a20, a30, ..., a60). Each data row
    contains: [RD_index, G, R, y]. Rows where column 0 is not a valid
    integer in 1–30 are skipped (header or metadata rows).

    The base case (no gust disturbance) is assigned G=R=y=0.0 automatically
    since it does not appear in the xlsx.

    Args:
        rd_params_path: path to RD_list_FT2023.xlsx
        angles: list of angles to load. If None, loads all five.

    Returns:
        rd_params: nested dict  rd_params[angle][case_id] = {'G': ..., 'R': ..., 'y': ...}
                   e.g.  rd_params[20]['RD5'] = {'G': -2.2, 'R': 0.75, 'y': 0.0}
    """
    if angles is None:
        angles = [20, 30, 40, 50, 60]

    sheet_map = {20: 'a20', 30: 'a30', 40: 'a40', 50: 'a50', 60: 'a60'}
    rd_params = {}

    for angle in angles:
        rd_params[angle] = {'base': {'G': 0.0, 'R': 0.0, 'y': 0.0}}

        sheet = sheet_map[angle]
        df = pd.read_excel(rd_params_path, sheet_name=sheet, header=None)

        for _, row in df.iterrows():
            try:
                rd_idx = int(row.iloc[0])
                G = float(row.iloc[1])
                R = float(row.iloc[2])
                y = float(row.iloc[3])
                if 1 <= rd_idx <= 30:
                    rd_params[angle][f'RD{rd_idx}'] = {'G': G, 'R': R, 'y': y}
            except (ValueError, TypeError):
                continue  # skip header / metadata rows

        n_loaded = len(rd_params[angle]) - 1  # exclude base
        print(f"[PARAMS] AoA={angle}°: 1 base case + {n_loaded} RD cases loaded")

    return rd_params


def load_all_cases(vorticity_dir, lift_dir, angles=None):
    """
    Load all simulation cases from the dataset.

    Args:
        vorticity_dir: path to vorticity/ folder containing vort_aXX/ subfolders
        lift_dir: path to lift/ folder containing lift_aXX/ subfolders
        angles: list of angles to load, e.g. [20, 30, 40, 50, 60].
                If None, loads all available angles.

    Returns:
        cases: list of dicts, each with keys:
            'omega': (Nt, Ny, Nx) vorticity field
            'cl': (Nt,) lift coefficient
            'angle': angle of attack (int)
            'case_id': case identifier string (e.g. "base", "RD1")
    """
    if angles is None:
        angles = [20, 30, 40, 50, 60]

    cases = []

    for angle in angles:
        vort_folder = os.path.join(vorticity_dir, f"vort_a{angle}")
        lift_folder = os.path.join(lift_dir, f"lift_a{angle}")

        if not os.path.isdir(vort_folder):
            print(f"[WARN] Vorticity folder not found: {vort_folder}")
            continue

        mat_files = sorted(glob.glob(os.path.join(vort_folder, f"Vort_AoA{angle}_*.mat")))

        for mat_path in mat_files:
            fname = os.path.basename(mat_path)
            # Extract case id: "Vort_AoA20_RD1.mat" -> "RD1", "Vort_AoA20_base.mat" -> "base"
            case_id = fname.replace(f"Vort_AoA{angle}_", "").replace(".mat", "")

            csv_path = os.path.join(lift_folder, f"Lift_AoA{angle}_{case_id}.csv")
            if not os.path.exists(csv_path):
                print(f"[WARN] Lift file not found for {fname}, skipping.")
                continue

            omega = load_vorticity(mat_path)
            cl = load_lift(csv_path)

            # Verify time alignment
            if omega.shape[0] != cl.shape[0]:
                print(f"[WARN] Time mismatch for AoA{angle}_{case_id}: "
                      f"omega={omega.shape[0]}, cl={cl.shape[0]}. Skipping.")
                continue

            cases.append({
                'omega': omega,
                'cl': cl,
                'angle': angle,
                'case_id': case_id,
            })

        print(f"[DATA] AoA={angle}°: loaded {sum(1 for c in cases if c['angle'] == angle)} cases")

    print(f"[DATA] Total cases loaded: {len(cases)}")
    return cases


# ==========================================
# NORMALIZATION
# ==========================================
def normalize_vorticity(cases, percentile=99):
    """
    Compute global normalization statistics across all cases and normalize.

    When percentile < 100, uses robust (percentile-based) normalization:
    the Pxx value of |ω| across all cases sets the scale, and values above
    that threshold are clipped to ±1.  This prevents a few extreme vortex
    cores from compressing the rest of the field toward zero.

    Args:
        cases: list of case dicts (modified in-place)
        percentile: percentile of |ω| to use as normalization scale.
                    100 = old behaviour (global absmax, no clipping).

    Returns:
        norm_stats: dict with 'omega_absmax', 'norm_scale', 'percentile'
    """
    # True global maximum (always reported for reference)
    omega_absmax = max(np.abs(c['omega']).max() for c in cases)

    if percentile >= 100:
        # Legacy behaviour: normalise by global absmax (no clipping)
        norm_scale = omega_absmax
    else:
        # Robust normalization: sample-based percentile (memory-efficient)
        rng = np.random.RandomState(42)
        samples_per_case = 100_000
        samples = []
        for c in cases:
            flat = np.abs(c['omega']).ravel()
            if len(flat) > samples_per_case:
                idx = rng.choice(len(flat), samples_per_case, replace=False)
                samples.append(flat[idx])
            else:
                samples.append(flat)
        all_samples = np.concatenate(samples)
        norm_scale = float(np.percentile(all_samples, percentile))
        del samples, all_samples

    for c in cases:
        c['omega'] = np.clip(c['omega'] / (norm_scale + 1e-8), -1.0, 1.0)

    print(f"[NORM] Global |ω|_max = {omega_absmax:.4f}")
    if percentile < 100:
        print(f"[NORM] P{percentile} |ω| = {norm_scale:.4f} (normalization scale)")
        print(f"[NORM] Clipping ratio: {omega_absmax / (norm_scale + 1e-8):.2f}x above P{percentile}")
    return {'omega_absmax': float(omega_absmax), 'norm_scale': float(norm_scale),
            'percentile': percentile}


# ==========================================
# SEQUENCE CREATION
# ==========================================
def create_sequences(cases, time_steps, stride=None, warmup_steps=0):
    """
    Convert each simulation case into a set of fixed-length sequences,
    then stack all sequences from all cases into a single batch.

    Each simulation case contains a full time series of Nt=1190 frames.
    The network cannot consume arbitrary-length sequences, so this function
    chops each case into windows of exactly TIME_STEPS frames.

    When stride < time_steps, windows overlap. This increases the number
    of training sequences without requiring additional simulation data.

    When warmup_steps > 0, each vorticity window is extended by warmup_steps
    frames before the supervised region. Only windows with enough preceding
    context (start >= warmup_steps) are created. The Cl target covers only
    the supervised TIME_STEPS region.

    For a single case with Nt=1190 frames, TIME_STEPS=100, stride=50:
        n_seqs  = (1190 - 100) // 50 + 1 = 22 sequences
        Each window: 100 consecutive frames, offset by 50 from the previous

    Args:
        cases: list of case dicts, each with:
                 'omega' (Nt, Ny, Nx) — normalized vorticity field
                 'cl'    (Nt,)        — lift coefficient time series
                 'angle'   int        — angle of attack
                 'case_id' str        — e.g. "base", "RD1"
        time_steps: number of supervised frames per sequence (T)
        stride: step between consecutive windows (default: time_steps = no overlap)
        warmup_steps: extra preceding frames prepended to each window for LIF
                      membrane warm-up. These frames are not supervised.

    Returns:
        omega_seqs  : (N_total, W+T, Ny, Nx) vorticity sequence tensor
        cl_seqs     : (N_total, T)             lift sequence tensor (supervised only)
        case_labels : list of length N_total with (angle, case_id) tuples,
                      one per sequence, for traceability back to the source
                      simulation
    """
    if stride is None:
        stride = time_steps

    omega_list = []
    cl_list = []
    case_labels = []

    for i, c in enumerate(cases):
        nt = c['omega'].shape[0]
        min_frames = warmup_steps + time_steps

        if nt < min_frames:
            print(f"[WARN] AoA{c['angle']}_{c['case_id']}: "
                  f"only {nt} frames < {min_frames} (warmup+TIME_STEPS), skipping.")
            continue

        # Sliding window: start positions (first supervised frame, always >= warmup_steps)
        starts = list(range(warmup_steps, nt - time_steps + 1, stride))
        n_seqs = len(starts)

        omega_windows = []
        cl_windows = []
        for t0 in starts:
            omega_windows.append(c['omega'][t0 - warmup_steps : t0 + time_steps])
            cl_windows.append(c['cl'][t0 : t0 + time_steps])

        omega = np.stack(omega_windows, axis=0)  # (n_seqs, W+T, Ny, Nx)
        cl = np.stack(cl_windows, axis=0)         # (n_seqs, T)

        omega_list.append(torch.tensor(omega, dtype=torch.float16))
        cl_list.append(torch.tensor(cl, dtype=torch.float32))
        case_labels.extend([(c['angle'], c['case_id'])] * n_seqs)

        # Free raw numpy arrays immediately — they can be large
        c['omega'] = None
        c['cl'] = None

    del cases
    gc.collect()

    omega_seqs = torch.cat(omega_list, dim=0)  # (N_total, W+T, Ny, Nx)
    del omega_list
    cl_seqs = torch.cat(cl_list, dim=0)        # (N_total, T)
    del cl_list
    gc.collect()

    total_input = warmup_steps + time_steps
    overlap_pct = (1 - stride / time_steps) * 100 if stride < time_steps else 0
    warmup_info = f", warmup={warmup_steps}" if warmup_steps > 0 else ""
    print(f"[SEQ] Created {omega_seqs.shape[0]} sequences of {time_steps} supervised steps "
          f"(input={total_input}{warmup_info}, stride={stride}"
          + (f", {overlap_pct:.0f}% overlap)" if overlap_pct > 0 else ")"))
    print(f"[SEQ] Vorticity shape: {tuple(omega_seqs.shape)}")
    print(f"[SEQ] Lift shape: {tuple(cl_seqs.shape)}")

    return omega_seqs, cl_seqs, case_labels


# ==========================================
# SPIKE ENCODING
# ==========================================
def _delta_encode(field, threshold, chunk_size=10):
    """
    Delta modulation encoding for vorticity sequences.

    Args:
        field: (N_samples, T, Ny, Nx) normalized vorticity tensor
        threshold: spike firing threshold
        chunk_size: samples per processing chunk (smaller = less peak RAM)
    Returns:
        spikes: (N_samples, T, 2, Ny, Nx) uint8 — channels [positive, negative]
    """
    n_samples = field.shape[0]
    chunks = []

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = field[start:end].float()  # float32 for spikegen
        # spikegen.delta expects (T, Batch, ...) input
        spike_raw = spikegen.delta(
            chunk.permute(1, 0, 2, 3),
            threshold=threshold, padding=True, off_spike=True
        )
        del chunk
        spk_pos = torch.clamp(spike_raw, min=0, max=1)
        spk_neg = torch.abs(torch.clamp(spike_raw, min=-1, max=0))
        del spike_raw
        # Stack as (T, Batch, 2, Ny, Nx) then permute to (Batch, T, 2, Ny, Nx)
        chunk_spk = torch.stack([spk_pos, spk_neg], dim=2).permute(1, 0, 2, 3, 4)
        del spk_pos, spk_neg
        chunks.append(chunk_spk.to(torch.uint8).cpu())
        del chunk_spk
        gc.collect()

    return torch.cat(chunks, dim=0)


def _rate_encode(field, bins=1, chunk_size=10, gamma=1.0, gain=1.0):
    """
    Poisson rate coding for vorticity sequences.
    Normalized magnitude --> spike probability. Positive and negative split.

    An amplified transfer function ``p = min(1, gain · |x|^gamma)`` replaces
    the plain ``p = |x|`` mapping.  With gamma < 1, moderate vorticity values
    are boosted relative to peaks; with gain > 1, the overall firing rate
    increases.  Both together move the encoding from the ~1 % regime (too
    sparse for convolutions) into the 15–25 % range.

    Args:
        field: (N_samples, T, Ny, Nx) normalized vorticity tensor (values in [-1, 1])
        bins: spike repetitions per original timestep
        chunk_size: samples per processing chunk (smaller = less peak RAM)
        gamma: power-law exponent (< 1 boosts weak signals)
        gain: multiplicative firing-rate amplifier (> 1 increases activity)
    Returns:
        spikes: (N_samples, T*bins, 2, Ny, Nx) uint8 — channels [positive, negative]
    """
    n_samples = field.shape[0]
    abs_max = field.abs().max().item()
    scale = 1.0 / (abs_max + 1e-8)

    chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = (field[start:end].float()) * scale  # float32 for bernoulli

        u_pos = torch.clamp(chunk, min=0)
        u_neg = torch.clamp(-chunk, min=0)
        del chunk

        # Amplified mapping: p = min(1, gain · |x|^gamma)
        if gamma != 1.0:
            u_pos = u_pos.pow(gamma)
            u_neg = u_neg.pow(gamma)
        if gain != 1.0:
            u_pos = torch.clamp(u_pos * gain, max=1.0)
            u_neg = torch.clamp(u_neg * gain, max=1.0)

        if bins > 1:
            u_pos = u_pos.repeat_interleave(bins, dim=1)
            u_neg = u_neg.repeat_interleave(bins, dim=1)

        spk_pos = torch.bernoulli(u_pos)
        spk_neg = torch.bernoulli(u_neg)
        del u_pos, u_neg

        chunk_spk = torch.stack([spk_pos, spk_neg], dim=2).to(torch.uint8).cpu()
        chunks.append(chunk_spk)
        del spk_pos, spk_neg, chunk_spk
        gc.collect()

    return torch.cat(chunks, dim=0)


def _latency_encode(field, chunk_size=10):
    """
    Latency (time-to-first-spike) encoding for vorticity sequences.
    Higher vorticity magnitude --> higher spike probability (earlier spike).

    Args:
        field: (N_samples, T, Ny, Nx) normalized vorticity tensor
        chunk_size: samples per processing chunk (smaller = less peak RAM)
    Returns:
        spikes: (N_samples, T, 2, Ny, Nx) uint8 — channels [positive, negative]
    """
    n_samples = field.shape[0]
    abs_max = field.abs().max().item()
    scale = 1.0 / (abs_max + 1e-8)

    chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = (field[start:end].float()) * scale  # float32 for bernoulli

        prob_pos = torch.clamp(chunk, min=0)
        prob_neg = torch.clamp(-chunk, min=0)
        del chunk

        spk_pos = torch.bernoulli(prob_pos)
        spk_neg = torch.bernoulli(prob_neg)
        del prob_pos, prob_neg

        chunk_spk = torch.stack([spk_pos, spk_neg], dim=2).to(torch.uint8).cpu()
        chunks.append(chunk_spk)
        del spk_pos, spk_neg, chunk_spk
        gc.collect()

    return torch.cat(chunks, dim=0)


def encode_spikes(field, mode, threshold=None, bins=1, chunk_size=10,
                   gamma=1.0, gain=1.0):
    """Dispatch to the selected encoding method."""
    if mode == 'delta':
        if threshold is None:
            raise ValueError("Delta encoding requires a threshold value.")
        return _delta_encode(field, threshold, chunk_size)
    elif mode == 'rate':
        return _rate_encode(field, bins, chunk_size, gamma=gamma, gain=gain)
    elif mode == 'latency':
        return _latency_encode(field, chunk_size)
    else:
        raise ValueError(f"Unknown encoding '{mode}'. Use 'rate', 'delta', or 'latency'.")


# ==========================================
# FULL-CASE DELTA ENCODING  (encode first, window later)
# ==========================================
def _delta_encode_full_case(omega_norm, threshold, zero_first_frame=True):
    """
    Delta-encode a single case's full vorticity sequence.

    Encoding the full simulation before windowing avoids the artificial
    spike burst that padding=True creates at t=0 of every window. With
    this approach, the burst only occurs once at the start of the full
    simulation and can optionally be zeroed out.

    Args:
        omega_norm:       (Nt, Ny, Nx) normalized vorticity (float32)
        threshold:        delta spike threshold
        zero_first_frame: if True, zero out spikes at t=0 to suppress
                          the artificial burst from the zero-padding baseline

    Returns:
        spikes: (Nt, 2, Ny, Nx) uint8 tensor [positive, negative channels]
    """
    # spikegen.delta expects (T, Batch, ...) input
    field = torch.tensor(omega_norm).float().unsqueeze(1)   # (Nt, 1, Ny, Nx)

    spike_raw = spikegen.delta(field, threshold=threshold,
                               padding=True, off_spike=True)
    del field
    # spike_raw: (Nt, 1, Ny, Nx)  values ∈ {-1, 0, +1}

    spk_pos = torch.clamp(spike_raw[:, 0], min=0, max=1)         # (Nt, Ny, Nx)
    spk_neg = torch.abs(torch.clamp(spike_raw[:, 0], min=-1, max=0))
    del spike_raw

    if zero_first_frame:
        spk_pos[0] = 0
        spk_neg[0] = 0

    spikes = torch.stack([spk_pos, spk_neg], dim=1).to(torch.uint8)
    del spk_pos, spk_neg
    return spikes                                                  # (Nt, 2, Ny, Nx)


def _rate_encode_full_case(omega_norm, gamma=1.0, gain=1.0):
    """
    Rate-encode a single case's full normalized vorticity sequence.

    Positive and negative vorticity are encoded into separate channels using
    Bernoulli spike generation with the amplified transfer function
    p = min(1, gain · |ω|^gamma).

    Args:
        omega_norm: (Nt, Ny, Nx) normalized vorticity in [-1, 1] (float32)
        gamma:      power-law exponent (< 1 boosts weak signals)
        gain:       multiplicative firing-rate amplifier

    Returns:
        spikes: (Nt, 2, Ny, Nx) uint8 [rate_pos, rate_neg channels]
    """
    field = torch.from_numpy(omega_norm).float()   # (Nt, Ny, Nx)

    prob_pos = torch.clamp(field, min=0.0)
    prob_neg = torch.clamp(-field, min=0.0)
    del field

    if gamma != 1.0:
        prob_pos = prob_pos.pow(gamma)
        prob_neg = prob_neg.pow(gamma)
    if gain != 1.0:
        prob_pos = torch.clamp(prob_pos * gain, max=1.0)
        prob_neg = torch.clamp(prob_neg * gain, max=1.0)

    spk_pos = torch.bernoulli(prob_pos)
    spk_neg = torch.bernoulli(prob_neg)
    del prob_pos, prob_neg

    spikes = torch.stack([spk_pos, spk_neg], dim=1).to(torch.uint8)
    del spk_pos, spk_neg
    return spikes   # (Nt, 2, Ny, Nx)


def _hybrid_encode_full_case(omega_norm, threshold, gamma=1.0, gain=1.0,
                              zero_first_frame=True):
    """
    Hybrid rate+delta encoding for a single case's full vorticity sequence.

    Produces 4 channels in order:
        ch 0: rate_pos  — positive vorticity, Bernoulli spikes
        ch 1: rate_neg  — negative vorticity, Bernoulli spikes
        ch 2: delta_pos — positive vorticity changes (Δω > threshold)
        ch 3: delta_neg — negative vorticity changes (Δω < -threshold)

    Rate channels capture absolute amplitude state; delta channels capture
    temporal transitions. The network learns how to combine both.

    Args:
        omega_norm:       (Nt, Ny, Nx) normalized vorticity in [-1, 1] (float32)
        threshold:        delta spike threshold (global, from training set)
        gamma:            rate encoding power-law exponent
        gain:             rate encoding firing-rate amplifier
        zero_first_frame: zero delta spikes at t=0 (suppress padding artifact)

    Returns:
        spikes: (Nt, 4, Ny, Nx) uint8
    """
    rate_spk  = _rate_encode_full_case(omega_norm, gamma=gamma, gain=gain)
    delta_spk = _delta_encode_full_case(omega_norm, threshold,
                                        zero_first_frame=zero_first_frame)
    return torch.cat([rate_spk, delta_spk], dim=1).to(torch.uint8)  # (Nt, 4, Ny, Nx)


def create_spike_sequences(cases, time_steps, stride=None, warmup_steps=0):
    """
    Window pre-encoded spike sequences and lift targets into fixed-length
    chunks, analogous to create_sequences() but operating on already-encoded
    spike tensors rather than raw vorticity.

    Used when encoding must happen before windowing (e.g., delta modulation)
    so that each case's full temporal sequence is encoded as a continuous
    signal, avoiding artificial spike bursts at window boundaries.

    When warmup_steps > 0, each spike window is extended by warmup_steps
    frames before the supervised region. Only windows with enough preceding
    context (start >= warmup_steps) are created. The Cl target covers only
    the supervised TIME_STEPS region.

    Args:
        cases: list of case dicts, each with:
            'spikes': (Nt, C, Ny, Nx) uint8 pre-encoded spikes
            'cl':     (Nt,) lift coefficient (numpy float32)
            'angle':  int
            'case_id': str
        time_steps: supervised window length (T)
        stride: step between consecutive windows (default: time_steps)
        warmup_steps: extra preceding frames prepended to each window for LIF
                      membrane warm-up. These frames are not supervised.

    Returns:
        spikes:      (N_total, W+T, C, Ny, Nx) uint8
        cl_seqs:     (N_total, T) float32   (supervised region only)
        case_labels: list of (angle, case_id) per sequence
    """
    if stride is None:
        stride = time_steps

    spike_list = []
    cl_list = []
    case_labels = []

    for c in cases:
        nt = c['spikes'].shape[0]
        min_frames = warmup_steps + time_steps
        if nt < min_frames:
            print(f"[WARN] AoA{c['angle']}_{c['case_id']}: "
                  f"only {nt} frames < {min_frames} (warmup+TIME_STEPS), skipping.")
            continue

        # First supervised frame is always >= warmup_steps so full context exists
        starts = list(range(warmup_steps, nt - time_steps + 1, stride))
        n_seqs = len(starts)

        for t0 in starts:
            spike_list.append(c['spikes'][t0 - warmup_steps : t0 + time_steps])
            cl_list.append(
                torch.tensor(c['cl'][t0 : t0 + time_steps], dtype=torch.float32))

        case_labels.extend([(c['angle'], c['case_id'])] * n_seqs)

        # Free per-case data immediately
        c['spikes'] = None
        c['cl'] = None

    del cases
    gc.collect()

    spikes = torch.stack(spike_list)     # (N_total, W+T, C, Ny, Nx)
    del spike_list
    cl_seqs = torch.stack(cl_list)       # (N_total, T)
    del cl_list
    gc.collect()

    total_input = warmup_steps + time_steps
    overlap_pct = (1 - stride / time_steps) * 100 if stride < time_steps else 0
    warmup_info = f", warmup={warmup_steps}" if warmup_steps > 0 else ""
    print(f"[SEQ] Created {spikes.shape[0]} spike sequences of {time_steps} supervised steps "
          f"(input={total_input}{warmup_info}, stride={stride}"
          + (f", {overlap_pct:.0f}% overlap)" if overlap_pct > 0 else ")"))
    print(f"[SEQ] Spike shape: {tuple(spikes.shape)}")
    print(f"[SEQ] Lift shape:  {tuple(cl_seqs.shape)}")

    return spikes, cl_seqs, case_labels


# ==========================================
# ENCODING VERIFICATION
# ==========================================
def verify_encoding(spk_all, omega_seqs, encoding, bins=1,
                    threshold=None, save_path=None):  # threshold kept for API compat
    """
    Print sparsity diagnostics and generate spike density plots.

    Args:
        spk_all: (N, T_spk, 2, Ny, Nx) spike tensor
        omega_seqs: (N, T, Ny, Nx) continuous vorticity tensor
        encoding: "rate" | "delta" | "latency"
        bins: rate coding bins
        threshold: delta modulation threshold
        save_path: if provided, save figure to this path instead of showing
    """
    print("\n" + "=" * 55)
    print(f"  ENCODING DIAGNOSTICS ({encoding.upper()})")
    print("=" * 55)

    N, T_spk, C, Ny, Nx = spk_all.shape
    spatial_temporal    = N * T_spk * Ny * Nx          # T×H×W per channel
    pos_spikes          = spk_all[:, :, 0].sum().item()
    neg_spikes          = spk_all[:, :, 1].sum().item()
    pos_rate            = pos_spikes / spatial_temporal * 100
    neg_rate            = neg_spikes / spatial_temporal * 100
    channel_mean_rate   = (pos_rate + neg_rate) / 2
    combined_pixel_rate = pos_rate + neg_rate

    print(f"  Spike tensor       : {tuple(spk_all.shape)}")
    print(f"  Total spikes       : {int(pos_spikes + neg_spikes):,}")
    print(f"  channel_mean_rate  : {channel_mean_rate:.2f}%")
    print(f"  combined_pixel_rate: {combined_pixel_rate:.2f}%")

    if encoding == 'delta':
        if channel_mean_rate < 0.5:
            print("  Warning: Too sparse — consider lowering THRESHOLD_FACTOR")
        elif channel_mean_rate > 15.0:
            print("  Warning: Too dense — consider raising THRESHOLD_FACTOR")
        else:
            print("  Healthy range (0.5%–15%)")

    # Fidelity check — chunked to avoid full float32 copies in RAM
    T_orig = omega_seqs.shape[1]
    print(f"\n  Fidelity check:")

    if encoding == 'delta':
        # For delta: need cumsum across time, process in sample chunks
        chunk = 5
        sum_err2, sum_tgt2 = 0.0, 0.0
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            sp = spk_all[s:e, :, 0].float()
            sn = spk_all[s:e, :, 1].float()
            net = (sp - sn).permute(1, 0, 2, 3)
            raw_recon = torch.cumsum(net, dim=0)
            recon_c = raw_recon - raw_recon.mean(dim=0, keepdim=True)
            tgt = omega_seqs[s:e].float().permute(1, 0, 2, 3)
            tgt_c = tgt - tgt.mean(dim=0, keepdim=True)
            sc = (tgt_c * recon_c).sum() / ((recon_c ** 2).sum() + 1e-8)
            diff = tgt_c - recon_c * sc
            sum_err2 += (diff ** 2).sum().item()
            sum_tgt2 += (tgt_c ** 2).sum().item()
            del sp, sn, net, raw_recon, recon_c, tgt, tgt_c, diff
        rel_e = (sum_err2 ** 0.5) / (sum_tgt2 ** 0.5 + 1e-8) * 100
        print(f"    [ω] rel_err={rel_e:.2f}%")
    else:
        chunk = 5
        sum_err2, sum_tgt2 = 0.0, 0.0
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            sp = spk_all[s:e, :, 0].float()
            sn = spk_all[s:e, :, 1].float()
            if bins > 1:
                sp = sp.view(e - s, T_orig, bins, Ny, Nx).mean(2)
                sn = sn.view(e - s, T_orig, bins, Ny, Nx).mean(2)
            recovered = sp - sn
            tgt = omega_seqs[s:e].float()
            diff = recovered - tgt
            sum_err2 += (diff ** 2).sum().item()
            sum_tgt2 += (tgt ** 2).sum().item()
            del sp, sn, recovered, tgt, diff
        rel_e = (sum_err2 ** 0.5) / (sum_tgt2 ** 0.5 + 1e-8) * 100
        print(f"    [ω] rel_err={rel_e:.2f}%")

    # Spike density maps
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for i, label in enumerate(["ω+", "ω-"]):
        density = spk_all[:, :, i].sum(dim=(0, 1)).cpu().numpy()
        axes[i].imshow(density, cmap='hot', origin='lower')
        axes[i].set_title(f"Density [{label}]")
        axes[i].axis('off')

    plt.suptitle(f"Spike Density Maps — {encoding.upper()} encoding", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    print("=" * 55)


# ==========================================
# CASE ASSIGNMENT HELPERS  (split logic only, no data tensors)
# ==========================================
def _case_tag(angle, case_id, rd_params):
    """Build a human-readable tag like 'AoA20_RD5_G_3.2' for a case."""
    g_val = 0.0
    if angle in rd_params and case_id in rd_params[angle]:
        g_val = rd_params[angle][case_id]['G']
    g_str = f"{abs(g_val):.1f}" if g_val >= 0 else f"n{abs(g_val):.1f}"
    return f"AoA{angle}_{case_id}_G_{g_str}"


_FORCED_CASE_RE = re.compile(
    r'^AoA(\d+)_(base|RD\d+)_G_(n?)(\d+(?:\.\d+)?)$'
)


def _parse_forced_cases(forced, rd_params, split_name):
    """Parse a list of case identifier strings into (angle, case_id) tuples.

    Format: "AoA{angle}_RD{idx}_G_{value}" (use 'n' prefix for negative G).
    Also accepts "AoA{angle}_base_G_0.0".

    Validates that each parsed case exists in rd_params and that the canonical
    tag produced by `_case_tag` matches the user-supplied string — this catches
    typos, wrong angles, and G-precision mismatches.
    """
    if not forced:
        return []
    parsed = []
    for raw in forced:
        m = _FORCED_CASE_RE.match(raw.strip())
        if not m:
            raise ValueError(
                f"[FORCED-CASE] {split_name}: cannot parse identifier "
                f"'{raw}'. Expected format 'AoA{{angle}}_RD{{idx}}_G_{{value}}' "
                f"(use 'n' prefix for negative G)."
            )
        angle = int(m.group(1))
        case_id = m.group(2)
        if angle not in rd_params:
            raise ValueError(
                f"[FORCED-CASE] {split_name}: angle {angle} (from '{raw}') "
                f"not present in rd_params (available: {sorted(rd_params)})."
            )
        if case_id not in rd_params[angle]:
            raise ValueError(
                f"[FORCED-CASE] {split_name}: case '{case_id}' (from '{raw}') "
                f"not present in rd_params[{angle}]."
            )
        expected = _case_tag(angle, case_id, rd_params)
        if expected != raw.strip():
            raise ValueError(
                f"[FORCED-CASE] {split_name}: identifier '{raw}' does not "
                f"match the canonical tag '{expected}'. Check the G value "
                f"(rd_params[{angle}]['{case_id}']['G'] = "
                f"{rd_params[angle][case_id]['G']})."
            )
        parsed.append((angle, case_id))
    return parsed


def _case_sort_key(case_id):
    """Natural sort: base first, then RD1, RD2, ..., RD10, ..."""
    if case_id == 'base':
        return (0, 0)
    m = re.match(r'^RD(\d+)$', case_id)
    if m:
        return (1, int(m.group(1)))
    return (2, case_id)


def write_dataset_split_file(case_split_map, rd_params, forced_keys, output_path):
    """Write a human-readable listing of which cases landed in each split.

    Args:
        case_split_map : dict {(angle, case_id) -> 'train'|'val'|'test'|'unused'}
        rd_params      : nested dict from load_rd_params(), used for tag G-values
        forced_keys    : set of (angle, case_id) tuples that the user pinned;
                         these get a '[forced]' marker in the listing
        output_path    : path to the dataset_split.txt to write
    """
    forced_keys = set(forced_keys) if forced_keys else set()
    groups = {'train': [], 'val': [], 'test': [], 'unused': []}
    for (angle, case_id), split in case_split_map.items():
        if split not in groups:
            continue
        groups[split].append((angle, case_id))

    for key in groups:
        groups[key].sort(key=lambda ac: (ac[0], _case_sort_key(ac[1])))

    section_order = [('TRAIN', 'train'), ('VAL', 'val'),
                     ('TEST', 'test'), ('UNUSED', 'unused')]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for header, key in section_order:
            entries = groups[key]
            if key == 'unused' and not entries:
                continue
            f.write(f"{header} ({len(entries)} cases)\n")
            f.write("=" * 50 + "\n")
            for angle, case_id in entries:
                tag = _case_tag(angle, case_id, rd_params)
                marker = "[forced] " if (angle, case_id) in forced_keys else ""
                f.write(f"  {marker}{tag}\n")
            f.write("\n")


def _assign_cases_gust_intensity(rd_params, val_ratio, test_ratio):
    """Assign cases to train/val/test by gust intensity ranking.

    Returns case_split_map: {(angle, case_id): 'train'|'val'|'test'}
    """
    case_split_map = {}
    for angle, angle_params in rd_params.items():
        case_split_map[(angle, 'base')] = 'train'
        rd_cases = [
            (cid, abs(info['G']))
            for cid, info in angle_params.items()
            if cid != 'base'
        ]
        rd_cases.sort(key=lambda x: x[1])
        n_rd = len(rd_cases)
        n_test = max(1, round(test_ratio * n_rd))
        n_val = max(1, round(val_ratio * n_rd))
        n_train = n_rd - n_val - n_test
        for cid, _ in rd_cases[:n_train]:
            case_split_map[(angle, cid)] = 'train'
        for cid, _ in rd_cases[n_train:n_train + n_val]:
            case_split_map[(angle, cid)] = 'val'
        for cid, _ in rd_cases[n_train + n_val:]:
            case_split_map[(angle, cid)] = 'test'
    return case_split_map


def _assign_cases_random(rd_params, val_ratio, test_ratio,
                          train_cases=None, val_cases=None, test_cases=None):
    """Assign cases to train/val/test by stratified random holdout.

    Optionally accepts three lists of user-supplied case identifiers
    (`train_cases`, `val_cases`, `test_cases`) that pin specific RD cases
    to a given split. The remaining cases are distributed using the
    existing stratified random logic, with per-stratum target counts
    reduced by the number of cases already pinned.

    Returns case_split_map: {(angle, case_id): 'train'|'val'|'test'}
    """
    import random
    rng = random.Random(42)
    case_split_map = {}

    # Parse + validate forced lists against rd_params, then build the
    # (angle, case_id) -> split lookup that overrides random assignment.
    train_forced = _parse_forced_cases(train_cases, rd_params, 'TRAIN_CASES')
    val_forced   = _parse_forced_cases(val_cases,   rd_params, 'VAL_CASES')
    test_forced  = _parse_forced_cases(test_cases,  rd_params, 'TEST_CASES')

    forced_assignments = {}
    for split_name, forced_list in (('train', train_forced),
                                    ('val',   val_forced),
                                    ('test',  test_forced)):
        for key in forced_list:
            if key in forced_assignments:
                raise ValueError(
                    f"[FORCED-CASE] Case {key} appears in more than one of "
                    f"TRAIN_CASES / VAL_CASES / TEST_CASES."
                )
            forced_assignments[key] = split_name

    strata_bounds = [(0, 1.0, 'mild'), (1.0, 2.5, 'medium'),
                     (2.5, float('inf'), 'strong')]
    for angle, angle_params in rd_params.items():
        # Base case always goes to train unless the user explicitly forced it.
        case_split_map[(angle, 'base')] = forced_assignments.get(
            (angle, 'base'), 'train'
        )
        rd_cases = [
            (cid, abs(info['G']))
            for cid, info in angle_params.items()
            if cid != 'base'
        ]
        for lo, hi, _ in strata_bounds:
            if lo == 0:
                stratum = [(cid, g) for cid, g in rd_cases if g <= hi]
            else:
                stratum = [(cid, g) for cid, g in rd_cases if lo < g <= hi]
            n = len(stratum)
            if n == 0:
                continue

            # Partition the stratum into forced (pre-assigned) and available
            # (subject to random selection). Targets are still computed from
            # the full stratum size so the ratio math matches the original.
            forced_in_stratum = {
                cid: forced_assignments[(angle, cid)]
                for cid, _ in stratum
                if (angle, cid) in forced_assignments
            }
            available = [(cid, g) for cid, g in stratum
                         if cid not in forced_in_stratum]

            for cid, split in forced_in_stratum.items():
                case_split_map[(angle, cid)] = split

            n_test_target = max(1, round(test_ratio * n)) if n >= 3 else 0
            n_val_target = (max(1, round(val_ratio * n))
                            if n - n_test_target >= 2 else 0)

            forced_test_n  = sum(1 for s in forced_in_stratum.values() if s == 'test')
            forced_val_n   = sum(1 for s in forced_in_stratum.values() if s == 'val')

            n_test_remain = max(0, n_test_target - forced_test_n)
            n_val_remain  = max(0, n_val_target  - forced_val_n)

            # Clamp remaining picks to what's actually available.
            n_test_remain = min(n_test_remain, len(available))
            n_val_remain  = min(n_val_remain, len(available) - n_test_remain)

            rng.shuffle(available)
            n_train_remain = len(available) - n_val_remain - n_test_remain
            for cid, _ in available[:n_train_remain]:
                case_split_map[(angle, cid)] = 'train'
            for cid, _ in available[n_train_remain:n_train_remain + n_val_remain]:
                case_split_map[(angle, cid)] = 'val'
            for cid, _ in available[n_train_remain + n_val_remain:]:
                case_split_map[(angle, cid)] = 'test'
    return case_split_map


def _assign_cases_angle(rd_params, train_angles, val_angles, test_angles,
                        val_ratio):
    """Assign cases to train/val/test by angle of attack.

    Returns case_split_map: {(angle, case_id): 'train'|'val'|'test'|'unused'}
    """
    import random
    rng = random.Random(42)
    case_split_map = {}
    # TRAIN: all cases from train angles
    train_cases = set()
    for angle in train_angles:
        if angle not in rd_params:
            continue
        for cid in rd_params[angle]:
            train_cases.add((angle, cid))
            case_split_map[(angle, cid)] = 'train'
    # TEST: all cases from test angles
    for angle in test_angles:
        if angle not in rd_params:
            continue
        for cid in rd_params[angle]:
            case_split_map[(angle, cid)] = 'test'
    # VAL: proportional subset from val angles
    n_train_cases = len(train_cases)
    n_val_target = max(1, round(n_train_cases * val_ratio))
    n_val_angles = len([a for a in val_angles if a in rd_params])
    n_per_val_angle = max(1, n_val_target // max(n_val_angles, 1))
    for angle in val_angles:
        if angle not in rd_params:
            continue
        if 'base' in rd_params[angle]:
            case_split_map[(angle, 'base')] = 'val'
        rd_cids = [cid for cid in rd_params[angle] if cid != 'base']
        selected = rng.sample(rd_cids, min(n_per_val_angle, len(rd_cids)))
        for cid in selected:
            case_split_map[(angle, cid)] = 'val'
        for cid in rd_cids:
            if cid not in selected:
                case_split_map[(angle, cid)] = 'unused'
    return case_split_map


def _pre_compute_case_split_map(cfg, rd_params):
    """Pre-compute case→split assignments without needing data tensors.

    Used to identify training cases before encoding, so that the delta
    threshold can be computed from training data only.
    """
    split_mode = getattr(cfg, 'SPLIT_MODE', 'gust_intensity')
    if split_mode == 'angle':
        return _assign_cases_angle(
            rd_params, cfg.TRAIN_ANGLES, cfg.VAL_ANGLES,
            cfg.TEST_ANGLES, cfg.VAL_RATIO)
    elif split_mode == 'random_case':
        return _assign_cases_random(
            rd_params, cfg.VAL_RATIO, cfg.TEST_RATIO,
            train_cases=getattr(cfg, 'TRAIN_CASES', None),
            val_cases=getattr(cfg, 'VAL_CASES', None),
            test_cases=getattr(cfg, 'TEST_CASES', None),
        )
    else:
        return _assign_cases_gust_intensity(
            rd_params, cfg.VAL_RATIO, cfg.TEST_RATIO)


# ==========================================
# DATA SPLITS
# ==========================================
def get_case_splits(spikes, targets, case_labels, rd_params, val_ratio, test_ratio):
    """
    Split sequences into train / val / test by gust parameter value.

    This implements the professor's recommended split strategy:
    out-of-distribution generalisation across gust severity.

    Strategy (applied per angle of attack):
        1. Rank all RD cases for that angle by absolute gust strength |G|.
        2. The mildest 80% of RD cases → train.
        3. The next 10% (moderate-extreme) → val.
        4. The top 10% most extreme |G| → test.
        5. The base case (G=0, no gust) always goes to train.

    This guarantees that the test set contains gust conditions more severe
    than anything seen during training, making it a genuine OOD test of
    generalisation across gust severity — the operationally relevant scenario.

    Because the split is at the simulation-case level, all sequences from a
    given RD case land entirely in one split. There is no sequence-level
    leakage between splits.

    Example (AoA=20°, 30 RD cases, val_ratio=0.1, test_ratio=0.1):
        n_test  = round(0.1 × 30) = 3   most extreme |G|  → test
        n_val   = round(0.1 × 30) = 3   next most extreme → val
        n_train = 30 - 3 - 3 = 24       mildest           → train
        base case always                                   → train

    Args:
        spikes      : (N, T_spk, C, Ny, Nx) spike tensor
        targets     : (N, T) lift tensor
        case_labels : list of length N with (angle, case_id) tuples,
                      one per sequence — produced by create_sequences()
        rd_params   : nested dict from load_rd_params():
                      rd_params[angle][case_id] = {'G': ..., 'R': ..., 'y': ...}
        val_ratio   : fraction of RD cases (per angle) assigned to val
        test_ratio  : fraction of RD cases (per angle) assigned to test

    Returns:
        train_ds, val_ds, test_ds : torch Subset datasets
        split_indices : dict with 'train', 'val', 'test' index lists
        case_split_map : dict mapping (angle, case_id) → 'train'|'val'|'test',
                         for inspection and reporting
    """
    case_split_map = _assign_cases_gust_intensity(rd_params, val_ratio, test_ratio)

    train_cases = {k for k, v in case_split_map.items() if v == 'train'}
    val_cases   = {k for k, v in case_split_map.items() if v == 'val'}
    test_cases  = {k for k, v in case_split_map.items() if v == 'test'}

    # Print per-angle split summary
    for angle, angle_params in rd_params.items():
        rd_cases = sorted(
            [(cid, abs(info['G'])) for cid, info in angle_params.items()
             if cid != 'base'],
            key=lambda x: x[1])
        n_rd = len(rd_cases)
        n_test = sum(1 for c, _ in rd_cases if (angle, c) in test_cases)
        n_val  = sum(1 for c, _ in rd_cases if (angle, c) in val_cases)
        n_train = n_rd - n_val - n_test + 1  # +1 for base
        g_train_max = max((g for c, g in rd_cases if (angle, c) in train_cases), default=0.0)
        g_test_min  = min((g for c, g in rd_cases if (angle, c) in test_cases), default=0.0)
        print(f"[SPLIT] AoA={angle}°: train={n_train} val={n_val} test={n_test} "
              f"| train |G|≤{g_train_max:.2f}, test |G|≥{g_test_min:.2f}")

    # Map case assignments to sequence indices
    train_idx = [i for i, lbl in enumerate(case_labels) if lbl in train_cases]
    val_idx   = [i for i, lbl in enumerate(case_labels) if lbl in val_cases]
    test_idx  = [i for i, lbl in enumerate(case_labels) if lbl in test_cases]

    full_dataset = TensorDataset(spikes, targets)
    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    test_ds  = Subset(full_dataset, test_idx)

    split_indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    total = len(case_labels)
    print(f"[SPLIT] Sequences — train={len(train_idx)} ({len(train_idx)/total*100:.1f}%), "
          f"val={len(val_idx)} ({len(val_idx)/total*100:.1f}%), "
          f"test={len(test_idx)} ({len(test_idx)/total*100:.1f}%)")

    return train_ds, val_ds, test_ds, split_indices, case_split_map


# ==========================================
# DATA SPLITS — BY ANGLE OF ATTACK
# ==========================================
def get_angle_splits(spikes, targets, case_labels, rd_params,
                     train_angles, val_angles, test_angles,
                     val_ratio, test_ratio):
    """
    Split sequences into train / val / test by angle of attack.

    Strategy:
        1. ALL cases from train_angles → train.
        2. ALL cases from test_angles → test.
        3. A proportional subset from val_angles → val.
           N_val_target = N_train_cases × val_ratio.
           Cases are distributed evenly across val angles and selected
           randomly (seed=42) within each angle. Base case is always
           included. If N_val_target exceeds available val-angle cases,
           all available cases are used.

    This tests generalisation across angle of attack rather than gust
    severity. All gust intensities (mild to extreme) are seen during
    training for the training angles.

    Args:
        spikes       : (N, T_spk, C, Ny, Nx) spike tensor
        targets      : (N, T) lift tensor
        case_labels  : list of length N with (angle, case_id) tuples
        rd_params    : nested dict from load_rd_params()
        train_angles : list of angles for training, e.g. [30, 40]
        val_angles   : list of angles for validation, e.g. [20, 50]
        test_angles  : list of angles for testing, e.g. [60]
        val_ratio    : controls val size as fraction of train case count
        test_ratio   : unused (test gets ALL cases from test_angles),
                       kept for interface consistency

    Returns:
        train_ds, val_ds, test_ds : torch Subset datasets
        split_indices : dict with 'train', 'val', 'test' index lists
        case_split_map : dict mapping (angle, case_id) → split name
    """
    case_split_map = _assign_cases_angle(
        rd_params, train_angles, val_angles, test_angles, val_ratio)

    train_cases = {k for k, v in case_split_map.items() if v == 'train'}
    val_cases   = {k for k, v in case_split_map.items() if v == 'val'}
    test_cases  = {k for k, v in case_split_map.items() if v == 'test'}

    n_train_cases = len(train_cases)
    n_val_angles = len([a for a in val_angles if a in rd_params])
    n_val_target = max(1, round(n_train_cases * val_ratio))
    n_per_val_angle = max(1, n_val_target // max(n_val_angles, 1))

    # Map case assignments to sequence indices (skip 'unused')
    train_idx = [i for i, lbl in enumerate(case_labels) if lbl in train_cases]
    val_idx   = [i for i, lbl in enumerate(case_labels) if lbl in val_cases]
    test_idx  = [i for i, lbl in enumerate(case_labels) if lbl in test_cases]

    full_dataset = TensorDataset(spikes, targets)
    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    test_ds  = Subset(full_dataset, test_idx)

    split_indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    total = len(case_labels)
    n_unused = total - len(train_idx) - len(val_idx) - len(test_idx)
    print(f"[SPLIT] Mode: angle-of-attack")
    print(f"[SPLIT] Train angles: {train_angles} → {len(train_cases)} cases, "
          f"{len(train_idx)} sequences")
    print(f"[SPLIT] Val angles:   {val_angles} → {len(val_cases)} cases, "
          f"{len(val_idx)} sequences "
          f"(target: {n_val_target} cases, {n_per_val_angle}/angle)")
    print(f"[SPLIT] Test angles:  {test_angles} → {len(test_cases)} cases, "
          f"{len(test_idx)} sequences")
    if n_unused > 0:
        print(f"[SPLIT] Unused: {n_unused} sequences from val-angle cases not selected")

    return train_ds, val_ds, test_ds, split_indices, case_split_map


# ==========================================
# DATA SPLITS — RANDOM CASE HOLDOUT (STRATIFIED)
# ==========================================
def get_random_case_splits(spikes, targets, case_labels, rd_params,
                           val_ratio, test_ratio,
                           forced_train=None, forced_val=None, forced_test=None):
    """
    Split sequences into train / val / test by randomly holding out
    entire RD cases, stratified by gust intensity.

    All 5 angles are included in every split — this tests generalisation
    to unseen gust profiles at known angles (interpolation), not to
    unseen angles.

    Strategy (applied per angle):
        1. Rank RD cases by |G| and divide into 3 strata:
           mild (|G| <= 1), medium (1 < |G| <= 2.5), strong (|G| > 2.5).
        2. From each stratum, randomly select test_ratio cases → test,
           val_ratio cases → val, remainder → train.
        3. Base case (G=0) always goes to train.

    This ensures val and test sets contain a representative mix of gust
    intensities, preventing bias toward mild or extreme conditions.

    Args:
        spikes       : (N, T_spk, C, Ny, Nx) spike tensor
        targets      : (N, T) lift tensor
        case_labels  : list of length N with (angle, case_id) tuples
        rd_params    : nested dict from load_rd_params()
        val_ratio    : fraction of RD cases per stratum assigned to val
        test_ratio   : fraction of RD cases per stratum assigned to test
        forced_train : list of identifier strings (e.g. "AoA20_RD19_G_n2.6")
                       to pin to the train split. None / [] disables.
        forced_val   : same, for val split.
        forced_test  : same, for test split.

    Returns:
        train_ds, val_ds, test_ds : torch Subset datasets
        split_indices : dict with 'train', 'val', 'test' index lists
        case_split_map : dict mapping (angle, case_id) → 'train'|'val'|'test'
    """
    case_split_map = _assign_cases_random(
        rd_params, val_ratio, test_ratio,
        train_cases=forced_train, val_cases=forced_val, test_cases=forced_test,
    )

    train_set = {k for k, v in case_split_map.items() if v == 'train'}
    val_set   = {k for k, v in case_split_map.items() if v == 'val'}
    test_set  = {k for k, v in case_split_map.items() if v == 'test'}

    forced_keys = set()
    for forced in (forced_train, forced_val, forced_test):
        forced_keys.update(_parse_forced_cases(forced, rd_params, '_'))

    # Print per-angle-per-stratum summary
    strata_bounds = [(0, 1.0, 'mild'), (1.0, 2.5, 'medium'),
                     (2.5, float('inf'), 'strong')]
    for angle, angle_params in rd_params.items():
        rd_cases = [
            (cid, abs(info['G']))
            for cid, info in angle_params.items()
            if cid != 'base'
        ]
        for lo, hi, stratum_name in strata_bounds:
            if lo == 0:
                stratum = [(cid, g) for cid, g in rd_cases if g <= hi]
            else:
                stratum = [(cid, g) for cid, g in rd_cases if lo < g <= hi]
            n = len(stratum)
            if n == 0:
                continue
            s_train = sum(1 for c, _ in stratum if (angle, c) in train_set)
            s_val   = sum(1 for c, _ in stratum if (angle, c) in val_set)
            s_test  = sum(1 for c, _ in stratum if (angle, c) in test_set)
            print(f"[SPLIT] AoA={angle}° {stratum_name:6s} ({n:2d} cases): "
                  f"train={s_train} val={s_val} test={s_test}")

    # Map case assignments to sequence indices
    train_idx = [i for i, lbl in enumerate(case_labels) if lbl in train_set]
    val_idx   = [i for i, lbl in enumerate(case_labels) if lbl in val_set]
    test_idx  = [i for i, lbl in enumerate(case_labels) if lbl in test_set]

    full_dataset = TensorDataset(spikes, targets)
    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    test_ds  = Subset(full_dataset, test_idx)

    split_indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    total = len(case_labels)
    print(f"[SPLIT] Mode: random_case (stratified by |G|)")
    print(f"[SPLIT] Cases — train={len(train_set)} val={len(val_set)} "
          f"test={len(test_set)}")
    print(f"[SPLIT] Sequences — train={len(train_idx)} ({len(train_idx)/total*100:.1f}%), "
          f"val={len(val_idx)} ({len(val_idx)/total*100:.1f}%), "
          f"test={len(test_idx)} ({len(test_idx)/total*100:.1f}%)")

    # Forced-case summary + ratio overflow warning
    if forced_keys:
        print(f"[SPLIT] Forced cases ({len(forced_keys)} total):")
        for split_name, identifiers in (('train', forced_train),
                                         ('val',   forced_val),
                                         ('test',  forced_test)):
            if identifiers:
                tags = ", ".join(identifiers)
                print(f"[SPLIT]   {split_name}: {tags}")

        # Count RD-only cases (excluding base) to compare against ratios
        rd_total = sum(len([c for c in angle_params if c != 'base'])
                       for angle_params in rd_params.values())
        if rd_total > 0:
            rd_val  = sum(1 for k in val_set  if k[1] != 'base')
            rd_test = sum(1 for k in test_set if k[1] != 'base')
            actual_val_ratio  = rd_val  / rd_total
            actual_test_ratio = rd_test / rd_total
            tol = 0.05
            if (actual_val_ratio  - val_ratio  > tol or
                actual_test_ratio - test_ratio > tol):
                print(f"[SPLIT][WARN] Forced cases pushed actual ratios above "
                      f"targets by >5pp: "
                      f"val {actual_val_ratio:.2%} (target {val_ratio:.0%}), "
                      f"test {actual_test_ratio:.2%} (target {test_ratio:.0%}).")

    return train_ds, val_ds, test_ds, split_indices, case_split_map


def _dispatch_split(cfg, spikes, targets, case_labels, rd_params):
    """Route to the correct split function based on cfg.SPLIT_MODE."""
    split_mode = getattr(cfg, 'SPLIT_MODE', 'gust_intensity')
    if split_mode == 'angle':
        return get_angle_splits(
            spikes, targets, case_labels, rd_params,
            cfg.TRAIN_ANGLES, cfg.VAL_ANGLES, cfg.TEST_ANGLES,
            cfg.VAL_RATIO, cfg.TEST_RATIO,
        )
    elif split_mode == 'random_case':
        return get_random_case_splits(
            spikes, targets, case_labels, rd_params,
            cfg.VAL_RATIO, cfg.TEST_RATIO,
            forced_train=getattr(cfg, 'TRAIN_CASES', None),
            forced_val=getattr(cfg, 'VAL_CASES', None),
            forced_test=getattr(cfg, 'TEST_CASES', None),
        )
    else:
        return get_case_splits(
            spikes, targets, case_labels, rd_params,
            cfg.VAL_RATIO, cfg.TEST_RATIO,
        )


# ==========================================
# INTERPOLATED DATASET — STREAMING SUPPORT
# ==========================================
# These utilities enable a memory-efficient pipeline for the temporally
# up-sampled dataset produced by dataset_interpolation.py. The interpolated
# dataset can be 10x larger than the original, so the legacy "load all cases
# into RAM, encode all spikes at once" pipeline does not fit. Instead, we:
#   1. enumerate case files (paths only),
#   2. compute the global P99 normalization scale via streamed sampling,
#   3. compute the global delta threshold via streamed |Δω| accumulation,
#   4. build a sliding-window manifest from .npy headers (no data load),
#   5. let StreamingSpikeDataset.__getitem__ mmap-load only the required
#      vorticity slice, normalize it, encode spikes, and return one window.

from collections import OrderedDict


def list_interpolated_cases(interp_root, angles):
    """Enumerate (angle, case_id, vort_npy, cl_npy) tuples — paths only.

    Returns a list sorted deterministically by (angle, case_id) so manifests
    are reproducible across runs.
    """
    cases = []
    vort_root = os.path.join(interp_root, 'vorticity')
    lift_root = os.path.join(interp_root, 'lift')
    for angle in angles:
        vort_folder = os.path.join(vort_root, f"vort_a{angle}")
        lift_folder = os.path.join(lift_root, f"lift_a{angle}")
        if not os.path.isdir(vort_folder):
            print(f"[WARN] Interpolated vorticity folder missing: {vort_folder}")
            continue
        npy_files = sorted(glob.glob(
            os.path.join(vort_folder, f"Vort_AoA{angle}_*.npy")))
        for vort_path in npy_files:
            fname = os.path.basename(vort_path)
            case_id = fname.replace(f"Vort_AoA{angle}_", "").replace(".npy", "")
            cl_path = os.path.join(
                lift_folder, f"Lift_AoA{angle}_{case_id}.npy")
            if not os.path.exists(cl_path):
                print(f"[WARN] Lift .npy missing for {fname}, skipping.")
                continue
            cases.append((angle, case_id, vort_path, cl_path))
    cases.sort(key=lambda t: (t[0], t[1]))
    print(f"[INTERP] Enumerated {len(cases)} interpolated cases "
          f"across angles {angles}")
    return cases


def compute_norm_scale_streaming(case_paths, percentile,
                                  samples_per_case=100_000, frame_chunk=50):
    """Compute robust normalization scale by streaming through cases.

    Uses a single sequential chunked pass per case so that no case is ever
    fully loaded into RAM. Both the running absmax and the per-case random
    sample are accumulated inside the same loop, avoiding the two-full-load
    anti-pattern that would otherwise read each ~1.4 GB file twice.

    Args:
        case_paths:      list of (angle, case_id, vort_path, cl_path) tuples
        percentile:      percentile of |ω| to use (100 = absmax, no clipping)
        samples_per_case: target number of |ω| samples drawn per case for the
                         percentile estimator; distributed evenly across chunks
        frame_chunk:     frames read per iteration (controls peak RAM per case)

    Returns:
        norm_stats: {'omega_absmax', 'norm_scale', 'percentile'}
    """
    rng = np.random.RandomState(42)
    omega_absmax = 0.0
    samples = []
    n_cases = len(case_paths)

    for i, (angle, case_id, vort_path, _) in enumerate(case_paths):
        mm = np.load(vort_path, mmap_mode='r')
        Nt = mm.shape[0]
        n_chunks = max(1, (Nt + frame_chunk - 1) // frame_chunk)
        # Budget samples evenly across chunks so we never read the full flat array
        samples_per_chunk = max(1, samples_per_case // n_chunks) if percentile < 100 else 0

        case_max = 0.0
        case_samples = []

        for t0 in range(0, Nt, frame_chunk):
            chunk = np.array(mm[t0:min(t0 + frame_chunk, Nt)], dtype=np.float32)
            abs_chunk = np.abs(chunk)
            chunk_max = float(abs_chunk.max())
            if chunk_max > case_max:
                case_max = chunk_max

            if percentile < 100:
                flat = abs_chunk.ravel()
                n_draw = min(samples_per_chunk, len(flat))
                idx = rng.choice(len(flat), n_draw, replace=False)
                case_samples.append(flat[idx])
                del flat

            del chunk, abs_chunk

        omega_absmax = max(omega_absmax, case_max)
        if percentile < 100 and case_samples:
            samples.append(np.concatenate(case_samples))
        del mm, case_samples
        gc.collect()

        if (i + 1) % 10 == 0 or (i + 1) == n_cases:
            print(f"[NORM] {i + 1}/{n_cases} cases scanned  "
                  f"(running |omega|_max = {omega_absmax:.4f})")

    if percentile >= 100:
        norm_scale = omega_absmax
    else:
        all_samples = np.concatenate(samples)
        norm_scale = float(np.percentile(all_samples, percentile))
        del samples, all_samples
        gc.collect()

    print(f"[NORM] Global |omega|_max = {omega_absmax:.4f}")
    if percentile < 100:
        print(f"[NORM] P{percentile} |omega| = {norm_scale:.4f} (streaming)")
        print(f"[NORM] Clipping ratio: {omega_absmax / (norm_scale + 1e-8):.2f}x")
    return {'omega_absmax': float(omega_absmax),
            'norm_scale': float(norm_scale),
            'percentile': percentile}


def compute_delta_threshold_streaming(case_paths, norm_scale, threshold_factor,
                                       train_keys, frame_chunk=200):
    """Compute the delta-encoding threshold by streaming training cases only.

    For each case in train_keys, normalize the vorticity in chunks of
    frame_chunk frames, accumulate |Δω| sum and element count, then return
    (sum/count) * threshold_factor.

    Note: with denser interpolated time grids, mean|Δω| naturally drops
    (~1/(N+1) for smooth signals), so the absolute threshold returned here
    will be smaller than for the original dataset. THRESHOLD_FACTOR remains
    meaningful as a multiplier on mean|Δω|.
    """
    delta_sum = 0.0
    delta_count = 0
    n_train_cases = 0
    n_train_total = sum(1 for a, cid, _, _ in case_paths if (a, cid) in train_keys)

    inv_scale = 1.0 / (norm_scale + 1e-8)

    for angle, case_id, vort_path, _ in case_paths:
        if (angle, case_id) not in train_keys:
            continue
        mm = np.load(vort_path, mmap_mode='r')
        Nt = mm.shape[0]
        prev_norm = None
        for t0 in range(0, Nt, frame_chunk):
            t1 = min(t0 + frame_chunk, Nt)
            chunk = np.array(mm[t0:t1], dtype=np.float32) * inv_scale
            np.clip(chunk, -1.0, 1.0, out=chunk)
            if prev_norm is not None:
                edge = np.abs(chunk[0] - prev_norm)
                delta_sum += float(edge.sum())
                delta_count += edge.size
                del edge
            if chunk.shape[0] > 1:
                diff = np.abs(np.diff(chunk, axis=0))
                delta_sum += float(diff.sum())
                delta_count += diff.size
                del diff
            prev_norm = chunk[-1].copy()
            del chunk
        n_train_cases += 1
        print(f"[ENCODE] Delta scan {n_train_cases}/{n_train_total}: "
              f"AoA{angle}_{case_id}")
        del mm, prev_norm
        gc.collect()

    if delta_count == 0:
        raise RuntimeError(
            "[ENCODE] No training cases found for delta threshold computation. "
            "Check SPLIT_MODE and ANGLES settings."
        )

    mean_abs_delta = delta_sum / delta_count
    threshold = mean_abs_delta * threshold_factor
    print(f"[ENCODE] Delta threshold = {threshold:.6f}"
          f"  (mean |Δω| = {mean_abs_delta:.6f} × {threshold_factor},"
          f" from {n_train_cases} training cases)")
    return threshold, mean_abs_delta, n_train_cases


def build_window_manifest(case_paths, time_steps, stride, warmup_steps):
    """Enumerate sliding windows across all interpolated cases.

    For each case, reads Nt_new from the .npy header (O(1)) and emits one
    tuple per supervised window. Sequence ordering is per-case, with cases
    iterated in case_paths order (which is sorted deterministically by
    list_interpolated_cases).

    Returns:
        manifest:    list of (case_idx, in_start, sup_start, sup_end) tuples
                     - in_start  : first frame of input window  (>= 0)
                     - sup_start : first supervised frame       (== in_start + warmup_steps)
                     - sup_end   : one past last supervised frame
        case_labels: list of (angle, case_id) parallel to manifest
    """
    if stride is None:
        stride = time_steps

    manifest = []
    case_labels = []

    for case_idx, (angle, case_id, vort_path, _) in enumerate(case_paths):
        # np.load with mmap reads only the header; .shape is O(1)
        mm = np.load(vort_path, mmap_mode='r')
        Nt = mm.shape[0]
        del mm

        min_frames = warmup_steps + time_steps
        if Nt < min_frames:
            print(f"[WARN] AoA{angle}_{case_id}: only {Nt} frames < "
                  f"{min_frames} (warmup+TIME_STEPS), skipping.")
            continue

        starts = list(range(warmup_steps, Nt - time_steps + 1, stride))
        for sup_start in starts:
            in_start  = sup_start - warmup_steps
            sup_end   = sup_start + time_steps
            manifest.append((case_idx, in_start, sup_start, sup_end))
            case_labels.append((angle, case_id))

    print(f"[MANIFEST] {len(manifest)} windows across "
          f"{len(set(case_labels))} cases "
          f"(T={time_steps}, stride={stride}, warmup={warmup_steps})")
    return manifest, case_labels


class StreamingSpikeDataset(torch.utils.data.Dataset):
    """Lazy spike-encoded sequence dataset for the interpolated dataset.

    __getitem__ mmaps the relevant case's vorticity .npy, slices the requested
    window (plus one delta-context frame at the start), normalizes the slice
    with the global norm_scale, and encodes the slice to spikes via the same
    full-case encoders used by the legacy pipeline. Returns
    (spikes_window, cl_window) per item.

    A small LRU cache keeps the most-recently-opened (vort_mmap, cl_array)
    pairs around so consecutive items within a case avoid reopening the file.

    Only used by decoder-free models (v5+) — the legacy decoder models would
    additionally need raw omega in the batch and are not supported here.
    """

    def __init__(self, manifest, case_paths, norm_scale, threshold,
                 encoding, gamma=1.0, gain=1.0, zero_first_frame=True,
                 lru_size=8):
        super().__init__()
        if encoding not in ('rate', 'delta', 'rate+delta'):
            raise ValueError(
                f"StreamingSpikeDataset: encoding={encoding!r} not supported. "
                f"Use 'rate', 'delta', or 'rate+delta'."
            )
        self.manifest = manifest
        self.case_paths = case_paths
        self.norm_scale = float(norm_scale)
        self.threshold = float(threshold) if threshold is not None else None
        self.encoding = encoding
        self.gamma = float(gamma)
        self.gain = float(gain)
        self.zero_first_frame = bool(zero_first_frame)
        self.lru_size = int(lru_size)
        self._cache = OrderedDict()  # case_idx -> (vort_mmap, cl_array)

    def __len__(self):
        return len(self.manifest)

    def _open_case(self, case_idx):
        if case_idx in self._cache:
            self._cache.move_to_end(case_idx)
            return self._cache[case_idx]
        _, _, vort_path, cl_path = self.case_paths[case_idx]
        vort_mm = np.load(vort_path, mmap_mode='r')
        cl_arr = np.load(cl_path)  # small, full load
        self._cache[case_idx] = (vort_mm, cl_arr)
        if len(self._cache) > self.lru_size:
            self._cache.popitem(last=False)
        return vort_mm, cl_arr

    def _encode_window(self, omega_slice, prepend_first):
        """Encode an (Nw, Ny, Nx) float32 slice in [-1, 1] to spikes.

        If prepend_first is True, the slice already includes one extra leading
        delta-context frame at index 0 that must be dropped from the returned
        spike tensor. The full-case encoders zero out spikes at t=0 when
        zero_first_frame is set, so the returned spike tensor's first frame
        is the supervised one (or input one if warmup>0).
        """
        if self.encoding == 'rate':
            spikes = _rate_encode_full_case(
                omega_slice, gamma=self.gamma, gain=self.gain)
        elif self.encoding == 'delta':
            spikes = _delta_encode_full_case(
                omega_slice, self.threshold,
                zero_first_frame=self.zero_first_frame)
        else:  # rate+delta
            spikes = _hybrid_encode_full_case(
                omega_slice, self.threshold,
                gamma=self.gamma, gain=self.gain,
                zero_first_frame=self.zero_first_frame)

        if prepend_first:
            spikes = spikes[1:]
        return spikes

    def __getitem__(self, idx):
        case_idx, in_start, sup_start, sup_end = self.manifest[idx]
        vort_mm, cl_arr = self._open_case(case_idx)

        # One extra leading frame for delta context. If in_start == 0 (no
        # earlier frame available), duplicate frame 0 — _delta_encode_full_case
        # zeroes out t=0 anyway when zero_first_frame=True, so the duplicated
        # frame has no effect on the supervised region.
        if in_start > 0:
            slice_start = in_start - 1
            prepend_first = True
        else:
            slice_start = 0
            prepend_first = False

        omega_raw = np.array(vort_mm[slice_start:sup_end], dtype=np.float32)
        if not prepend_first:
            # Pad with a duplicate of frame 0 so the encoder sees the same
            # window length regardless of in_start. Drop the duplicate after.
            omega_raw = np.concatenate(
                [omega_raw[0:1], omega_raw], axis=0)
            prepend_first = True

        # Normalize in-place
        omega_raw *= (1.0 / (self.norm_scale + 1e-8))
        np.clip(omega_raw, -1.0, 1.0, out=omega_raw)

        spikes = self._encode_window(omega_raw, prepend_first=prepend_first)
        cl_window = cl_arr[sup_start:sup_end].astype(np.float32, copy=False)

        # Legacy TensorDataset path stores cl with .unsqueeze(-1) (shape (T, 1))
        # so train_SNN.py can do cl_batch.permute(1, 0, 2). Match that here so the
        # train loop is identical for original and interpolated datasets.
        cl_tensor = torch.from_numpy(cl_window).unsqueeze(-1)
        return spikes, cl_tensor


def get_case_nt(case_path: tuple) -> int:
    """Return the number of timesteps for a case without loading data into RAM."""
    _, _, vort_path, _ = case_path
    mm = np.load(vort_path, mmap_mode='r')
    return int(mm.shape[0])


def load_encoded_chunk(
    case_path: tuple,
    start: int,
    end: int,
    norm_scale: float,
    threshold,
    cfg,
) -> tuple:
    """Load and encode the contiguous chunk [start, end) from one case.

    Mirrors StreamingSpikeDataset.__getitem__ but operates on an arbitrary
    [start, end) range without a manifest layer.  Designed for stateful TBPTT
    where chunks are iterated sequentially inside the training loop.

    Args:
        case_path:  (angle, case_id, vort_path, cl_path) tuple
        start:      first supervised frame index (inclusive)
        end:        one-past last supervised frame (exclusive)
        norm_scale: global normalization scale from norm_stats
        threshold:  delta-encoding threshold, or None for rate-only
        cfg:        namespace with ENCODING, GAMMA, GAIN, ZERO_FIRST_FRAME

    Returns:
        x_chunk:  (T, C, H, W) float32 spike tensor   (T = end - start)
        cl_chunk: (T, 1) float32 Cl target tensor
    """
    _, _, vort_path, cl_path = case_path

    # Prepend one extra frame for delta-context (same logic as __getitem__)
    if start > 0:
        slice_start   = start - 1
        prepend_first = True
    else:
        slice_start   = 0
        prepend_first = False

    vort_mm = np.load(vort_path, mmap_mode='r')
    cl_arr  = np.load(cl_path)

    omega_raw = np.array(vort_mm[slice_start:end], dtype=np.float32)

    if not prepend_first:
        # Duplicate frame 0 as context so the encoder always has a prior frame
        # _delta_encode_full_case zeroes t=0 spikes when zero_first_frame=True,
        # so this duplicate has no effect on the supervised output.
        omega_raw     = np.concatenate([omega_raw[0:1], omega_raw], axis=0)
        prepend_first = True

    omega_raw *= 1.0 / (norm_scale + 1e-8)
    np.clip(omega_raw, -1.0, 1.0, out=omega_raw)

    encoding         = cfg.ENCODING
    gamma            = getattr(cfg, 'GAMMA', 1.0)
    gain             = getattr(cfg, 'GAIN', 1.0)
    zero_first_frame = getattr(cfg, 'ZERO_FIRST_FRAME', True)

    if encoding == 'rate':
        spikes = _rate_encode_full_case(omega_raw, gamma=gamma, gain=gain)
    elif encoding == 'delta':
        spikes = _delta_encode_full_case(omega_raw, threshold,
                                         zero_first_frame=zero_first_frame)
    else:  # rate+delta
        spikes = _hybrid_encode_full_case(omega_raw, threshold,
                                          gamma=gamma, gain=gain,
                                          zero_first_frame=zero_first_frame)

    if prepend_first:
        spikes = spikes[1:]  # drop context frame → (T, C, H, W)

    cl_window = cl_arr[start:end].astype(np.float32, copy=False)
    x_chunk   = spikes.float()                            # (T, C, H, W)
    cl_chunk  = torch.from_numpy(cl_window).unsqueeze(-1) # (T, 1)
    return x_chunk, cl_chunk


def _load_and_encode_interpolated(cfg, force_recompute=False):
    """Streaming pipeline for DATASET_TYPE='interpolated'.

    Builds a StreamingSpikeDataset rather than materializing all spikes in
    RAM. The expensive precomputation (norm scale, delta threshold, window
    manifest) is cached to a small .pt file so repeated runs with the same
    config skip straight to dataset construction.
    """
    interp_root = cfg.INTERP_ROOT
    if interp_root is None:
        raise RuntimeError(
            "[INTERP] cfg.INTERP_ROOT is None — set DATASET_TYPE='interpolated' "
            "and INTERP_METHOD/N_INTERPOLATED in train_SNN.py."
        )
    if not os.path.isdir(interp_root):
        raise FileNotFoundError(
            f"[INTERP] Interpolated dataset folder not found: {interp_root}\n"
            f"Run: python main/dataset_interpolation.py "
            f"--method {getattr(cfg, 'INTERP_METHOD', 'cubic_spline')} "
            f"--n {getattr(cfg, 'N_INTERPOLATED', 9)}"
        )

    rd_params = load_rd_params(cfg.RD_PARAMS_PATH, cfg.ANGLES)
    cache_dir = os.path.join(interp_root, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Manifest cache file is small: just the manifest list, case_paths, and
    # the precomputed scalars (norm_scale, threshold). No tensor data.
    manifest_path = cfg.CACHE_PATH  # train_SNN.py points this at interp_root/cache/...
    manifest_gz = manifest_path + '.gz'
    actual_cache = manifest_gz if os.path.exists(manifest_gz) else manifest_path

    saved = None
    if os.path.exists(actual_cache) and not force_recompute:
        print(f"\n[CACHE] Loading interpolated manifest from "
              f"'{os.path.basename(actual_cache)}'...")
        if actual_cache.endswith('.gz'):
            with gzip.open(actual_cache, 'rb') as f:
                saved = torch.load(f, map_location='cpu', weights_only=False)
        else:
            saved = torch.load(actual_cache, map_location='cpu',
                               weights_only=False)
        cached_encoding = saved.get('encoding', None)
        if cached_encoding is not None and cached_encoding != cfg.ENCODING:
            raise RuntimeError(
                f"[CACHE] Encoding mismatch: cache='{cached_encoding}' "
                f"vs cfg='{cfg.ENCODING}'. Delete the manifest cache or set "
                f"force_recompute=True."
            )
        case_paths = saved['case_paths']
        manifest = saved['manifest']
        case_labels = saved['case_labels']
        norm_stats = saved['norm_stats']
        threshold = saved['threshold']

    if saved is None:
        # --- Full streaming pipeline ---
        print(f"\n[INTERP] Building manifest from {interp_root}")
        case_paths = list_interpolated_cases(interp_root, cfg.ANGLES)
        if len(case_paths) == 0:
            raise RuntimeError(
                f"[INTERP] No interpolated cases found under {interp_root}/"
                f"vorticity/. Did you run dataset_interpolation.py?"
            )

        # Streaming P99 normalization
        percentile = getattr(cfg, 'NORM_PERCENTILE', 100)
        norm_stats = compute_norm_scale_streaming(case_paths, percentile)

        # Determine training cases for threshold computation
        threshold = None
        if cfg.ENCODING in ('delta', 'rate+delta'):
            pre_split_map = _pre_compute_case_split_map(cfg, rd_params)
            train_keys = {k for k, v in pre_split_map.items() if v == 'train'}
            threshold, _, _ = compute_delta_threshold_streaming(
                case_paths, norm_stats['norm_scale'],
                cfg.THRESHOLD_FACTOR, train_keys,
            )

        # Build the sliding-window manifest (no data load — only .npy headers)
        stride = getattr(cfg, 'STRIDE', None) or cfg.TIME_STEPS
        warmup_steps = getattr(cfg, 'WARMUP_STEPS', 0)
        manifest, case_labels = build_window_manifest(
            case_paths, cfg.TIME_STEPS, stride, warmup_steps,
        )

        # Persist the manifest cache
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        save_dict = {
            'manifest': manifest,
            'case_paths': case_paths,
            'case_labels': case_labels,
            'norm_stats': norm_stats,
            'threshold': threshold,
            'bins': getattr(cfg, 'BINS', 1),
            'encoding': cfg.ENCODING,
        }
        print(f"[CACHE] Saving manifest to "
              f"'{os.path.basename(manifest_gz)}'...")
        with gzip.open(manifest_gz, 'wb', compresslevel=4) as f:
            torch.save(save_dict, f)
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
            print(f"[CACHE] Removed old uncompressed manifest")

    # Build the streaming dataset
    gamma = getattr(cfg, 'GAMMA', 1.0)
    gain = getattr(cfg, 'GAIN', 1.0)
    zero_first = getattr(cfg, 'ZERO_FIRST_FRAME', True)
    full_dataset = StreamingSpikeDataset(
        manifest=manifest,
        case_paths=case_paths,
        norm_scale=norm_stats['norm_scale'],
        threshold=threshold,
        encoding=cfg.ENCODING,
        gamma=gamma,
        gain=gain,
        zero_first_frame=zero_first,
        lru_size=getattr(cfg, 'STREAM_LRU_SIZE', 8),
    )

    # Apply the case-level split to derive train/val/test index lists.
    # Mirrors _dispatch_split's case→sequence-index logic, but skips building
    # a TensorDataset since we already have a Dataset instance.
    case_split_map = _pre_compute_case_split_map(cfg, rd_params)
    train_cases = {k for k, v in case_split_map.items() if v == 'train'}
    val_cases   = {k for k, v in case_split_map.items() if v == 'val'}
    test_cases  = {k for k, v in case_split_map.items() if v == 'test'}

    train_idx = [i for i, lbl in enumerate(case_labels) if lbl in train_cases]
    val_idx   = [i for i, lbl in enumerate(case_labels) if lbl in val_cases]
    test_idx  = [i for i, lbl in enumerate(case_labels) if lbl in test_cases]

    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    test_ds  = Subset(full_dataset, test_idx)

    split_indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    total = max(len(case_labels), 1)
    print(f"[SPLIT] Mode: {getattr(cfg, 'SPLIT_MODE', 'gust_intensity')} "
          f"(streaming)")
    print(f"[SPLIT] Cases — train={len(train_cases)} val={len(val_cases)} "
          f"test={len(test_cases)}")
    print(f"[SPLIT] Sequences — train={len(train_idx)} ({len(train_idx)/total*100:.1f}%), "
          f"val={len(val_idx)} ({len(val_idx)/total*100:.1f}%), "
          f"test={len(test_idx)} ({len(test_idx)/total*100:.1f}%)")

    return {
        'train_dataset': train_ds,
        'val_dataset': val_ds,
        'test_dataset': test_ds,
        'split_indices': split_indices,
        'case_split_map': case_split_map,
        'rd_params': rd_params,
        'spikes': None,
        'targets': None,
        'norm_stats': norm_stats,
        'case_labels': case_labels,
        'threshold': threshold,
        'bins': getattr(cfg, 'BINS', 1),
        'case_paths': case_paths,
        'manifest': manifest,
        'streaming_dataset': full_dataset,
    }


# ==========================================
# ORIGINAL DATASET — STATEFUL TBPTT SUPPORT
# ==========================================
def _list_original_cases_as_npy(cfg, force_recompute=False):
    """Mirror the original .mat/.csv files as per-case .npy files.

    The stateful TBPTT pipeline (`load_encoded_chunk`, `get_case_nt`) was
    designed around mmap-able .npy files. This helper writes one .npy per
    case the first time it runs, then returns the (angle, case_id, vort_npy,
    cl_npy) tuples on subsequent calls. Stored vorticity is RAW (un-normalized,
    un-encoded) so `load_encoded_chunk` can apply norm_scale and encoding
    on the fly — same contract as the interpolated path.
    """
    cache_root = os.path.dirname(cfg.CACHE_PATH)
    npy_dir = os.path.join(cache_root, 'per_case_npy')
    os.makedirs(npy_dir, exist_ok=True)

    case_paths = []
    n_written = 0
    for angle in cfg.ANGLES:
        vort_folder = os.path.join(cfg.VORTICITY_DIR, f"vort_a{angle}")
        lift_folder = os.path.join(cfg.LIFT_DIR, f"lift_a{angle}")
        if not os.path.isdir(vort_folder):
            print(f"[WARN] Vorticity folder missing: {vort_folder}")
            continue

        mat_files = sorted(glob.glob(
            os.path.join(vort_folder, f"Vort_AoA{angle}_*.mat")))
        for mat_path in mat_files:
            fname = os.path.basename(mat_path)
            case_id = fname.replace(f"Vort_AoA{angle}_", "").replace(".mat", "")
            csv_path = os.path.join(
                lift_folder, f"Lift_AoA{angle}_{case_id}.csv")
            if not os.path.exists(csv_path):
                print(f"[WARN] Lift CSV missing for {fname}, skipping.")
                continue

            vort_npy = os.path.join(npy_dir, f"Vort_AoA{angle}_{case_id}.npy")
            cl_npy   = os.path.join(npy_dir, f"Lift_AoA{angle}_{case_id}.npy")

            need_write = (
                force_recompute
                or not os.path.exists(vort_npy)
                or not os.path.exists(cl_npy)
            )
            if need_write:
                omega = load_vorticity(mat_path)
                cl    = load_lift(csv_path)
                if omega.shape[0] != cl.shape[0]:
                    print(f"[WARN] Time mismatch for AoA{angle}_{case_id}: "
                          f"omega={omega.shape[0]}, cl={cl.shape[0]}. Skipping.")
                    continue
                np.save(vort_npy, omega.astype(np.float32))
                np.save(cl_npy,   cl.astype(np.float32))
                n_written += 1
                del omega, cl

            case_paths.append((angle, case_id, vort_npy, cl_npy))

    case_paths.sort(key=lambda t: (t[0], t[1]))
    if n_written:
        print(f"[ORIG-STATEFUL] Wrote {n_written} new per-case .npy pairs "
              f"to {npy_dir}")
    print(f"[ORIG-STATEFUL] Enumerated {len(case_paths)} cases "
          f"across angles {cfg.ANGLES}")
    return case_paths


def _load_and_encode_original_stateful(cfg, force_recompute=False):
    """Stateful TBPTT pipeline for DATASET_TYPE='original'.

    Reuses the streaming infrastructure built for the interpolated dataset:
    converts each .mat/.csv pair into a per-case .npy pair once, then
    computes norm_scale and the delta threshold by streaming through those
    files. Returns the same case_paths / norm_stats / threshold contract that
    `_run_stateful_tbptt` expects.
    """
    rd_params = load_rd_params(cfg.RD_PARAMS_PATH, cfg.ANGLES)

    cache_path = cfg.CACHE_PATH
    # Separate manifest filename so it doesn't collide with the random_window
    # spike-tensor cache that lives in the same folder.
    manifest_path = cache_path.replace('processed_', 'manifest_stateful_')
    if manifest_path == cache_path:
        manifest_path = os.path.join(
            os.path.dirname(cache_path),
            'manifest_stateful_' + os.path.basename(cache_path),
        )
    manifest_gz = manifest_path + '.gz'
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    actual_cache = manifest_gz if os.path.exists(manifest_gz) else manifest_path

    saved = None
    if os.path.exists(actual_cache) and not force_recompute:
        print(f"\n[CACHE] Loading original-stateful manifest from "
              f"'{os.path.basename(actual_cache)}'...")
        if actual_cache.endswith('.gz'):
            with gzip.open(actual_cache, 'rb') as f:
                saved = torch.load(f, map_location='cpu', weights_only=False)
        else:
            saved = torch.load(actual_cache, map_location='cpu',
                               weights_only=False)
        cached_encoding = saved.get('encoding', None)
        if cached_encoding is not None and cached_encoding != cfg.ENCODING:
            raise RuntimeError(
                f"[CACHE] Encoding mismatch: cache='{cached_encoding}' "
                f"vs cfg='{cfg.ENCODING}'. Delete the manifest cache or set "
                f"force_recompute=True."
            )
        case_paths = saved['case_paths']
        norm_stats = saved['norm_stats']
        threshold  = saved['threshold']

        # Sanity-check that the .npy files referenced by the cached manifest
        # still exist. If the user wiped per_case_npy/ we have to rebuild.
        missing = [p for _, _, p, _ in case_paths if not os.path.exists(p)]
        if missing:
            print(f"[CACHE] {len(missing)} per-case .npy files missing — "
                  f"rebuilding manifest.")
            saved = None

    # `_pre_compute_case_split_map` is deterministic (fixed seed in random_case
    # mode), so we compute the split once and reuse it for both the
    # train-only threshold scan and the final return.
    case_split_map = _pre_compute_case_split_map(cfg, rd_params)

    if saved is None:
        case_paths = _list_original_cases_as_npy(
            cfg, force_recompute=force_recompute)
        if not case_paths:
            raise RuntimeError(
                f"[ORIG-STATEFUL] No cases found under {cfg.VORTICITY_DIR}"
            )

        percentile = getattr(cfg, 'NORM_PERCENTILE', 100)
        norm_stats = compute_norm_scale_streaming(case_paths, percentile)

        threshold = None
        if cfg.ENCODING in ('delta', 'rate+delta'):
            train_keys = {k for k, v in case_split_map.items() if v == 'train'}
            threshold, _, _ = compute_delta_threshold_streaming(
                case_paths, norm_stats['norm_scale'],
                cfg.THRESHOLD_FACTOR, train_keys,
            )

        save_dict = {
            'case_paths': case_paths,
            'norm_stats': norm_stats,
            'threshold': threshold,
            'bins': getattr(cfg, 'BINS', 1),
            'encoding': cfg.ENCODING,
        }
        print(f"[CACHE] Saving stateful manifest to "
              f"'{os.path.basename(manifest_gz)}'...")
        with gzip.open(manifest_gz, 'wb', compresslevel=4) as f:
            torch.save(save_dict, f)
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

    train_cases = {k for k, v in case_split_map.items() if v == 'train'}
    val_cases   = {k for k, v in case_split_map.items() if v == 'val'}
    test_cases  = {k for k, v in case_split_map.items() if v == 'test'}
    print(f"[SPLIT] Mode: {getattr(cfg, 'SPLIT_MODE', 'gust_intensity')} "
          f"(original-stateful)")
    print(f"[SPLIT] Cases — train={len(train_cases)} val={len(val_cases)} "
          f"test={len(test_cases)}")

    # Case-level labels and split indices so downstream consumers (test.py,
    # write_dataset_split_file) can resolve test/val/train cases without
    # needing a windowed sequence layout.
    case_labels = [(a, c) for a, c, _, _ in case_paths]
    split_indices = {
        'train': [i for i, k in enumerate(case_labels) if k in train_cases],
        'val':   [i for i, k in enumerate(case_labels) if k in val_cases],
        'test':  [i for i, k in enumerate(case_labels) if k in test_cases],
    }

    return {
        'case_paths': case_paths,
        'case_labels': case_labels,
        'case_split_map': case_split_map,
        'split_indices': split_indices,
        'rd_params': rd_params,
        'norm_stats': norm_stats,
        'threshold': threshold,
        'bins': getattr(cfg, 'BINS', 1),
        # Compatibility fields — unused on the stateful path but consumers
        # (e.g. test.py) may probe them.
        'train_dataset': None,
        'val_dataset': None,
        'test_dataset': None,
        'spikes': None,
        'targets': None,
    }


# ==========================================
# MAIN ENTRY POINT
# ==========================================
def load_and_encode(cfg, force_recompute=False):
    """
    Main entry point. Loads raw data, normalizes, encodes spikes, caches
    results, and returns train/val/test splits.

    Args:
        cfg: config namespace with required attributes:
            VORTICITY_DIR    — path to vorticity/ folder
            LIFT_DIR         — path to lift/ folder
            RD_PARAMS_PATH   — path to RD_list_FT2023.xlsx
            CACHE_PATH       — path for the cached .pt file
            ANGLES           — list of angles to load, e.g. [20, 30, 40, 50, 60]
            TIME_STEPS       — frames per sequence
            ENCODING         — "rate" | "delta" | "latency"
            BINS             — rate coding bins (default 1)
            THRESHOLD_FACTOR — multiplier for delta threshold (default 1.4)
            VAL_RATIO        — fraction of RD cases per angle assigned to val
            TEST_RATIO       — fraction of RD cases per angle assigned to test
        force_recompute: if True, ignore cache and reprocess from raw data

    Returns:
        dict with keys:
            'train_dataset', 'val_dataset', 'test_dataset': Subset datasets
            'split_indices': dict with 'train', 'val', 'test' index lists
            'case_split_map': dict mapping (angle, case_id) → split name
            'spikes': (N, T_spk, 2, Ny, Nx) full spike tensor
            'targets': (N, T) full lift target tensor
            'norm_stats': normalization constants
            'case_labels': list of (angle, case_id) per sequence
            'rd_params': gust parameter dict from load_rd_params()
            'threshold': delta threshold or None
            'bins': rate coding bins
    """
    if getattr(cfg, 'DATASET_TYPE', 'original') == 'interpolated':
        return _load_and_encode_interpolated(cfg, force_recompute)

    if getattr(cfg, 'TRAINING_MODE', 'random_window') == 'stateful_tbptt':
        return _load_and_encode_original_stateful(cfg, force_recompute)

    cache_path = cfg.CACHE_PATH
    cache_gz = cache_path + '.gz'

    # Load gust parameters (needed for splitting, not cached)
    rd_params = load_rd_params(cfg.RD_PARAMS_PATH, cfg.ANGLES)
    actual_cache = cache_gz if os.path.exists(cache_gz) else cache_path

    if os.path.exists(actual_cache) and not force_recompute:
        print(f"\n[CACHE] Loading from '{os.path.basename(actual_cache)}'...")
        if actual_cache.endswith('.gz'):
            with gzip.open(actual_cache, 'rb') as f:
                saved = torch.load(f, map_location='cpu', weights_only=False)
        else:
            saved = torch.load(actual_cache, map_location='cpu', weights_only=False)

        cached_encoding = saved.get('encoding', None)
        if cached_encoding is not None and cached_encoding != cfg.ENCODING:
            raise RuntimeError(
                f"[CACHE] Encoding mismatch: cache contains '{cached_encoding}' spikes "
                f"but cfg.ENCODING='{cfg.ENCODING}'. Delete the cache or set "
                f"force_recompute=True."
            )

        train_ds, val_ds, test_ds, split_indices, case_split_map = _dispatch_split(
            cfg, saved['spikes'], saved['targets'], saved['case_labels'], rd_params
        )

        return {
            'train_dataset': train_ds,
            'val_dataset': val_ds,
            'test_dataset': test_ds,
            'split_indices': split_indices,
            'case_split_map': case_split_map,
            'rd_params': rd_params,
            **saved,
        }

    # --- Full processing pipeline ---
    print(f"\n[PIPELINE] Processing raw data...")

    # 1. Load all cases
    cases = load_all_cases(cfg.VORTICITY_DIR, cfg.LIFT_DIR, cfg.ANGLES)

    # 2. Normalize vorticity
    percentile = getattr(cfg, 'NORM_PERCENTILE', 100)
    norm_stats = normalize_vorticity(cases, percentile=percentile)

    stride = getattr(cfg, 'STRIDE', None)        # None = non-overlapping (backward compat)
    bins = getattr(cfg, 'BINS', 1)
    warmup_steps = getattr(cfg, 'WARMUP_STEPS', 0)
    threshold = None

    if cfg.ENCODING == 'delta':
        # ---- DELTA PIPELINE: encode full cases BEFORE windowing ----
        # This avoids artificial spike bursts at every window boundary.
        # The burst from padding=True only occurs at t=0 of the full
        # simulation and is optionally zeroed out.

        # 3a. Pre-compute case split to identify training cases
        pre_split_map = _pre_compute_case_split_map(cfg, rd_params)

        # 3b. Compute threshold from TRAINING cases only
        delta_sum = 0.0
        delta_count = 0
        n_train_cases = 0
        for c in cases:
            key = (c['angle'], c['case_id'])
            if pre_split_map.get(key) != 'train':
                continue
            diff = np.abs(np.diff(c['omega'], axis=0))
            delta_sum += float(diff.sum())
            delta_count += diff.size
            n_train_cases += 1
            del diff
        if delta_count == 0:
            raise RuntimeError(
                "[ENCODE] No training cases found for delta threshold computation. "
                "Check SPLIT_MODE and ANGLES settings."
            )
        mean_abs_delta = delta_sum / delta_count
        threshold = mean_abs_delta * cfg.THRESHOLD_FACTOR
        print(f"[ENCODE] Delta threshold = {threshold:.6f}"
              f"  (mean |Δω| = {mean_abs_delta:.6f} × {cfg.THRESHOLD_FACTOR},"
              f" from {n_train_cases} training cases)")

        # 3c. Delta-encode each case's full vorticity sequence
        zero_first = getattr(cfg, 'ZERO_FIRST_FRAME', True)
        print(f"[ENCODE] Encoding vorticity (delta, full-case-first"
              f"{', zero_first_frame' if zero_first else ''})...")
        for c in cases:
            c['spikes'] = _delta_encode_full_case(
                c['omega'], threshold, zero_first_frame=zero_first)
            c['omega'] = None   # free raw vorticity
        gc.collect()

        # 3d. Window encoded spikes into fixed-length sequences
        spikes, cl_seqs, case_labels = create_spike_sequences(
            cases, cfg.TIME_STEPS, stride=stride, warmup_steps=warmup_steps)
        del cases
        gc.collect()

    elif cfg.ENCODING == 'rate+delta':
        # ---- HYBRID RATE+DELTA PIPELINE: encode full cases BEFORE windowing ----
        # Rate channels capture absolute amplitude; delta channels capture temporal
        # transitions. Encoding before windowing avoids artificial delta bursts at
        # every window boundary. The global delta threshold is computed from the
        # training set only, so it does not adapt to individual samples.

        # 3a. Pre-compute case split to identify training cases
        pre_split_map = _pre_compute_case_split_map(cfg, rd_params)

        # 3b. Compute global delta threshold from TRAINING cases only
        delta_sum = 0.0
        delta_count = 0
        n_train_cases = 0
        for c in cases:
            key = (c['angle'], c['case_id'])
            if pre_split_map.get(key) != 'train':
                continue
            diff = np.abs(np.diff(c['omega'], axis=0))
            delta_sum += float(diff.sum())
            delta_count += diff.size
            n_train_cases += 1
            del diff
        if delta_count == 0:
            raise RuntimeError(
                "[ENCODE] No training cases found for delta threshold computation. "
                "Check SPLIT_MODE and ANGLES settings."
            )
        mean_abs_delta = delta_sum / delta_count
        threshold = mean_abs_delta * cfg.THRESHOLD_FACTOR
        print(f"[ENCODE] Delta threshold = {threshold:.6f}"
              f"  (mean |Δω| = {mean_abs_delta:.6f} × {cfg.THRESHOLD_FACTOR},"
              f" from {n_train_cases} training cases)")

        # 3c. Hybrid-encode each case's full vorticity sequence
        gamma = getattr(cfg, 'GAMMA', 1.0)
        gain  = getattr(cfg, 'GAIN', 1.0)
        zero_first = getattr(cfg, 'ZERO_FIRST_FRAME', True)
        print(f"[ENCODE] Encoding vorticity (rate+delta, full-case-first, "
              f"gamma={gamma}, gain={gain}"
              f"{', zero_first_frame' if zero_first else ''})...")
        for c in cases:
            c['spikes'] = _hybrid_encode_full_case(
                c['omega'], threshold,
                gamma=gamma, gain=gain, zero_first_frame=zero_first)
            c['omega'] = None   # free raw vorticity
        gc.collect()

        # 3d. Window encoded spikes into fixed-length sequences
        spikes, cl_seqs, case_labels = create_spike_sequences(
            cases, cfg.TIME_STEPS, stride=stride, warmup_steps=warmup_steps)
        del cases
        gc.collect()

    else:
        # ---- RATE / LATENCY PIPELINE: window first, then encode ----
        # Rate and latency encoding are stateless per-frame, so encoding
        # before or after windowing gives identical results.

        # 3. Create sequences (with optional overlap via stride)
        omega_seqs, cl_seqs, case_labels = create_sequences(
            cases, cfg.TIME_STEPS, stride=stride, warmup_steps=warmup_steps)
        del cases
        gc.collect()

        # 4. Encode vorticity into spikes
        print(f"[ENCODE] Encoding vorticity ({cfg.ENCODING})...")
        gamma = getattr(cfg, 'GAMMA', 1.0)
        gain = getattr(cfg, 'GAIN', 1.0)
        if cfg.ENCODING == 'rate' and (gamma != 1.0 or gain != 1.0):
            print(f"[ENCODE] Amplified mapping: p = min(1, {gain} · |x|^{gamma})")
        spikes = encode_spikes(omega_seqs, cfg.ENCODING, threshold=threshold,
                               bins=bins, chunk_size=10, gamma=gamma, gain=gain)
        del omega_seqs
        gc.collect()

    # Print spike statistics over all C channels (chunked to avoid int64 OOM)
    _N, _T, _C, _H, _W = spikes.shape
    ch_totals = [0] * _C
    for i in range(0, _N, 100):
        chunk = spikes[i:i+100].to(torch.int64)
        for ch in range(_C):
            ch_totals[ch] += chunk[:, :, ch].sum().item()
    spatial_temporal    = _N * _T * _H * _W          # per-channel space-time volume
    ch_rates            = [t / spatial_temporal * 100 for t in ch_totals]
    channel_mean_rate   = sum(ch_rates) / _C
    combined_pixel_rate = sum(ch_rates)
    if _C == 4:
        # rate+delta: report rate and delta channels separately
        rate_ch_mean       = (ch_rates[0] + ch_rates[1]) / 2
        rate_combined_px   = ch_rates[0] + ch_rates[1]
        delta_ch_mean      = (ch_rates[2] + ch_rates[3]) / 2
        delta_combined_px  = ch_rates[2] + ch_rates[3]
        print(f"[ENCODE] Spikes: {tuple(spikes.shape)}  "
              f"rate_channel_mean={rate_ch_mean:.2f}%  rate_combined_pixel={rate_combined_px:.2f}%  "
              f"delta_channel_mean={delta_ch_mean:.2f}%  delta_combined_pixel={delta_combined_px:.2f}%")
    else:
        print(f"[ENCODE] Spikes: {tuple(spikes.shape)}  "
              f"channel_mean={channel_mean_rate:.1f}%  combined_pixel={combined_pixel_rate:.1f}%")
    print(f"[ENCODE] Targets (Cl): {tuple(cl_seqs.shape)}")

    # 6. Cache (split indices are not cached — they are deterministic from rd_params)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_dict = {
        'spikes': spikes,
        'targets': cl_seqs,
        'norm_stats': norm_stats,
        'case_labels': case_labels,
        'threshold': threshold,
        'bins': bins,
        'encoding': cfg.ENCODING,
    }
    print(f"[CACHE] Saving compressed to '{os.path.basename(cache_gz)}'...")
    with gzip.open(cache_gz, 'wb', compresslevel=4) as f:
        torch.save(save_dict, f)
    del save_dict
    # Remove old uncompressed cache if it exists
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"[CACHE] Removed old uncompressed cache")

    # 7. Split dataset
    train_ds, val_ds, test_ds, split_indices, case_split_map = _dispatch_split(
        cfg, spikes, cl_seqs, case_labels, rd_params
    )

    return {
        'train_dataset': train_ds,
        'val_dataset': val_ds,
        'test_dataset': test_ds,
        'split_indices': split_indices,
        'case_split_map': case_split_map,
        'rd_params': rd_params,
        'spikes': spikes,
        'targets': cl_seqs,
        'norm_stats': norm_stats,
        'case_labels': case_labels,
        'threshold': threshold,
        'bins': bins,
    }


# ==========================================
# Standalone test
# ==========================================
if __name__ == "__main__":
    raise RuntimeError(
        "pre_encoder.py is not meant to run standalone. "
        "Run train_SNN.py instead, which provides the config namespace."
    )
