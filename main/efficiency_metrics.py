"""
efficiency_metrics.py — Computational & energy-efficiency metrics for a trained
Cl-reconstruction model.

This script complements the accuracy metrics in test.py by quantifying the
computational cost of a model on the test set, following the thesis section
"Computational and Energy-Efficiency Metrics":

    1. Number of parameters            N_theta
    2. Spike activity (firing density) rho_l  (per spiking layer)
    3. Multiply-accumulate (MAC) and accumulate (AC) operations
    4. Estimated inference energy       E_hat  (Horowitz 45 nm costs)
    5. Weight and temporal-state memory M_weights, M_state, M_persistent

How to use
----------
    1. Set MODEL below to a checkpoint folder under training/, prefixed with
       the training-type subfolder it was written into, e.g.
           MODEL = "SNN_training/RateOnly_InterpolatedN4_Stateful_Nz16_Best"
           MODEL = "Parameter_matched_training/CNN_ParamMatched_Nz16"
           MODEL = "Energy_matched_training/CNN_EnergyMatched_Nz16"
       The model may be spiking (the SNN encoder, SNN.py) or a fully non-spiking
       CNN baseline. The code auto-detects which it is from the
       presence of spiking (LIF) layers, so no per-metric special-casing is
       needed — a dense CNN simply reports every weighted layer as MAC-only,
       zero AC, and no temporal-state memory.
    2. Run:  python efficiency_metrics.py
    3. A timestamped folder is created under efficiency_results/ holding
       efficiency_metric_summary.txt with all metrics (and a short description of
       the evaluated model on top). Folder names use the same complete-date
       convention as results/, so they can be renamed by hand afterwards.

Method notes
------------
  * Firing density rho_l is measured empirically by running the model over every
    test encounter (single forward pass per encounter, full T_i time steps).
  * MAC counts are analytical (dense conv/linear formulas). AC counts for a
    spike-driven weighted layer use N_AC = rho_pre * N_MAC,dense, where rho_pre
    is the measured firing density of the spike train ENTERING that layer (so a
    layer fed by pooled spikes correctly uses the pooled density).
  * A weighted layer is classified MAC vs AC automatically: if its presynaptic
    activations are binary (spikes) it is an AC layer, otherwise MAC. This is
    what lets the same code handle a spiking encoder and a dense CNN.
  * Normalisation (GroupNorm/LayerNorm), LIF membrane updates and the encoder
    are NOT counted as MAC/AC — matching the thesis, which associates MACs with
    conv/feedforward layers and ACs with spike-driven weighted layers only.
  * E_hat is an analytical estimate (32-bit op costs applied to all configs),
    not a hardware measurement.
"""

import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'models'))


# ==========================================
# CONFIGURATION
# ==========================================
# Checkpoint folder under training/ (contains config_snapshot.txt and the .pth
# weights), prefixed with its training-type subfolder — SNN_training/,
# Parameter_matched_training/, or Energy_matched_training/. This is the SAME
# identifier used by test.py's model_folder_name.
MODEL = "Energy_matched_training/CNN_EnergyMatched_Nz16"

# Model family. "auto" infers it from the checkpoint (any LIF layer -> snn,
# otherwise cnn):  "auto" | "snn" | "cnn". The metric code is family-agnostic;
# this flag only guards the input pipeline (spike-encoded vs raw frames) and the
# report wording. The parameter-matched CNN checkpoints report as "cnn".
MODEL_TYPE = "auto"

# Weights file inside the checkpoint folder. None -> best.pth (with a *best*.pth
# fallback). Runs trained with VAL_RATIO=0 have no best.pth, so name a periodic
# checkpoint explicitly, e.g. "epoch_100.pth".
WEIGHTS_FILE = "epoch_100.pth"

# Batch size B used only for the temporal-state memory estimate M_state,B.
STATE_BATCH_SIZE = 1

# Frames per forward call when measuring a non-spiking CNN (inference only;
# purely a throughput knob — does not affect the reported per-frame metrics).
FRAME_BATCH = 1024

# Bit widths for the memory estimates.
BITS_WEIGHT = 32          # b_w : bits per stored parameter
BITS_STATE = 32           # b_m : bits per membrane-potential value

# Horowitz (45 nm, 0.9 V) arithmetic-energy costs.
E_MAC_PJ = 4.6            # eps_MAC = 3.7 (mult) + 0.9 (add) pJ
E_AC_PJ = 0.9            # eps_AC  = 0.9 pJ (accumulate only)


# ==========================================
# REUSED INFRASTRUCTURE (from test.py / pre_encoder.py)
# ==========================================
import test as _test                       # noqa: E402  (config reconstruction + device)
from pre_encoder import (                   # noqa: E402
    load_and_encode,
    load_rd_params,
    list_interpolated_cases,
    _pre_compute_case_split_map,
    _rate_encode_full_case,
    _delta_encode_full_case,
    _hybrid_encode_full_case,
)


# ==========================================
# MODEL CONSTRUCTION
# ==========================================
def _build_model(cfg, device):
    """Instantiate + load the model described by cfg: the spiking SNN or one of
    the parameter-/energy-matched non-spiking CNN baselines."""
    v = cfg.MODEL_TYPE

    # ---- Non-spiking CNN baselines ----
    if v in ("cnn_param_matched", "cnn_energy_matched"):
        if v == "cnn_param_matched":
            from parameter_matched import ParameterMatchedCNN
            model = ParameterMatchedCNN(n_z=cfg.N_Z, in_channels=cfg.IN_CHANNELS)
        else:
            from energy_matched import EnergyMatchedCNN
            model = EnergyMatchedCNN(n_z=cfg.N_Z, in_channels=cfg.IN_CHANNELS)
        model = model.to(device)
        state_dict = torch.load(cfg.BEST_MODEL_PATH, map_location='cpu',
                                weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [COMPAT] Missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"  [COMPAT] Unexpected keys in checkpoint: {unexpected}")
        model.eval()
        return model

    # ---- Spiking SNN ----
    from SNN import SNN_Model
    model = SNN_Model(
        n_z=cfg.N_Z,
        in_channels=cfg.IN_CHANNELS,
        beta_init=cfg.BETA_INIT,
        fc_threshold=float(getattr(cfg, 'FC_THRESHOLD', 0.5)),
        gradient_checkpointing=False,
    )

    model = model.to(device)
    state_dict = torch.load(cfg.BEST_MODEL_PATH, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [COMPAT] Missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"  [COMPAT] Unexpected keys in checkpoint: {unexpected}")
    model.eval()
    return model


# ==========================================
# ENCODING (single forward pass per encounter)
# ==========================================
def _encode_case(omega_norm, cfg, threshold):
    """Spike-encode one case's normalised, clipped vorticity to (Nt, C, H, W)."""
    enc = cfg.ENCODING
    gamma = float(getattr(cfg, 'GAMMA', 1.0))
    gain = float(getattr(cfg, 'GAIN', 1.0))
    zero_first = bool(getattr(cfg, 'ZERO_FIRST_FRAME', True))
    if enc == 'rate':
        return _rate_encode_full_case(omega_norm, gamma=gamma, gain=gain)
    if enc == 'delta':
        return _delta_encode_full_case(omega_norm, threshold, zero_first_frame=zero_first)
    if enc == 'rate+delta':
        return _hybrid_encode_full_case(omega_norm, threshold, gamma=gamma, gain=gain,
                                        zero_first_frame=zero_first)
    raise ValueError(f"Unsupported ENCODING={enc!r}")


# ==========================================
# EFFICIENCY PROBE (forward hooks)
# ==========================================
class EfficiencyProbe:
    """Accumulates per-layer spike counts and operation counts over inference.

    - Every Conv2d / Linear is hooked to record its dense-MAC count per time
      step, whether its presynaptic activations are binary (=> AC layer) or
      continuous (=> MAC layer), and the running total of presynaptic spikes.
    - Every LIF (snn.Leaky) is hooked to record its output firing density.
    All tensor reductions stay on-device; .item() is only called at the end.
    """

    def __init__(self, model):
        self.handles = []
        # Weighted layers (conv / linear), keyed by module name.
        self.mac_dense_ts = {}   # dense MAC per time step (constant per layer)
        self.in_numel_ts = {}    # presynaptic elements per time step (per sample)
        self.is_binary = {}      # presynaptic activations binary (spikes)?
        self.kind = {}           # 'conv' | 'linear'
        self.calls = {}          # total samples seen (sum of batch sizes = frames/steps)
        self._presyn_spk = {}    # running sum of presynaptic spikes (device tensor)
        # Spiking (LIF) layers, keyed by module name.
        self.lif_kind = {}       # marks a module as a LIF spike source
        self._lif_spk = {}       # running spike sum (device tensor)
        self.lif_cnt = {}        # running possible-activation count (per sample)
        self._register(model)

    def _register(self, model):
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.handles.append(m.register_forward_hook(self._weighted_hook(name)))
            elif isinstance(m, snn.Leaky):
                self.handles.append(m.register_forward_hook(self._lif_hook(name)))

    def _weighted_hook(self, name):
        def hook(mod, inp, out):
            x = inp[0]
            if name not in self.mac_dense_ts:
                if isinstance(mod, nn.Conv2d):
                    out_spatial = int(out.shape[-2] * out.shape[-1])
                    kh, kw = mod.kernel_size
                    c_in = mod.in_channels // mod.groups
                    mac = mod.out_channels * out_spatial * c_in * kh * kw
                    self.kind[name] = 'conv'
                else:  # nn.Linear
                    mac = mod.in_features * mod.out_features
                    self.kind[name] = 'linear'
                self.mac_dense_ts[name] = int(mac)
                # per-sample presynaptic element count (assumes batch dim first)
                self.in_numel_ts[name] = int(x[0].numel())
                # binary presynaptic activations => spike-driven (AC) layer
                self.is_binary[name] = bool(
                    torch.logical_or(x == 0, x == 1).all().item())
                self._presyn_spk[name] = torch.zeros((), device=x.device)
                self.calls[name] = 0
            # Count samples (= batch size), so per-step accounting is correct for
            # both the SNN (B=1 per time step) and the CNN (B frames per call).
            self.calls[name] += int(out.shape[0])
            if self.is_binary[name]:
                self._presyn_spk[name] = self._presyn_spk[name] + x.sum()
        return hook

    def _lif_hook(self, name):
        def hook(mod, inp, out):
            spk = out[0] if isinstance(out, tuple) else out
            if name not in self._lif_spk:
                self.lif_kind[name] = 'lif'
                self._lif_spk[name] = torch.zeros((), device=spk.device)
                self.lif_cnt[name] = 0
            self._lif_spk[name] = self._lif_spk[name] + spk.sum()
            # per-sample neuron count x batch size = total possible activations
            self.lif_cnt[name] += int(spk[0].numel()) * int(spk.shape[0])
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    # ---- Finalised accessors (call after inference) ----
    def presyn_spikes(self, name):
        return float(self._presyn_spk[name].item())

    def lif_density(self, name):
        cnt = self.lif_cnt[name]
        return float(self._lif_spk[name].item()) / cnt if cnt else 0.0


# ==========================================
# INFERENCE OVER THE TEST SET
# ==========================================
def measure_snn(model, data, cfg, device):
    """SNN path: spike-encode each test encounter and forward it one time step at
    a time (the model unrolls internally), collecting efficiency stats."""
    case_paths = data['case_paths']
    by_key = {(a, c): (vp, cp) for a, c, vp, cp in case_paths}
    split = data['case_split_map']
    norm_scale = float(data['norm_stats']['norm_scale'])
    threshold = data.get('threshold', None)

    test_cases = sorted(k for k, v in split.items() if v == 'test')
    if not test_cases:
        raise RuntimeError("No test cases found in case_split_map.")

    probe = EfficiencyProbe(model)
    in_spk_total = 0.0     # encoded-input spikes
    in_cnt_total = 0.0     # encoded-input possible activations
    t_total = 0
    n_enc = 0

    print(f"\n>>> Measuring efficiency on {len(test_cases)} test encounters "
          f"(single forward pass, full T per encounter)...")
    print(f"    norm_scale={norm_scale:.4f}  threshold={threshold}")

    model.eval()
    with torch.inference_mode():
        for angle, case_id in test_cases:
            if (angle, case_id) not in by_key:
                print(f"    [WARN] test case ({angle}, {case_id}) missing — skipped.")
                continue
            vort_path, _cl_path = by_key[(angle, case_id)]

            omega = np.array(np.load(vort_path, mmap_mode='r'), dtype=np.float32)
            omega *= 1.0 / (norm_scale + 1e-8)
            np.clip(omega, -1.0, 1.0, out=omega)

            spikes = _encode_case(omega, cfg, threshold)          # (Nt, C, H, W)
            del omega
            x_input = spikes.unsqueeze(1).to(device, dtype=torch.float32)  # (T,1,C,H,W)
            del spikes

            in_spk_total += float(x_input.sum().item())
            in_cnt_total += float(x_input.numel())               # B=1

            model(x_input)

            t_total += int(x_input.shape[0])
            n_enc += 1
            print(f"    [{n_enc:>2}/{len(test_cases)}] AoA{angle} {case_id}: "
                  f"T={x_input.shape[0]}")
            del x_input
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    probe.remove()
    stats = {
        'test_cases': test_cases,
        'n_enc': n_enc,
        't_total': t_total,
        'input_density': (in_spk_total / in_cnt_total) if in_cnt_total else 0.0,
    }
    return probe, stats


def measure_cnn(model, test_cases, by_key, device):
    """CNN path: feed the RAW continuous vorticity frames of each test encounter
    (no encoding, no scaling) through the model in batches, collecting stats.

    Every conv/linear is auto-classified MAC (continuous input) by the probe, so
    AC = 0 and there is no temporal-state memory.
    """
    probe = EfficiencyProbe(model)
    t_total = 0
    n_enc = 0

    print(f"\n>>> Measuring efficiency on {len(test_cases)} test encounters "
          f"(per-frame CNN, batches of {FRAME_BATCH})...")

    model.eval()
    with torch.inference_mode():
        for angle, case_id in test_cases:
            if (angle, case_id) not in by_key:
                print(f"    [WARN] test case ({angle}, {case_id}) missing — skipped.")
                continue
            vort_path, _cl_path = by_key[(angle, case_id)]

            omega = np.load(vort_path, mmap_mode='r')          # (Nt, H, W)
            nt = int(omega.shape[0])
            for i in range(0, nt, FRAME_BATCH):
                frames = np.array(omega[i:i + FRAME_BATCH], dtype=np.float32)
                x = torch.from_numpy(frames).unsqueeze(1).to(device)  # (B,1,H,W) raw
                model(x)
                del x, frames
            del omega

            t_total += nt
            n_enc += 1
            print(f"    [{n_enc:>2}/{len(test_cases)}] AoA{angle} {case_id}: "
                  f"frames={nt}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    probe.remove()
    stats = {
        'test_cases': test_cases,
        'n_enc': n_enc,
        't_total': t_total,
        'input_density': None,     # continuous input — no spike density
    }
    return probe, stats


# ==========================================
# METRIC AGGREGATION
# ==========================================
def compute_metrics(model, probe, stats, cfg, device):
    """Turn the raw probe accumulators into the reported efficiency metrics."""
    n_enc = stats['n_enc']
    t_total = stats['t_total']

    # ---- 1. Parameters ----
    n_params = int(sum(p.numel() for p in model.parameters()))
    n_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    n_neuron_params = int(sum(
        p.numel() for n, p in model.named_parameters()
        if n.endswith('threshold') or n.endswith('beta')))

    # ---- 2. Spike activity (firing density per spiking layer) ----
    lif_densities = [(name, probe.lif_density(name)) for name in probe.lif_kind]

    # ---- 3. MAC / AC operations ----
    # Per weighted layer: totals over the whole test set. AC layers use the
    # measured presynaptic firing density; MAC layers are dense every step.
    layer_ops = []          # list of dicts, in model order
    total_mac = 0.0         # continuous MACs (test-set total)
    total_ac = 0.0          # spike-driven ACs (test-set total)
    total_dense = 0.0       # dense-equivalent MACs if every layer were dense
    for name in probe.mac_dense_ts:
        T_l = probe.calls[name]
        mac_ts = probe.mac_dense_ts[name]
        dense_total = mac_ts * T_l
        total_dense += dense_total
        entry = {'name': name, 'kind': probe.kind[name], 'mac_ts': mac_ts,
                 'steps': T_l}
        if probe.is_binary[name]:
            spk = probe.presyn_spikes(name)
            fanout = mac_ts / probe.in_numel_ts[name]
            ac_total = spk * fanout
            rho_pre = spk / (probe.in_numel_ts[name] * T_l) if T_l else 0.0
            entry.update(op='AC', rho_pre=rho_pre, ops_total=ac_total,
                         dense_total=dense_total)
            total_ac += ac_total
        else:
            entry.update(op='MAC', rho_pre=None, ops_total=dense_total,
                         dense_total=dense_total)
            total_mac += dense_total
        layer_ops.append(entry)

    # ---- 4. Estimated inference energy (pJ) ----
    energy_total_pj = total_mac * E_MAC_PJ + total_ac * E_AC_PJ
    energy_dense_pj = total_dense * E_MAC_PJ     # same model run as a dense net

    # ---- 5. Memory ----
    m_weights = n_params * BITS_WEIGHT / 8.0     # bytes
    if hasattr(model, 'init_state'):
        state = model.init_state(STATE_BATCH_SIZE, device)
        state_elems = int(sum(v.numel() for v in state.values()))
        n_state_layers = len(state)
    else:
        state_elems = 0
        n_state_layers = 0
    m_state = state_elems * BITS_STATE / 8.0     # bytes (already includes B)
    m_persistent = m_weights + m_state

    return {
        'n_params': n_params,
        'n_trainable': n_trainable,
        'n_neuron_params': n_neuron_params,
        'input_density': stats['input_density'],
        'lif_densities': lif_densities,
        'layer_ops': layer_ops,
        'total_mac': total_mac,
        'total_ac': total_ac,
        'total_dense': total_dense,
        'energy_total_pj': energy_total_pj,
        'energy_dense_pj': energy_dense_pj,
        'm_weights': m_weights,
        'm_state': m_state,
        'm_persistent': m_persistent,
        'state_elems': state_elems,
        'n_state_layers': n_state_layers,
        'n_enc': n_enc,
        't_total': t_total,
    }


# ==========================================
# FORMATTING HELPERS
# ==========================================
def _fmt_count(n):
    """Human-readable operation/parameter count."""
    n = float(n)
    for unit, scale in (('G', 1e9), ('M', 1e6), ('k', 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:,.3f} {unit}"
    return f"{n:,.0f}"


def _fmt_energy(pj):
    """Adaptive energy unit from a value in picojoules."""
    for unit, scale in (('J', 1e12), ('mJ', 1e9), ('uJ', 1e6), ('nJ', 1e3)):
        if abs(pj) >= scale:
            return f"{pj / scale:,.4f} {unit}"
    return f"{pj:,.2f} pJ"


def _fmt_bytes(b):
    """Adaptive byte unit."""
    for unit, scale in (('MB', 1024**2), ('KB', 1024)):
        if abs(b) >= scale:
            return f"{b / scale:,.3f} {unit}"
    return f"{b:,.0f} B"


# ==========================================
# REPORT
# ==========================================
def build_report(cfg, metrics, is_snn, device):
    n_enc = metrics['n_enc']
    t_total = metrics['t_total']
    L = []
    add = L.append

    fam = "SNN (spiking)" if is_snn else "CNN (non-spiking)"
    add("=" * 70)
    add("EFFICIENCY METRICS SUMMARY")
    add("=" * 70)
    add(f"Model folder      : {MODEL}")
    add(f"Weights file      : {os.path.basename(cfg.BEST_MODEL_PATH)}")
    add(f"Model family      : {fam}")
    add(f"Model type        : {cfg.MODEL_TYPE}")
    add(f"Encoding          : {cfg.ENCODING}  (in_channels={cfg.IN_CHANNELS})")
    add(f"Latent dim N_z    : {cfg.N_Z}")
    add(f"Dataset           : {cfg.DATASET_TYPE}"
        + (f" (N={cfg.N_INTERPOLATED})" if cfg.DATASET_TYPE == 'interpolated' else ""))
    add(f"Test encounters   : {n_enc}")
    add(f"Total time steps  : {t_total:,}  (sum of T_i over encounters)")
    add(f"Device            : {device}")
    add(f"Generated         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add("-" * 70)
    add("These are analytical estimates of the arithmetic workload and its")
    add("energy/memory footprint on the test set. Firing densities are measured")
    add("empirically; MAC/AC counts follow the thesis formulas; the energy is a")
    add("32-bit-op approximation (Horowitz 45 nm), not a hardware measurement.")
    add("")

    # ---- 1. Parameters ----
    add("1. NUMBER OF PARAMETERS")
    add("-" * 70)
    add(f"   Total parameters  N_theta : {metrics['n_params']:,} "
        f"({_fmt_count(metrics['n_params'])})")
    add(f"   Trainable                 : {metrics['n_trainable']:,}")
    if is_snn:
        add(f"   Learnable neuron params   : {metrics['n_neuron_params']:,} "
            f"(LIF threshold + beta, one scalar each per layer)")
    add("")

    # ---- 2. Spike activity ----
    add("2. SPIKE ACTIVITY  (firing density rho_l = spikes / possible activations)")
    add("-" * 70)
    if is_snn:
        add(f"   {'Encoded input':<28s} : {metrics['input_density']:.4f}")
        for name, rho in metrics['lif_densities']:
            add(f"   {name:<28s} : {rho:.4f}")
        if metrics['lif_densities']:
            mean_rho = np.mean([r for _, r in metrics['lif_densities']])
            add(f"   {'mean over LIF layers':<28s} : {mean_rho:.4f}")
    else:
        add("   (non-spiking model — no spike activity)")
    add("")

    # ---- 3. MAC / AC ----
    add("3. ARITHMETIC OPERATIONS  (MAC = continuous, AC = spike-driven)")
    add("-" * 70)
    add(f"   {'Layer':<14s} {'type':<7s} {'op':<4s} {'rho_pre':>8s} "
        f"{'dense MAC/step':>16s} {'ops (test-set)':>16s}")
    for e in metrics['layer_ops']:
        rho = f"{e['rho_pre']:.4f}" if e['rho_pre'] is not None else "   -  "
        add(f"   {e['name']:<14s} {e['kind']:<7s} {e['op']:<4s} {rho:>8s} "
            f"{_fmt_count(e['mac_ts']):>16s} {_fmt_count(e['ops_total']):>16s}")
    add("   " + "-" * 64)
    per_enc = (lambda x: x / n_enc if n_enc else 0.0)
    per_step = (lambda x: x / t_total if t_total else 0.0)
    add(f"   Total MAC : test-set {_fmt_count(metrics['total_mac'])}  |  "
        f"per encounter {_fmt_count(per_enc(metrics['total_mac']))}  |  "
        f"per step {_fmt_count(per_step(metrics['total_mac']))}")
    add(f"   Total AC  : test-set {_fmt_count(metrics['total_ac'])}  |  "
        f"per encounter {_fmt_count(per_enc(metrics['total_ac']))}  |  "
        f"per step {_fmt_count(per_step(metrics['total_ac']))}")
    add(f"   Dense-equiv MAC (all layers dense) : "
        f"per encounter {_fmt_count(per_enc(metrics['total_dense']))}")
    add("")

    # ---- 4. Energy ----
    add("4. ESTIMATED INFERENCE ENERGY  (eps_MAC = "
        f"{E_MAC_PJ} pJ, eps_AC = {E_AC_PJ} pJ)")
    add("-" * 70)
    et = metrics['energy_total_pj']
    add(f"   E_hat per encounter (mean) : {_fmt_energy(per_enc(et))}")
    add(f"   E_hat per time step        : {_fmt_energy(per_step(et))}")
    add(f"   E_hat total (test set)     : {_fmt_energy(et)}")
    if is_snn:
        ed = metrics['energy_dense_pj']
        ratio = (ed / et) if et else float('nan')
        add(f"   Dense-equivalent E (per enc): {_fmt_energy(per_enc(ed))}  "
            f"(x{ratio:.2f} of the spiking estimate)")
    add("")

    # ---- 5. Memory ----
    add("5. MEMORY")
    add("-" * 70)
    add(f"   Weight memory  M_weights          : {_fmt_bytes(metrics['m_weights'])}  "
        f"({metrics['n_params']:,} params x {BITS_WEIGHT}-bit)")
    if is_snn:
        add(f"   Temporal-state memory M_state,B={STATE_BATCH_SIZE}  : "
            f"{_fmt_bytes(metrics['m_state'])}  "
            f"({metrics['state_elems']:,} membrane values x {BITS_STATE}-bit, "
            f"{metrics['n_state_layers']} LIF layers)")
    add(f"   Persistent memory M_persistent    : {_fmt_bytes(metrics['m_persistent'])}")
    add("")
    add("=" * 70)
    return "\n".join(L)


# ==========================================
# OUTPUT LOCATION
# ==========================================
_OUTPUT_SUBFOLDER = {
    "snn":                "SNN_results",
    "cnn_param_matched":  "Parameter_matched_results",
    "cnn_energy_matched": "Energy_matched_results",
}


def create_output_dir(cfg):
    """Timestamped folder under efficiency_results/, grouped by model type —
    not by N_Z — the same organisation as test.create_results_dir uses for
    results/ (SNN_results/, Parameter_matched_results/, Energy_matched_results/)."""
    base = os.path.join(_HERE, '..', 'efficiency_results')
    timestamp = datetime.now().strftime("%B_%d_%Y_%Hh_%Mm")
    subfolder = _OUTPUT_SUBFOLDER[cfg.MODEL_TYPE]
    out_dir = os.path.join(base, subfolder, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ==========================================
# MAIN
# ==========================================
def main():
    ckpt_dir = os.path.join(_HERE, '..', 'training', MODEL)

    # Reconstruct the exact training/encoding config from the checkpoint.
    _test.MODEL_WEIGHTS_FILE = WEIGHTS_FILE       # honoured by _build_cfg_from_checkpoint
    cfg = _test._build_cfg_from_checkpoint(ckpt_dir)

    device = _test.get_device()
    print("=" * 60)
    print(f"  EFFICIENCY METRICS — {MODEL}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model : {cfg.MODEL_TYPE}  encoding={cfg.ENCODING}  "
          f"N_z={cfg.N_Z}  in_channels={cfg.IN_CHANNELS}")

    # Build + load model.
    print("\n>>> Building model...")
    model = _build_model(cfg, device)
    is_snn = any(isinstance(m, snn.Leaky) for m in model.modules())
    if MODEL_TYPE == "snn" and not is_snn:
        raise RuntimeError("MODEL_TYPE='snn' but no LIF layers found in the model.")
    if MODEL_TYPE == "cnn" and is_snn:
        raise RuntimeError("MODEL_TYPE='cnn' but the model contains LIF layers.")
    print(f"    family: {'SNN' if is_snn else 'CNN'} | "
          f"params: {sum(p.numel() for p in model.parameters()):,}")

    # Measure + aggregate. SNN and CNN use different input pipelines but the same
    # probe / metric aggregation.
    t0 = time.time()
    if is_snn:
        # Same encoding pipeline as test.py (needs norm_scale + split).
        print("\n>>> Loading data...")
        data = load_and_encode(cfg)
        probe, stats = measure_snn(model, data, cfg, device)
    else:
        # CNN: raw continuous frames. Reproduce the SNN train/test split from the
        # config (pinned TEST_CASES => deterministic) without spike encoding.
        print("\n>>> Resolving test encounters...")
        rd_params = load_rd_params(cfg.RD_PARAMS_PATH, angles=cfg.ANGLES)
        case_paths = list_interpolated_cases(cfg.INTERP_ROOT, cfg.ANGLES)
        by_key = {(a, c): (vp, cp) for a, c, vp, cp in case_paths}
        split = _pre_compute_case_split_map(cfg, rd_params)
        test_cases = sorted(k for k, v in split.items() if v == 'test')
        if not test_cases:
            raise RuntimeError("No test cases found for the CNN (check cfg split).")
        probe, stats = measure_cnn(model, test_cases, by_key, device)
    metrics = compute_metrics(model, probe, stats, cfg, device)
    print(f"\n>>> Done in {time.time() - t0:.1f}s "
          f"({stats['n_enc']} encounters, {stats['t_total']:,} steps).")

    # Report.
    report = build_report(cfg, metrics, is_snn, device)
    print("\n" + report)

    out_dir = create_output_dir(cfg)
    out_path = os.path.join(out_dir, 'efficiency_metric_summary.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\nSaved: {out_path}")

    # Machine-readable metrics for the comparison plots (efficiency_metrics_plots.py).
    # Raw base units — energy in pJ, operations as counts — so the plotter can
    # convert/round however the user configures it.
    n_enc = metrics['n_enc']
    t_total = metrics['t_total']
    ops_total = metrics['total_mac'] + metrics['total_ac']
    et = metrics['energy_total_pj']
    _pe = (lambda v: v / n_enc if n_enc else 0.0)     # per encounter
    _ps = (lambda v: v / t_total if t_total else 0.0)  # per time step
    data_json = {
        'model_folder': MODEL,
        'model_type': cfg.MODEL_TYPE,
        'family': 'snn' if is_snn else 'cnn',
        'n_params': int(metrics['n_params']),
        'n_encounters': int(n_enc),
        't_total_steps': int(t_total),
        # Energy (picojoules)
        'energy_total_pj': float(et),
        'energy_per_encounter_pj': float(_pe(et)),
        'energy_per_step_pj': float(_ps(et)),
        # Operations (raw counts): MAC (continuous) + AC (spike-driven)
        'total_mac': float(metrics['total_mac']),
        'total_ac': float(metrics['total_ac']),
        'ops_total': float(ops_total),
        'ops_per_encounter': float(_pe(ops_total)),
        'ops_per_step': float(_ps(ops_total)),
        'mac_per_encounter': float(_pe(metrics['total_mac'])),
        'ac_per_encounter': float(_pe(metrics['total_ac'])),
        'mac_per_step': float(_ps(metrics['total_mac'])),
        'ac_per_step': float(_ps(metrics['total_ac'])),
    }
    json_path = os.path.join(out_dir, 'efficiency_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_json, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
