"""
energy_matched.py — Energy-matched non-spiking CNN baseline.

A dense feedforward CNN with the same encoder-to-lift structure as the
parameter-matched CNN (parameter_matched.py), but with its convolutional block
widths reduced so its estimated per-time-step inference energy approaches the
reference SNN's budget. It answers: at approximately the same estimated energy,
does the SNN or a conventional CNN reconstruct Cl more accurately?

Energy matching (data-independent — the CNN is all-MAC, no spike sparsity):
  SNN budget: E_SNN,step = 25.5066 uJ/step
              (measured, efficiency_results/SNN_results/Reference_model_SNN).
  Per-step CNN energy = N_MAC,step * eps_MAC (eps_MAC = 4.6 pJ).
  Chosen widths (c1, c2) = (4, 4), N_z = 16  ->  5.707 M MAC/step
              = 26.25 uJ/step  (+2.9% vs the SNN budget, within ~5% tolerance).
  The listed 2x-expansion candidates cannot reach 5% (closest (4,8) is +21.7%);
  run `python energy_matched.py` to print the full calibration table.

Structure (identical shape to ParameterMatchedCNN, widths reduced):
    Conv(1->c1)  + GN + ReLU
    Conv(c1->c1) + GN + ReLU + MaxPool(4)
    Conv(c1->c2) + GN + ReLU
    Conv(c2->c2) + GN + ReLU + MaxPool(5)
    Flatten(c2*6*12)
    LayerNorm + ReLU
    Linear(-> N_z) + LayerNorm + ReLU
    Linear(N_z -> 1)

Parameters are NOT matched to the SNN (they will be far fewer — dense MACs cost
more than sparse ACs under the proxy); the count is reported for transparency.

Interface (identical to ParameterMatchedCNN):
    forward(x, return_latent=False)  # x: (B, 1, 120, 240) -> cl (B, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Reference SNN per-step energy budget (uJ/step) and Horowitz MAC cost (pJ).
SNN_ENERGY_PER_STEP_UJ = 25.5066
E_MAC_PJ = 4.6

# Chosen energy-matched configuration (fixed before training).
DEFAULT_C1 = 4
DEFAULT_C2 = 4
DEFAULT_NZ = 16


def _make_gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with adaptive group count. Matches the SNN (=8 groups) for the
    usual channel counts (divisible by 8), but falls back to the largest divisor
    <= 8 for the reduced/odd widths this energy-matched model may use (e.g. 12)."""
    for g in (8, 6, 4, 3, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


def mac_per_step(c1: int, c2: int, n_z: int, H: int = 120, W: int = 240) -> int:
    """Dense MAC operations for one forward pass (one frame) of this CNN.

    Spatial sizes follow the fixed pooling: input H x W -> /4 -> /5. Same-padding
    stride-1 convs preserve spatial size within a block.
    """
    h1, w1 = H, W                    # block 1 resolution
    h2, w2 = H // 4, W // 4          # block 2 resolution (after MaxPool(4))
    hf, wf = h2 // 5, w2 // 5        # flattened resolution (after MaxPool(5))
    flat = c2 * hf * wf
    k = 9                            # 3x3 kernel
    mac = 0
    mac += h1 * w1 * c1 * 1 * k      # b1_conv1 (in=1)
    mac += h1 * w1 * c1 * c1 * k     # b1_conv2
    mac += h2 * w2 * c2 * c1 * k     # b2_conv1
    mac += h2 * w2 * c2 * c2 * k     # b2_conv2
    mac += flat * n_z                # z_proj
    mac += n_z * 1                   # cl_readout
    return int(mac)


def energy_per_step_uj(c1: int, c2: int, n_z: int) -> float:
    """Estimated inference energy per frame (uJ) = MAC * eps_MAC."""
    return mac_per_step(c1, c2, n_z) * E_MAC_PJ * 1e-6   # pJ -> uJ


class EnergyMatchedCNN(nn.Module):
    """Reduced-width dense CNN matched to the SNN's per-step energy budget.

    Module names match the reference SNN / parameter-matched CNN so the
    efficiency-metrics op table lines up layer-for-layer.
    """

    TRAINING_DEFAULTS = {"LR": 1e-4, "WEIGHT_DECAY": 1e-4, "GRAD_CLIP": 1.0,
                         "EPOCHS": 100}

    def __init__(self, c1: int = DEFAULT_C1, c2: int = DEFAULT_C2,
                 n_z: int = DEFAULT_NZ, in_channels: int = 1):
        super().__init__()

        self.c1 = c1
        self.c2 = c2
        self.n_z = n_z
        # flat_dim follows the fixed pooling: 120x240 -> /4 -> 30x60 -> /5 -> 6x12
        self.flat_dim = c2 * 6 * 12

        # ---- Conv Block 1: C_in -> c1, MaxPool(4x4) ----
        self.b1_conv1 = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.b1_gn1 = _make_gn(c1)
        self.b1_conv2 = nn.Conv2d(c1, c1, 3, padding=1)
        self.b1_gn2 = _make_gn(c1)
        self.pool1 = nn.MaxPool2d(4)

        # ---- Conv Block 2: c1 -> c2, MaxPool(5x5) ----
        self.b2_conv1 = nn.Conv2d(c1, c2, 3, padding=1)
        self.b2_gn1 = _make_gn(c2)
        self.b2_conv2 = nn.Conv2d(c2, c2, 3, padding=1)
        self.b2_gn2 = _make_gn(c2)
        self.pool2 = nn.MaxPool2d(5)

        # ---- Flatten + latent projection: flat_dim -> N_z ----
        self.flat_ln = nn.LayerNorm(self.flat_dim)
        self.z_proj = nn.Linear(self.flat_dim, n_z)
        self.z_ln = nn.LayerNorm(n_z)

        # ---- Cl readout: N_z -> 1 ----
        self.cl_readout = nn.Linear(n_z, 1)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        h = F.relu(self.b1_gn1(self.b1_conv1(x)))
        h = F.relu(self.b1_gn2(self.b1_conv2(h)))
        h = self.pool1(h)

        h = F.relu(self.b2_gn1(self.b2_conv1(h)))
        h = F.relu(self.b2_gn2(self.b2_conv2(h)))
        h = self.pool2(h)

        h = h.flatten(1)
        h = F.relu(self.flat_ln(h))
        z = F.relu(self.z_ln(self.z_proj(h)))
        cl = self.cl_readout(z)

        if return_latent:
            return z, cl
        return cl


# ==========================================
# CALIBRATION  (run:  python energy_matched.py)
# ==========================================
def _print_calibration(n_z: int = DEFAULT_NZ):
    """Tabulate MAC/step, energy/step and %-diff vs the SNN budget for a grid of
    (c1, c2) candidates, so the chosen configuration is reproducible."""
    # user-listed 2x-expansion examples + the asymmetric candidates near budget
    candidates = [(4, 4), (4, 3), (3, 6), (4, 5), (4, 8),
                  (6, 12), (8, 16), (12, 24), (16, 32)]
    target = SNN_ENERGY_PER_STEP_UJ
    print(f"SNN budget: {target:.4f} uJ/step  (eps_MAC = {E_MAC_PJ} pJ)")
    print(f"{'(c1,c2)':>10s} {'MAC/step':>14s} {'uJ/step':>10s} {'diff':>9s}"
          f"   {'params':>8s}")
    print("-" * 58)
    for c1, c2 in candidates:
        mac = mac_per_step(c1, c2, n_z)
        e = mac * E_MAC_PJ * 1e-6
        diff = 100.0 * (e - target) / target
        params = sum(p.numel() for p in
                     EnergyMatchedCNN(c1=c1, c2=c2, n_z=n_z).parameters())
        star = "  <= chosen" if (c1, c2) == (DEFAULT_C1, DEFAULT_C2) else ""
        print(f"{f'({c1},{c2})':>10s} {mac:>14,d} {e:>10.4f} {diff:>+8.1f}%"
              f"   {params:>8,d}{star}")


if __name__ == "__main__":
    _print_calibration()
    m = EnergyMatchedCNN()
    n = sum(p.numel() for p in m.parameters())
    print(f"\nChosen: c1={m.c1} c2={m.c2} N_z={m.n_z}  ->  params: {n:,}")
    x = torch.randn(8, 1, 120, 240)
    z, cl = m(x, return_latent=True)
    print(f"forward: {tuple(x.shape)} -> z {tuple(z.shape)}, cl {tuple(cl.shape)}")
