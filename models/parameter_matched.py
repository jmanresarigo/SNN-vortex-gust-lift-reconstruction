"""
parameter_matched.py — Parameter-matched non-spiking CNN baseline.

A conventional (dense) convolutional network that reconstructs the instantaneous
lift coefficient Cl(t) directly from each continuous vorticity field. It is the
"equal model capacity" comparison point for the spiking encoder (SNN.py,
SNN_Model): the spatial architecture is preserved as closely as possible, but
every spiking element is removed.

Differences vs the SNN (SNN.py):
  * Input is ONE signed continuous channel omega(x,y,t) — no rate encoding,
    polarity split, stochastic sampling, clipping, or normalization.
  * Each LIF neuron is replaced by a ReLU. No membrane state, recurrence, or TBPTT
    — every vorticity snapshot is processed independently.
  * The learnable LIF threshold/beta parameters are gone.

Parameter matching (target: within ~1% of the SNN's 58,251 params):
  Dropping the 2nd input channel (-144 conv weights) and the 10 LIF parameters
  would leave the CNN ~8% below the SNN. To stay within tolerance while keeping
  N_z=16 and all convolutional widths, the reference model's pre-projection
  LayerNorm(2304) (its `flat_ln`) is retained. This yields 58,097 parameters
  (-0.26% vs the SNN) — the CNN only "adds back" a normalization the SNN already
  has, rather than altering the architecture to force an exact numeric match.

Spatial pathway (H=120, W=240), input (B, 1, 120, 240):
    Conv(1->16)  + GN + ReLU
    Conv(16->16) + GN + ReLU + MaxPool(4)      -> (B, 16, 30, 60)
    Conv(16->32) + GN + ReLU
    Conv(32->32) + GN + ReLU + MaxPool(5)       -> (B, 32, 6, 12)
    Flatten(2304)
    LayerNorm(2304) + ReLU
    Linear(2304 -> 16) + LayerNorm(16) + ReLU
    Linear(16 -> 1)                              -> Cl (B, 1)

Interface:
    forward(x, return_latent=False)
        x: (B, 1, H, W) continuous vorticity field for one time step
        -> cl (B, 1)                (default)
        -> (z, cl) with z (B, N_z)  when return_latent=True
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# HELPERS
# ==========================================
def _make_gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with adaptive group count (identical to SNN._make_gn)."""
    groups = min(8, channels)
    return nn.GroupNorm(groups, channels)


# ==========================================
# PARAMETER-MATCHED CNN
# ==========================================
class ParameterMatchedCNN(nn.Module):
    """Non-spiking CNN mirroring SNN_Model's spatial architecture.

    Module names match the reference SNN (b1_conv1, b1_gn1, ... z_proj, z_ln,
    cl_readout) so the efficiency-metrics op table lines up layer-for-layer.
    """

    TRAINING_DEFAULTS = {
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "GRAD_CLIP": 1.0,
        "EPOCHS": 100,
    }

    def __init__(self, n_z: int = 16, in_channels: int = 1):
        """
        Args:
            n_z:         latent dimension (matches the reference SNN, 16)
            in_channels: input channels (1: the signed continuous omega field)
        """
        super().__init__()

        self.n_z = n_z

        # Channel / feature-dim constants (identical to the SNN).
        #   Input 120 x 240 -> pool1=4 -> 30 x 60 -> pool2=5 -> 6 x 12
        #   flat_dim = 32 * 6 * 12 = 2304
        self.c1 = 16
        self.c2 = 32
        self.flat_dim = 2304

        # ---- Conv Block 1: C_in -> c1 (16), MaxPool(4x4) ----
        self.b1_conv1 = nn.Conv2d(in_channels, self.c1, 3, padding=1)
        self.b1_gn1 = _make_gn(self.c1)
        self.b1_conv2 = nn.Conv2d(self.c1, self.c1, 3, padding=1)
        self.b1_gn2 = _make_gn(self.c1)
        self.pool1 = nn.MaxPool2d(4)

        # ---- Conv Block 2: c1 (16) -> c2 (32), MaxPool(5x5) ----
        self.b2_conv1 = nn.Conv2d(self.c1, self.c2, 3, padding=1)
        self.b2_gn1 = _make_gn(self.c2)
        self.b2_conv2 = nn.Conv2d(self.c2, self.c2, 3, padding=1)
        self.b2_gn2 = _make_gn(self.c2)
        self.pool2 = nn.MaxPool2d(5)

        # ---- Flatten + latent projection: flat_dim -> N_z ----
        # LayerNorm(2304) mirrors the reference model's `flat_ln`; the ReLU takes
        # the place of that model's flat LIF nonlinearity.
        self.flat_ln = nn.LayerNorm(self.flat_dim)
        self.z_proj = nn.Linear(self.flat_dim, n_z)
        self.z_ln = nn.LayerNorm(n_z)

        # ---- Cl readout: N_z -> 1 ----
        self.cl_readout = nn.Linear(n_z, 1)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        """
        Args:
            x: (B, in_channels, H, W) continuous vorticity field at one time step.

        Returns:
            cl (B, 1), or (z, cl) with z (B, N_z) when return_latent=True.
        """
        # --- Conv Block 1 ---
        h = F.relu(self.b1_gn1(self.b1_conv1(x)))
        h = F.relu(self.b1_gn2(self.b1_conv2(h)))
        h = self.pool1(h)

        # --- Conv Block 2 ---
        h = F.relu(self.b2_gn1(self.b2_conv1(h)))
        h = F.relu(self.b2_gn2(self.b2_conv2(h)))
        h = self.pool2(h)

        # --- Flatten + latent projection ---
        h = h.flatten(1)                        # (B, 2304)
        h = F.relu(self.flat_ln(h))
        z = F.relu(self.z_ln(self.z_proj(h)))   # (B, N_z)

        # --- Direct Cl readout ---
        cl = self.cl_readout(z)                 # (B, 1)

        if return_latent:
            return z, cl
        return cl
