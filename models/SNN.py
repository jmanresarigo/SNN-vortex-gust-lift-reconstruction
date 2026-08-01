"""
SNN.py — Shallow spiking Cl encoder.

A spiking neural network that reads a spike-encoded vorticity field and predicts
the lift coefficient Cl directly (decoder-free). Two convolutional blocks of LIF
spiking neurons extract the vortical structures, a leaky-integrate flatten layer
carries the temporal state, and a direct latent projection plus a linear readout
produce the Cl prediction. The convolutional feature width is expanded in Block 2
(16 -> 32 channels) to enrich the representation before the strong latent
bottleneck (Linear(flat_dim -> N_z)).

Spatial pathway (H=120, W=240):
    Input  (B, C_in, 120, 240)
      | Block 1:  Conv(16)x2 + GN + LIF + MaxPool(4x4)  [spikes]
    (B, 16, 30, 60)
      | Block 2:  Conv(32)x2 + GN + LIF + MaxPool(5x5)  [spikes]
    (B, 32, 6, 12)
      | Flatten (B, 2304)
      | LN(2304) + LIF -> membrane(2304)  [continuous, carries temporal state]
      | Linear(2304 -> N_z) + LayerNorm -> z  (continuous latent)
      | Linear(N_z -> 1) -> Cl  (continuous prediction)

Interface:
    forward(x_seq, bptt_chunk=None, state=None, return_state=False)
        -> z_seq, cl_pred
        -> z_seq, cl_pred, mems   (when return_state=True)
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# ==========================================
# HELPERS
# ==========================================
def _make_lif(beta_init: float, fc_threshold: float = 0.5) -> snn.Leaky:
    """Create a LIF neuron with learnable threshold and decay rate."""
    return snn.Leaky(
        beta=beta_init,
        spike_grad=surrogate.atan(alpha=2.0),
        threshold=fc_threshold,
        learn_threshold=True,
        learn_beta=True,
        reset_mechanism="subtract",
        init_hidden=False,
    )


def _make_gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with adaptive group count."""
    groups = min(8, channels)
    return nn.GroupNorm(groups, channels)


# ==========================================
# SNN MODEL
# ==========================================
class SNN_Model(nn.Module):
    """
    Shallow spiking Cl encoder.

    Reads a spike-encoded vorticity sequence and predicts the lift coefficient
    Cl directly (decoder-free). Two convolutional spiking blocks extract vortical
    features (with a channel expansion 16 -> 32 in Block 2), a leaky-integrate
    flatten layer carries the temporal state, and a direct latent projection plus
    a linear readout produce the Cl prediction.

    Spatial pathway (H=120, W=240):
        Input  (B, C_in, 120, 240)
          | Block 1:  Conv(16)x2 + GN + LIF + MaxPool(4x4)  [spikes]
        (B, 16, 30, 60)
          | Block 2:  Conv(32)x2 + GN + LIF + MaxPool(5x5)  [spikes]
        (B, 32, 6, 12)
          | Flatten (B, 2304)
          | LN(2304) + LIF -> membrane(2304)  [continuous, carries temporal state]
          | Linear(2304 -> N_z) + LayerNorm -> z  (continuous)
          | Linear(N_z -> 1) -> Cl  (continuous)

    Interface:
        forward(x_seq, bptt_chunk=None, state=None, return_state=False)
            -> z_seq, cl_pred  (+ mems when return_state=True)
    """

    TRAINING_DEFAULTS = {
        "LR": 3e-4,
        "WEIGHT_DECAY": 1e-4,
        "GRAD_CLIP": 1.0,
        "BETA_INIT": 0.9,
        "W_LIFT": 1.0,
        "EPOCHS": 120,
    }

    def __init__(
        self,
        n_z: int = 16,
        in_channels: int = 2,
        beta_init: float = 0.9,
        fc_threshold: float = 0.5,
        gradient_checkpointing: bool = True,
    ):
        """
        Args:
            n_z:          latent dimension
            in_channels:  spike input channels (default 2: omega+, omega-)
            beta_init:    initial LIF membrane decay rate
            fc_threshold: LIF threshold used for all LIF layers
            gradient_checkpointing: trade compute for memory
        """
        super().__init__()

        self.n_z = n_z
        self.gradient_checkpointing = gradient_checkpointing

        # ---- Channel / feature-dim constants ----
        # Block 1 width, Block 2 width, and the flattened feature dimension fed
        # to the latent projection.
        #
        # Flat dim derivation (standard input 120 x 240):
        #   Input spatial size: 120 x 240
        #   After pool1=4:       30 x 60
        #   After pool2=5:        6 x 12
        #   Final channels:      32
        #   flat_dim = 32 * 6 * 12 = 2304
        self.c1 = 16
        self.c2 = 32
        self.flat_dim = 2304

        # ---- Conv Block 1: C_in -> c1 (16), MaxPool(4x4) ----
        self.b1_conv1 = nn.Conv2d(in_channels, self.c1, 3, padding=1)
        self.b1_gn1 = _make_gn(self.c1)
        self.b1_lif1 = _make_lif(beta_init, fc_threshold)

        self.b1_conv2 = nn.Conv2d(self.c1, self.c1, 3, padding=1)
        self.b1_gn2 = _make_gn(self.c1)
        self.b1_lif2 = _make_lif(beta_init, fc_threshold)

        self.pool1 = nn.MaxPool2d(4)

        # ---- Conv Block 2: c1 (16) -> c2 (32), MaxPool(5x5) ----
        # Block 2 widens the channels (16 -> 32) to enrich the convolutional
        # feature representation before the latent projection.
        self.b2_conv1 = nn.Conv2d(self.c1, self.c2, 3, padding=1)
        self.b2_gn1 = _make_gn(self.c2)
        self.b2_lif1 = _make_lif(beta_init, fc_threshold)

        self.b2_conv2 = nn.Conv2d(self.c2, self.c2, 3, padding=1)
        self.b2_gn2 = _make_gn(self.c2)
        self.b2_lif2 = _make_lif(beta_init, fc_threshold)

        self.pool2 = nn.MaxPool2d(5)

        # ---- Flat LIF + direct latent projection: flat_dim -> N_z ----
        # A LIF neuron integrates the flat_dim spike vector over time. The
        # membrane potential (continuous, graded) is then projected to z, giving
        # a richer signal than binary spikes. LayerNorm before LIF stabilises the
        # input scale for healthy spiking.
        self.flat_ln = nn.LayerNorm(self.flat_dim)
        self.flat_lif = _make_lif(beta_init, fc_threshold)
        self.z_proj = nn.Linear(self.flat_dim, n_z)
        self.z_ln = nn.LayerNorm(n_z)

        # ---- Cl readout: N_z -> 1 ----
        # Direct linear readout from latent space, no hidden layer.
        self.cl_readout = nn.Linear(n_z, 1)

    def _init_membranes(
        self,
        batch_size: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> dict:
        """Initialise all LIF membrane tensors to zero."""
        # Block 2 operates after pool1 (factor 4), so its spatial size is H//4, W//4.
        H4, W4 = H // 4, W // 4

        return {
            "b1_1": torch.zeros(batch_size, self.c1, H, W, device=device),
            "b1_2": torch.zeros(batch_size, self.c1, H, W, device=device),
            "b2_1": torch.zeros(batch_size, self.c2, H4, W4, device=device),
            "b2_2": torch.zeros(batch_size, self.c2, H4, W4, device=device),
            "flat": torch.zeros(batch_size, self.flat_dim, device=device),
        }

    def init_state(self, batch_size: int, device,
                   H: int = 120, W: int = 240) -> dict:
        """Return zeroed membrane state dict for stateful TBPTT."""
        return self._init_membranes(batch_size, H, W, device)

    @staticmethod
    def detach_state(state: dict) -> dict:
        """Detach all membrane tensors to cut the computation graph."""
        return {k: v.detach() for k, v in state.items()}

    def _forward_step(self, x: torch.Tensor, mems: dict):
        """
        Process one timestep through the full pipeline.

        Args:
            x:    (B, C_in, H, W) spike input for this timestep
            mems: membrane dict from _init_membranes() or previous step

        Returns:
            z:    (B, N_z) continuous latent vector
            cl:   (B, 1)   continuous Cl prediction
            mems: updated membrane dict
        """
        # --- Conv Block 1 --- (spikes between layers)
        h = self.b1_gn1(self.b1_conv1(x))
        spk, mems["b1_1"] = self.b1_lif1(h, mems["b1_1"])

        h = self.b1_gn2(self.b1_conv2(spk))
        spk, mems["b1_2"] = self.b1_lif2(h, mems["b1_2"])

        h = self.pool1(spk)

        # --- Conv Block 2 --- (spikes between layers)
        h = self.b2_gn1(self.b2_conv1(h))
        spk, mems["b2_1"] = self.b2_lif1(h, mems["b2_1"])

        h = self.b2_gn2(self.b2_conv2(spk))
        spk, mems["b2_2"] = self.b2_lif2(h, mems["b2_2"])

        h = self.pool2(spk)

        # --- Flatten + membrane-potential projection -> z ---
        # LN stabilises the sparse binary spike vector before LIF integration.
        # The LIF membrane (continuous) carries temporal state; reading it rather
        # than the binary spike output gives a graded, information-rich latent.
        spk_flat = h.flatten(1)                              # (B, 2304) binary spikes
        _, mems["flat"] = self.flat_lif(self.flat_ln(spk_flat), mems["flat"])

        z = self.z_ln(self.z_proj(mems["flat"]))             # (B, N_z) continuous

        # --- Direct Cl readout ---
        cl = self.cl_readout(z)                       # (B, 1)

        return z, cl, mems

    def forward(self, x_seq: torch.Tensor, bptt_chunk: int = None,
                state: dict = None, return_state: bool = False):
        """
        Run the full spiking encoder over a spike sequence.

        Args:
            x_seq:        (T, B, C_in, H, W) spike-encoded vorticity
            bptt_chunk:   detach membrane states every N steps.
                          None = full BPTT through all T steps.
            state:        membrane dict from a previous forward call, or None
                          to reset (standard independent-window training).
            return_state: if True, return the final membrane dict as a third
                          return value (required for stateful TBPTT).

        Returns:
            z_seq:   (T, B, N_z)   continuous latent trajectory
            cl_pred: (T, B, 1)     Cl prediction per timestep
            state:   membrane dict (only when return_state=True)
        """
        T, B, _, H, W = x_seq.shape

        mems = state if state is not None else self._init_membranes(B, H, W, x_seq.device)

        z_list = []
        cl_list = []

        for t in range(T):
            if bptt_chunk is not None and t > 0 and t % bptt_chunk == 0:
                mems = {k: v.detach() for k, v in mems.items()}

            z_t, cl_t, mems = self._forward_step(x_seq[t], mems)

            z_list.append(z_t)
            cl_list.append(cl_t)

        z_seq = torch.stack(z_list)       # (T, B, N_z)
        cl_pred = torch.stack(cl_list)    # (T, B, 1)

        if return_state:
            return z_seq, cl_pred, mems
        return z_seq, cl_pred
