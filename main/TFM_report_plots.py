"""
TFM_report_plots.py — Figures for the TFM report.

Generates illustrative figures used in the theoretical background of the
thesis. Currently produces:

    1. The Dirac delta function δ(t), drawn as an idealised unit impulse.
    2. An example spike train s_i(t) = Σ_k δ(t - t_i^k), shown both as a
       continuous-time train of impulses and as its discrete-time binary
       counterpart S_i[t] ∈ {0,1}.
    3. The Leaky Integrate-and-Fire (LIF) neuron dynamics, simulating the
       discrete update U_i[t+1] = β U_i[t] + W X[t] − S_i[t+1] U_th and
       annotating the Integrate, Leak, Fire and (subtractive) Reset mechanisms.
    4. Three spike-encoding strategies, each simulated from its defining rule:
       rate encoding (Bernoulli, P(S=1)=x̂), delta encoding (spike if
       |Δx̂| ≥ θ_Δ) and latency encoding (spike time t_spike = f(x̂)).
    5. The non-differentiability of the spike function: the Heaviside spike and
       its ill-defined derivative (zero almost everywhere, Dirac at threshold).
    6. The arctangent surrogate gradient that replaces the spike derivative in
       the backward pass, giving a nonzero learning signal near threshold.
    7. The recurrent nature of the LIF neuron: the folded (recurrent) form and
       its unrolling through time into a chain of membrane states.
    8. Backpropagation Through Time and the vanishing / exploding gradient
       problem: gradients chain backward as a product of temporal Jacobians,
       so their magnitude scales as |J|^(t−τ).

Chapter 5 (Interpolation & spike encoding) figure:

   11. Preprocessing / spike-encoding pipeline overview: raw vorticity ->
       temporal interpolation -> normalize + clip -> rate & delta spike
       channels, with the lift signal kept time-aligned. Built from one real
       snapshot (skipped automatically if the data is absent).

Chapter 4 (Dataset) figures, which read the raw simulation data from
Datasets/Dataset_original/ (skipped automatically if the data is absent):

    9. Representative vortex–gust cases: for each of a few (alpha, G, R, y)
       cases, three vorticity snapshots (pre-impingement, impingement,
       post-impingement) with the airfoil overlaid, above the lift history
       C_l(t) with those three instants marked.
   10. Schematic of the physical setup: airfoil at incidence, incoming Taylor
       vortex at x_0/c = -2 with (G, R, y) annotated, coordinate axes and the
       data subdomain outlined.

Output (PNG only):
    report_figures/
        dirac_delta.png
        spike_train.png
        lif_neuron.png
        rate_encoding.png
        delta_encoding.png
        latency_encoding.png
        non_differentiability.png
        surrogate_gradient.png
        recurrence_unrolling.png
        bptt_gradient_flow.png
        setup_schematic.png
        encoding_pipeline.png
        normalization_fields.png
        normalization_distribution.png
        rate_gamma_curve.png
        rate_gain_curve.png
        rate_encoding_fields.png
        delta_mechanism.png
        delta_threshold_effect.png
        random_case_split.png
        dataset_scatter.png
        network_architecture.png
        parameter_matched_architecture.png
        energy_matched_architecture.png
        case_AoA20_RD4.png
        case_AoA40_RD29.png
        case_AoA60_RD9.png
        case_AoA20_RD29_reconstructed.png

Usage:
    python TFM_report_plots.py
"""

import os

import numpy as np
import pandas as pd
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Circle,
                                Polygon, Rectangle, Arc)

# --------------------------------------------------------------------------- #
# Style
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

OUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "report_figures")
os.makedirs(OUT_DIR, exist_ok=True)

SPIKE_BLUE = "#1f5fa6"

# Colour code for the four LIF mechanisms (kept consistent between the
# annotations and the spike-train panels).
C_INTEGRATE = "#2e8b57"   # green  — integration of input spikes
C_LEAK = "#e08214"        # orange — membrane leakage
C_FIRE = SPIKE_BLUE       # blue   — spike emission
C_RESET = "#c0392b"       # red    — subtractive reset
C_THRESH = "#7f7f7f"      # grey   — threshold line


def _save(fig, name, bbox_extra_artists=None):
    """Save a figure as a PNG.

    bbox_extra_artists forces those artists into the tight-bbox crop; pass the
    z-axis label for 3D axes, which the tight bbox would otherwise clip.
    """
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, bbox_extra_artists=bbox_extra_artists)
    print(f"saved {name}.png")


# --------------------------------------------------------------------------- #
# Figure 1 — Dirac delta function
# --------------------------------------------------------------------------- #
def plot_dirac_delta():
    """Idealised unit impulse δ(t - t0): an infinitely tall, zero-width spike."""
    fig, ax = plt.subplots(figsize=(6.0, 3.4))

    t0 = 0.0
    ax.axhline(0.0, color="black", linewidth=0.8)

    # The impulse: an upward arrow of "unit area" located at t0.
    ax.annotate(
        "", xy=(t0, 1.0), xytext=(t0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=SPIKE_BLUE, lw=2.2,
                        mutation_scale=20),
    )
    ax.text(t0 + 0.08, 1.0, r"$(1)$", color=SPIKE_BLUE, va="center", fontsize=12)

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\delta(t - t_0)$", rotation=0, ha="right", va="center",
                  labelpad=12)
    ax.set_title(r"Dirac delta function")

    # Clean up: hide y ticks (the impulse is conceptually infinite-amplitude),
    # keep only a labelled t-axis.
    ax.set_yticks([])
    ax.set_xticks([t0])
    ax.set_xticklabels([r"$t_0$"])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    _save(fig, "dirac_delta")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 — Example spike train
# --------------------------------------------------------------------------- #
def plot_spike_train():
    """
    Continuous-time spike train s_i(t) = Σ_k δ(t - t_i^k) (top) alongside its
    discrete-time binary representation S_i[t] ∈ {0,1} (bottom).

    Spikes are placed at integer time positions so the continuous and discrete
    representations have an exact 1-to-1 correspondence (no binning ambiguity).
    """
    # Integer spike positions — one unique time step per spike.
    spike_steps = np.array([1, 2, 3, 5, 7, 8, 9, 11])  # 8 spikes
    n_steps = 13  # display range 0 … 12
    annotated_spike_idx = 3  # label the 4th spike (0-indexed) as t_i^4

    # Discrete binary array: 1 at each spike step, 0 elsewhere.
    binary = np.zeros(n_steps, dtype=int)
    binary[spike_steps] = 1
    step_idx = np.arange(n_steps)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 5.2), sharex=False,
        gridspec_kw={"hspace": 0.85},
    )

    # --- Top: continuous-time train of impulses --------------------------- #
    ax_top.axhline(0.0, color="black", linewidth=0.8)
    for tk in spike_steps:
        ax_top.annotate(
            "", xy=(tk, 1.0), xytext=(tk, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=SPIKE_BLUE, lw=2.0,
                            mutation_scale=16),
        )

    # Label the 4th spike as t_i^4, placed to its left at mid-height so it
    # sits beside the arrow rather than cluttering the axis.
    t4 = spike_steps[annotated_spike_idx]
    ax_top.text(t4 - 0.25, 0.55, r"$t_i^{4}$", ha="right", va="center",
                fontsize=12, color="black")

    ax_top.set_xlim(0, n_steps - 1)
    ax_top.set_ylim(-0.15, 1.35)
    ax_top.set_xlabel(r"$t$")
    ax_top.set_ylabel(r"$s_i(t)$", rotation=0, ha="right", va="center",
                      labelpad=12)
    ax_top.set_title(r"Continuous spike train $\;s_i(t)=\sum_k \delta(t - t_i^{k})$")
    ax_top.set_yticks([])
    ax_top.set_xticks(np.arange(0, n_steps))
    ax_top.set_xticklabels([str(i) for i in range(n_steps)])
    ax_top.tick_params(axis="x", length=3, pad=4)
    for spine in ("top", "right", "left"):
        ax_top.spines[spine].set_visible(False)

    # --- Bottom: discrete-time binary representation ---------------------- #
    ax_bot.bar(step_idx, binary, width=0.7, align="center",
               color=SPIKE_BLUE, edgecolor="white", linewidth=0.6)
    ax_bot.set_xlim(-0.5, n_steps - 0.5)
    ax_bot.set_ylim(0, 1.15)
    ax_bot.set_xlabel(r"discrete time step $t$")
    ax_bot.set_ylabel(r"$S_i[t]$", rotation=0, ha="right", va="center",
                      labelpad=12)
    ax_bot.set_title(r"Binary spike train representation $\;S_i[t]\in\{0,1\}$")
    ax_bot.set_yticks([0, 1])
    ax_bot.set_xticks(step_idx)
    ax_bot.set_xticklabels([str(i) for i in step_idx])
    for spine in ("top", "right"):
        ax_bot.spines[spine].set_visible(False)

    _save(fig, "spike_train")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 — Leaky Integrate-and-Fire neuron dynamics
# --------------------------------------------------------------------------- #
def plot_lif_neuron():
    """
    Time-domain illustration of the discrete LIF neuron, simulating

        U_i[t+1] = β U_i[t] + W X[t] − S_i[t+1] U_th ,
        S_i[t]   = 1  if  U_i[t] ≥ U_th ,  else 0 ,

    so that the four mechanisms are shown in action within a single run:
        Integrate  (+ W X[t]),  Leak (β U_i[t]),  Fire (U ≥ U_th)  and
        subtractive Reset (− U_th, residual retained).

    Layout (shared time axis):
        Top    — input spike train  X[t]
        Middle — membrane potential  U[t]  with threshold U_th
        Bottom — output spike train  S[t]
    """
    # ---- Simulation parameters -------------------------------------------- #
    beta = 0.9     # membrane decay (leak)
    W = 0.5        # synaptic weight applied to each input spike
    U_th = 1.0     # firing threshold
    T = 42         # number of discrete time steps

    # Input spikes: a sub-threshold bump (integrate + a long, clearly visible
    # leak), then a cluster that overshoots the threshold so the subtractive
    # reset leaves a clearly positive retained residual (fire + reset).
    in_spikes = [2, 4, 20, 23, 24]
    X = np.zeros(T)
    X[in_spikes] = 1.0

    # ---- Exact discrete recurrence ---------------------------------------- #
    U = np.zeros(T)
    S = np.zeros(T)
    for t in range(T - 1):
        if U[t] >= U_th:
            S[t] = 1.0
        U[t + 1] = beta * U[t] + W * X[t + 1] - S[t] * U_th
    if U[T - 1] >= U_th:
        S[T - 1] = 1.0

    t_axis = np.arange(T)
    in_idx = np.where(X == 1.0)[0]
    fire_idx = np.where(S == 1.0)[0]

    fig, (ax_in, ax_mem, ax_out) = plt.subplots(
        3, 1, figsize=(8.6, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 3.2, 1], "hspace": 0.18},
    )

    # ---- Top: input spike train ------------------------------------------ #
    ax_in.vlines(in_idx, 0, 1, color=C_INTEGRATE, lw=2.6)
    ax_in.plot(in_idx, np.ones_like(in_idx), "o", color=C_INTEGRATE, ms=5)
    ax_in.set_ylim(0, 1.3)
    ax_in.set_yticks([])
    ax_in.set_ylabel(r"$x_i[t]$", rotation=0, ha="right", va="center",
                     labelpad=16)
    ax_in.set_title(r"Input spikes  $x_i[t]$", color=C_INTEGRATE, fontsize=12)
    for spine in ("top", "right", "left"):
        ax_in.spines[spine].set_visible(False)

    # ---- Middle: membrane potential -------------------------------------- #
    ax_mem.axhline(U_th, color=C_THRESH, ls="--", lw=1.3)
    ax_mem.text(T - 0.5, U_th + 0.04, r"$U_{\mathrm{th}}$", color=C_THRESH,
                ha="right", va="bottom", fontsize=12)
    ax_mem.plot(t_axis, U, color="#222222", lw=1.8, marker="o", ms=3.2, zorder=3)
    # Highlight the membrane value at the firing step(s).
    ax_mem.plot(fire_idx, U[fire_idx], "o", color=C_FIRE, ms=8,
                zorder=4, label="fire")

    ax_mem.set_ylim(-0.15, 2.05)
    ax_mem.set_ylabel(r"$U_i[t]$", rotation=0, ha="right", va="center", labelpad=16)
    ax_mem.set_yticks([0, U_th])
    ax_mem.set_yticklabels(["0", r"$U_{\mathrm{th}}$"])
    for spine in ("top", "right"):
        ax_mem.spines[spine].set_visible(False)

    arrow = dict(arrowstyle="->", lw=1.4)

    # Integrate: an upward jump at an input spike.
    ax_mem.annotate(
        r"Integrate $+\,w_i^t\,x_i[t]$",
        xy=(4, U[4]), xytext=(5.2, 1.72),
        color=C_INTEGRATE, fontsize=11,
        arrowprops=dict(color=C_INTEGRATE, **arrow),
    )
    # Leak: exponential decay between inputs.
    leak_idx = 11
    ax_mem.annotate(
        r"Leak $\beta\,U_i[t]$",
        xy=(leak_idx, U[leak_idx]), xytext=(leak_idx + 1.0, 1.15),
        color=C_LEAK, fontsize=11,
        arrowprops=dict(color=C_LEAK, **arrow),
    )
    # Fire: threshold crossing.
    f0 = fire_idx[0]
    ax_mem.annotate(
        r"Fire $U_i[t] \geq U_{\mathrm{th}}$",
        xy=(f0, U[f0]), xytext=(f0 - 9.0, 1.78),
        color=C_FIRE, fontsize=11,
        arrowprops=dict(color=C_FIRE, **arrow),
    )
    # Reset: subtractive drop, residual retained.
    ax_mem.annotate(
        r"Reset $-\,U_{\mathrm{th}}$" + "\n(residual retained)",
        xy=(f0 + 1, U[f0 + 1]), xytext=(f0 + 1.6, 1.30),
        color=C_RESET, fontsize=11, va="center",
        arrowprops=dict(color=C_RESET, **arrow),
    )

    # ---- Bottom: output spike train -------------------------------------- #
    ax_out.vlines(fire_idx, 0, 1, color=C_FIRE, lw=2.6)
    ax_out.plot(fire_idx, np.ones_like(fire_idx), "o", color=C_FIRE, ms=5)
    ax_out.set_ylim(0, 1.3)
    ax_out.set_yticks([])
    ax_out.set_ylabel(r"$s_i[t]$", rotation=0, ha="right", va="center", labelpad=16)
    ax_out.set_title(r"Output spikes  $s_i[t]$", color=C_FIRE, fontsize=12)
    ax_out.set_xlabel(r"time step $t$")
    ax_out.set_xlim(0, T - 1)
    for spine in ("top", "right", "left"):
        ax_out.spines[spine].set_visible(False)

    _save(fig, "lif_neuron")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4a — Rate encoding
# --------------------------------------------------------------------------- #
def plot_rate_encoding():
    """
    Rate encoding: each time step draws a Bernoulli spike with probability
    P(S[t]=1) = x. Three constant input levels are shown so that a larger value
    visibly produces a denser spike train and a higher empirical rate
    r = (1/T) Σ S[t].
    """
    rng = np.random.default_rng(7)
    T = 40
    values = [0.15, 0.50, 0.85]

    fig, axes = plt.subplots(
        len(values), 1, figsize=(8.2, 4.2), sharex=True,
        gridspec_kw={"hspace": 0.55},
    )

    for ax, x in zip(axes, values):
        spikes = (rng.random(T) < x).astype(int)
        idx = np.where(spikes == 1)[0]
        ax.vlines(idx, 0, 1, color=SPIKE_BLUE, lw=2.0)
        r = spikes.sum() / T

        ax.set_ylim(0, 1.55)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_yticks([])
        ax.set_ylabel(rf"$\hat{{x}} = {x:.2f}$", rotation=0, ha="right",
                      va="center", labelpad=14)
        ax.text(T - 0.5, 1.45, rf"empirical rate $r = {r:.2f}$",
                ha="right", va="top", fontsize=10.5, color="#555555")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    axes[-1].set_xlabel(r"time step $t$")
    axes[0].set_title(r"Rate Encoding", fontsize=13)

    _save(fig, "rate_encoding")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4b — Delta encoding
# --------------------------------------------------------------------------- #
def plot_delta_encoding():
    """
    Delta encoding: a spike fires when the magnitude of the temporal change
    |Δx[t]| = |x[t] − x[t-1]| exceeds a threshold θ_Δ. The signal is built with
    flat regions (no spikes — sparsity), steep transitions (spikes) and one
    gentle slope below threshold (no spikes — the threshold matters).
    """
    # Piecewise input signal x[t].
    x = np.concatenate([
        np.full(8, 0.20),                    # flat low      → no change
        np.linspace(0.20, 0.85, 6)[1:],      # steep rise    → spikes
        np.full(8, 0.85),                    # flat high     → no change
        np.linspace(0.85, 0.72, 6)[1:],      # gentle slope  → below θ
        np.linspace(0.72, 0.20, 6)[1:],      # steep fall    → spikes
        np.full(8, 0.20),                    # flat low      → no change
    ])
    T = len(x)
    t = np.arange(T)

    dx = np.zeros(T)
    dx[1:] = np.diff(x)
    theta = 0.05
    spikes = (np.abs(dx) >= theta).astype(int)
    fired = spikes == 1

    fig, (ax_sig, ax_dx, ax_spk) = plt.subplots(
        3, 1, figsize=(8.4, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 2.0, 1.0], "hspace": 0.22},
    )

    # ---- Signal x[t] ----------------------------------------------------- #
    ax_sig.plot(t, x, color="#222222", lw=1.8, marker="o", ms=3.2)
    ax_sig.set_ylim(0.05, 1.0)
    ax_sig.set_ylabel(r"$\hat{x}[t]$", rotation=0, ha="right", va="center",
                      labelpad=14)
    ax_sig.set_title(r"Delta Encoding", fontsize=13)
    ax_sig.annotate("flat\n(no spikes)", xy=(4, 0.20), xytext=(2.0, 0.45),
                    fontsize=10, color=C_LEAK, ha="center",
                    arrowprops=dict(arrowstyle="->", color=C_LEAK, lw=1.2))
    ax_sig.annotate("rapid change\n(spikes)", xy=(10, 0.55), xytext=(13.5, 0.30),
                    fontsize=10, color=C_INTEGRATE, ha="center",
                    arrowprops=dict(arrowstyle="->", color=C_INTEGRATE, lw=1.2))
    ax_sig.annotate(r"gentle slope $<\theta_\Delta$ (no spikes)",
                    xy=(23, 0.78), xytext=(29, 0.94),
                    fontsize=10, color=C_LEAK, ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=C_LEAK, lw=1.2))
    for sp in ("top", "right"):
        ax_sig.spines[sp].set_visible(False)

    # ---- Increments Δx[t] with ±θ_Δ -------------------------------------- #
    colors = np.where(fired, SPIKE_BLUE, "#c9c9c9")
    ax_dx.bar(t, dx, width=0.8, color=colors)
    ax_dx.axhline(theta, color=C_RESET, ls="--", lw=1.2)
    ax_dx.axhline(-theta, color=C_RESET, ls="--", lw=1.2)
    ax_dx.text(T - 0.5, theta + 0.01, r"$+\theta_\Delta$", color=C_RESET,
               ha="right", va="bottom", fontsize=10.5)
    ax_dx.text(T - 0.5, -theta - 0.01, r"$-\theta_\Delta$", color=C_RESET,
               ha="right", va="top", fontsize=10.5)
    ax_dx.axhline(0.0, color="black", lw=0.7)
    ax_dx.set_ylabel(r"$\Delta \hat{x}[t]$", rotation=0, ha="right", va="center",
                     labelpad=14)
    for sp in ("top", "right"):
        ax_dx.spines[sp].set_visible(False)

    # ---- Spike train S_Δ[t] ---------------------------------------------- #
    ax_spk.vlines(t[fired], 0, 1, color=SPIKE_BLUE, lw=2.2)
    ax_spk.set_ylim(0, 1.3)
    ax_spk.set_yticks([])
    ax_spk.set_ylabel(r"$S_\Delta[t]$", rotation=0, ha="right", va="center",
                      labelpad=14)
    ax_spk.set_xlabel(r"time step $t$")
    ax_spk.set_xlim(-0.5, T - 0.5)
    for sp in ("top", "right", "left"):
        ax_spk.spines[sp].set_visible(False)

    _save(fig, "delta_encoding")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4c — Latency encoding
# --------------------------------------------------------------------------- #
def plot_latency_encoding():
    """
    Latency encoding: the value of an input is carried by a single spike time,
    t_spike = f(x̂). One spike train per neuron is shown for four input values,
    using the simple monotonically decreasing map t_spike = (1 − x̂) T, so a
    larger value fires earlier and a smaller value fires later.
    """
    T = 40
    # Ordered high → low so the spike marches left → right down the panels.
    values = np.array([0.90, 0.65, 0.40, 0.15])
    t_spike = (1.0 - values) * T

    fig, axes = plt.subplots(
        len(values), 1, figsize=(8.2, 4.8), sharex=True,
        gridspec_kw={"hspace": 0.45},
    )

    for i, (ax, x, ts) in enumerate(zip(axes, values, t_spike)):
        # Faint guide line from the time origin to the spike.
        ax.plot([0, ts], [1, 1], color="#dddddd", ls=":", lw=1.0, zorder=0)
        ax.vlines(ts, 0, 1, color=SPIKE_BLUE, lw=2.6, zorder=3)
        ax.plot(ts, 1, "o", color=SPIKE_BLUE, ms=6, zorder=4)

        ax.set_ylim(0, 1.35)
        ax.set_xlim(0, T)
        ax.set_yticks([])
        ax.set_ylabel(rf"neuron {i + 1}" + "\n" + rf"$\hat{{x}} = {x:.2f}$",
                      rotation=0, ha="right", va="center", labelpad=16)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    # Trend annotations on the extreme panels (high value fires first, low last).
    axes[0].annotate(r"high $\hat{x}\;\Rightarrow$ early spike",
                     xy=(t_spike[0], 1.0), xytext=(t_spike[0] + 4, 1.15),
                     fontsize=10.5, color="#555555", va="center",
                     arrowprops=dict(arrowstyle="->", color="#999999", lw=1.1))
    axes[-1].annotate(r"low $\hat{x}\;\Rightarrow$ late spike",
                      xy=(t_spike[-1], 1.0), xytext=(t_spike[-1] - 20, 1.15),
                      fontsize=10.5, color="#555555", va="center",
                      arrowprops=dict(arrowstyle="->", color="#999999", lw=1.1))

    axes[-1].set_xlabel(r"time step $t$")
    axes[0].set_title(r"Latency Encoding", fontsize=13)

    _save(fig, "latency_encoding")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Helper — Heaviside step drawn with explicit segments and jump markers
# --------------------------------------------------------------------------- #
def _draw_heaviside(ax, u_min, u_max, color="#222222", lw=2.2, label=None):
    """Draw S = Θ(U) on `ax`: 0 for U<0, 1 for U≥0, with open/filled circles."""
    ax.plot([u_min, 0], [0, 0], color=color, lw=lw, label=label)
    ax.plot([0, u_max], [1, 1], color=color, lw=lw)
    ax.plot([0, 0], [0, 1], color=color, ls=":", lw=1.0)
    ax.plot(0, 0, marker="o", mfc="white", mec=color, ms=6.5, mew=1.5, zorder=5)
    ax.plot(0, 1, marker="o", color=color, ms=6.5, zorder=5)


# --------------------------------------------------------------------------- #
# Figure 5 — Non-differentiability of the spike function
# --------------------------------------------------------------------------- #
def plot_non_differentiability():
    """
    The spike function S_i = Θ(U_i − U_thr) (top) is discontinuous, so its
    derivative (bottom) is zero everywhere except at the threshold, where it is
    a Dirac impulse (→∞). The threshold is placed at the origin, so the x-axis
    is the membrane potential measured relative to U_thr.
    """
    u_min, u_max = -1.5, 1.5

    fig, (ax_s, ax_d) = plt.subplots(
        2, 1, figsize=(7.2, 5.4), sharex=True,
        gridspec_kw={"hspace": 0.30},
    )

    # ---- Top: Heaviside spike function ----------------------------------- #
    ax_s.axvline(0, color=C_THRESH, ls="--", lw=1.2, zorder=0)
    _draw_heaviside(ax_s, u_min, u_max)
    ax_s.annotate("discontinuous jump", xy=(0, 0.5), xytext=(0.45, 0.45),
                  fontsize=10, color="#555555", va="center",
                  arrowprops=dict(arrowstyle="->", color="#999999", lw=1.1))
    ax_s.set_ylim(-0.25, 1.4)
    ax_s.set_yticks([0, 1])
    ax_s.set_ylabel(r"$s_i[t]$", rotation=0, ha="right", va="center", labelpad=14)
    ax_s.set_title(r"Spike function $s_i[t]=\Theta(U_i[t]-U_{\mathrm{th}})$",
                   fontsize=12.5)
    for sp in ("top", "right"):
        ax_s.spines[sp].set_visible(False)

    # ---- Bottom: the ill-defined derivative ------------------------------ #
    ax_d.axvline(0, color=C_THRESH, ls="--", lw=1.2, zorder=0)
    ax_d.plot([u_min, 0], [0, 0], color="#222222", lw=2.2)
    ax_d.plot([0, u_max], [0, 0], color="#222222", lw=2.2)
    ax_d.plot(0, 0, marker="o", mfc="white", mec="#222222", ms=6.5, mew=1.5,
              zorder=5)
    # Dirac impulse at the threshold.
    ax_d.annotate("", xy=(0, 1.0), xytext=(0, 0.0),
                  arrowprops=dict(arrowstyle="-|>", color=C_RESET, lw=2.6,
                                  mutation_scale=20))
    ax_d.text(0.08, 1.0, r"$\rightarrow \infty$", color=C_RESET, va="center",
              ha="left", fontsize=12)
    ax_d.annotate(r"$=0$ for all $U_i \neq U_{\mathrm{th}}$",
                  xy=(-0.7, 0.0), xytext=(-1.45, 0.55),
                  fontsize=10.5, color="#555555", va="center",
                  arrowprops=dict(arrowstyle="->", color="#999999", lw=1.1))
    ax_d.text(0.72, 0.88, "no learning signal\n(dead-neuron problem)",
              ha="center", va="center", fontsize=10.5, color=C_RESET)

    ax_d.set_ylim(-0.3, 1.35)
    ax_d.set_yticks([0])
    ax_d.set_ylabel(r"$\partial s_i / \partial U_i$", rotation=0, ha="right",
                    va="center", labelpad=16)
    ax_d.set_xlabel(r"membrane potential $U_i[t]$")
    ax_d.set_xlim(u_min, u_max)
    ax_d.set_xticks([0])
    ax_d.set_xticklabels([r"$U_{\mathrm{th}}$"])
    for sp in ("top", "right"):
        ax_d.spines[sp].set_visible(False)

    _save(fig, "non_differentiability")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 6 — Arctangent surrogate gradient
# --------------------------------------------------------------------------- #
def plot_surrogate_gradient():
    """
    Surrogate-gradient learning. Forward pass (top): the true binary spike Θ is
    kept, while the smooth arctangent surrogate S̃_i = (1/π)arctan(πU)+½ is its
    differentiable stand-in. Backward pass (bottom): the ill-defined spike
    derivative is replaced by the smooth surrogate gradient
        ∂S̃_i/∂U_i = (1/π) · 1/(1 + (πU_i)²),
    which is nonzero around the threshold, so near-firing neurons still learn.
    """
    u_min, u_max = -1.5, 1.5
    U = np.linspace(u_min, u_max, 1000)
    surrogate = (1.0 / np.pi) * np.arctan(np.pi * U) + 0.5
    surrogate_grad = (1.0 / np.pi) / (1.0 + (np.pi * U) ** 2)
    peak = 1.0 / np.pi

    fig, (ax_f, ax_b) = plt.subplots(
        2, 1, figsize=(7.4, 5.6), sharex=True,
        gridspec_kw={"hspace": 0.30},
    )

    # ---- Top: forward pass — true spike + surrogate function ------------- #
    ax_f.axvline(0, color=C_THRESH, ls="--", lw=1.2, zorder=0)
    _draw_heaviside(ax_f, u_min, u_max, label=r"true spike $\Theta$ (forward)")
    ax_f.plot(U, surrogate, color=SPIKE_BLUE, lw=2.4,
              label=r"surrogate $\tilde{s}_i$")
    ax_f.set_ylim(-0.25, 1.4)
    ax_f.set_yticks([0, 1])
    ax_f.set_ylabel(r"$s_i[t]$", rotation=0, ha="right", va="center", labelpad=14)
    ax_f.set_title(r"Forward pass: binary spike kept", fontsize=12.5)
    ax_f.legend(loc="center right", frameon=False, fontsize=10)
    for sp in ("top", "right"):
        ax_f.spines[sp].set_visible(False)

    # ---- Bottom: backward pass — surrogate gradient ---------------------- #
    ax_b.axvline(0, color=C_THRESH, ls="--", lw=1.2, zorder=0)
    ax_b.axhline(0.0, color="#bbbbbb", lw=1.4)
    # The surrogate gradient.
    ax_b.plot(U, surrogate_grad, color=SPIKE_BLUE, lw=2.6)
    ax_b.plot(0, peak, "o", color=SPIKE_BLUE, ms=6, zorder=5)
    ax_b.text(0.13, peak, r"$\dfrac{1}{\pi}$", color=SPIKE_BLUE, fontsize=12,
              va="center", ha="left")
    ax_b.text(-1.45, 0.27,
              r"$\dfrac{\partial \tilde{s}_i[t]}{\partial U_i[t]}="
              r"\dfrac{1}{\pi}\,\dfrac{1}{1+(\pi(U_i[t]-U_{\mathrm{th}}))^2}$",
              color=SPIKE_BLUE, fontsize=11, va="center", ha="left")
    ax_b.annotate("nonzero near threshold\n(neurons can still learn)",
                  xy=(0.35, surrogate_grad[np.argmin(np.abs(U - 0.35))]),
                  xytext=(0.6, 0.22), fontsize=10, color=C_INTEGRATE,
                  va="center",
                  arrowprops=dict(arrowstyle="->", color=C_INTEGRATE, lw=1.2))

    ax_b.set_ylim(-0.03, 0.42)
    ax_b.set_yticks([0])
    ax_b.set_ylabel(r"$\partial \tilde{s}_i / \partial U_i$", rotation=0,
                    ha="right", va="center", labelpad=16)
    ax_b.set_xlabel(r"membrane potential $U_i[t]$")
    ax_b.set_xlim(u_min, u_max)
    ax_b.set_xticks([0])
    ax_b.set_xticklabels([r"$U_{\mathrm{th}}$"])
    ax_b.set_title(r"Backward pass: surrogate gradient", fontsize=12.5)
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(False)

    _save(fig, "surrogate_gradient")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Helper — arrow drawn as a FancyArrowPatch (straight or arced)
# --------------------------------------------------------------------------- #
def _arrow(ax, p0, p1, color, lw=1.8, rad=0.0, ls="-", mut=12, alpha=1.0):
    """Draw an arrow from p0 to p1 on `ax`; `rad` curves it (arc3)."""
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mut,
                        connectionstyle=f"arc3,rad={rad}", color=color,
                        lw=lw, linestyle=ls, alpha=alpha, zorder=3)
    ax.add_patch(a)
    return a


# --------------------------------------------------------------------------- #
# Figure 7 — Recurrent nature of the LIF neuron (folded ↔ unrolled in time)
# --------------------------------------------------------------------------- #
def plot_recurrence_unrolling():
    """
    The membrane potential carries an internal temporal state: U_i[t+1] depends
    on U_i[t] through the leak β, so a LIF neuron is a recurrent unit. Left: the
    folded (recurrent) form, with the self-loop β feeding the state back. Right:
    the same neuron unrolled through time into a chain of states, each receiving
    an input current I_i, emitting a spike S_i, and being reset by −U_thr.
    """
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.set_xlim(0.2, 11.0)
    ax.set_ylim(0.0, 4.8)
    ax.axis("off")

    node_fc = "#dce6f2"

    # ---- Folded (recurrent) form ---------------------------------------- #
    cx, cy = 1.55, 2.5
    box = FancyBboxPatch((cx - 0.62, cy - 0.52), 1.24, 1.04,
                         boxstyle="round,pad=0.02,rounding_size=0.12",
                         linewidth=1.8, edgecolor=SPIKE_BLUE,
                         facecolor=node_fc, zorder=2)
    ax.add_patch(box)
    ax.text(cx, cy, r"$U_i$", ha="center", va="center", fontsize=14)

    # self-loop = leak / recurrent state feedback
    _arrow(ax, (cx + 0.62, cy + 0.30), (cx + 0.62, cy - 0.30),
           C_LEAK, lw=1.8, rad=-1.4)
    ax.text(cx + 1.18, cy, r"$\beta$", color=C_LEAK, fontsize=15,
            ha="center", va="center")

    _arrow(ax, (cx, cy - 1.45), (cx, cy - 0.54), C_INTEGRATE, lw=1.8)
    ax.text(cx, cy - 1.72, r"$I_i[t]$", color=C_INTEGRATE, fontsize=12,
            ha="center", va="center")
    _arrow(ax, (cx, cy + 0.54), (cx, cy + 1.45), C_FIRE, lw=1.8)
    ax.text(cx, cy + 1.72, r"$S_i[t]$", color=C_FIRE, fontsize=12,
            ha="center", va="center")
    ax.text(cx, 0.2, "recurrent form", ha="center", va="center",
            fontsize=11, style="italic", color="#444444")

    # ---- Unfold arrow ---------------------------------------------------- #
    _arrow(ax, (3.05, 2.5), (4.15, 2.5), "#555555", lw=2.6, mut=20)
    ax.text(3.6, 3.0, "unfold\nin time", ha="center", va="center",
            fontsize=10.5, color="#555555")

    # ---- Unrolled (time-step) form -------------------------------------- #
    r = 0.6
    xs = [5.6, 7.6, 9.6]
    state_lbls = [r"$U_i[t{-}1]$", r"$U_i[t]$", r"$U_i[t{+}1]$"]
    in_lbls = [r"$I_i[t{-}1]$", r"$I_i[t]$", r"$I_i[t{+}1]$"]
    sp_lbls = [r"$S_i[t{-}1]$", r"$S_i[t]$", r"$S_i[t{+}1]$"]

    ax.text(4.7, cy, r"$\cdots$", ha="center", va="center",
            fontsize=18, color="#888888")
    ax.text(10.55, cy, r"$\cdots$", ha="center", va="center",
            fontsize=18, color="#888888")

    for x, slbl, ilbl, splbl in zip(xs, state_lbls, in_lbls, sp_lbls):
        c = Circle((x, cy), r, facecolor=node_fc, edgecolor=SPIKE_BLUE,
                   lw=1.8, zorder=2)
        ax.add_patch(c)
        ax.text(x, cy, slbl, ha="center", va="center", fontsize=10)
        _arrow(ax, (x, cy - 1.45), (x, cy - r - 0.02), C_INTEGRATE, lw=1.6)
        ax.text(x, cy - 1.72, ilbl, color=C_INTEGRATE, fontsize=10,
                ha="center", va="center")
        _arrow(ax, (x, cy + r + 0.02), (x, cy + 1.45), C_FIRE, lw=1.6)
        ax.text(x, cy + 1.72, splbl, color=C_FIRE, fontsize=10,
                ha="center", va="center")

    # β recurrence arrows between consecutive states
    for x0, x1 in zip(xs[:-1], xs[1:]):
        _arrow(ax, (x0 + r + 0.02, cy), (x1 - r - 0.02, cy), C_LEAK, lw=1.8)
        ax.text((x0 + x1) / 2.0, cy + 0.28, r"$\beta$", color=C_LEAK,
                fontsize=13, ha="center", va="center")

    # subtractive reset shown once on the middle state
    xm = xs[1]
    _arrow(ax, (xm + 0.18, cy + 0.95), (xm + r - 0.05, cy + 0.30),
           C_RESET, lw=1.5, rad=-0.7)
    ax.text(xm + 0.95, cy + 0.78, r"$-U_{\mathrm{thr}}$", color=C_RESET,
            fontsize=10, ha="center", va="center")

    ax.text(7.6, 0.2, "unrolled through time", ha="center", va="center",
            fontsize=11, style="italic", color="#444444")

    _save(fig, "recurrence_unrolling")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 8 — BPTT and the vanishing / exploding gradient problem
# --------------------------------------------------------------------------- #
def plot_bptt_gradient_flow():
    """
    Why SNNs are hard to train. The gradient of the loss L[t] with respect to an
    earlier membrane state U[τ] is a product of t−τ temporal Jacobians, so its
    magnitude scales as |J|^(t−τ). On a linear axis this exponential shape is
    preserved: stepping backward in time from t, the gradient either decays to
    zero (vanishing, |J|<1) or grows without bound (exploding, |J|>1). The
    x-axis runs from t at the right back to t−N at the left.
    """
    N = 8
    d = np.arange(0, N + 1)              # temporal distance t − τ
    curves = [
        (0.6, SPIKE_BLUE, "o", "-",  r"$|J|=0.6$  (vanishing)"),
        (1.0, C_THRESH,   "s", "--", r"$|J|=1.0$  (marginal)"),
        (1.35, C_RESET,   "^", "-",  r"$|J|>1$  (exploding)"),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    # y-range zoomed so the marginal line sits high and the vanishing curve
    # has room; the exploding curve rises steeply out of the top of the frame.
    y_top = 3.0
    for J, color, marker, ls, lbl in curves:
        ax.plot(d, J ** d.astype(float), color=color, lw=2.3, ls=ls,
                marker=marker, ms=6, label=lbl, clip_on=True)

    # Emphasis on the vanishing case.
    ax.annotate("early states receive\nalmost no gradient",
                xy=(N - 0.25, 0.6 ** N), xytext=(N - 3.6, 0.6),
                color=SPIKE_BLUE, fontsize=10, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=SPIKE_BLUE, lw=1.3))
    ax.text(N - 2.3, 2.5, "gradient\nexplodes",
            color=C_RESET, fontsize=10, ha="center", va="center")

    # x-axis: rightmost tick = t, decreasing to the left.
    ax.set_xticks(d)
    ax.set_xticklabels([r"$t$" if k == 0 else rf"$t-{k}$" for k in d])
    ax.set_xlim(N + 0.3, -0.3)           # reversed: t on the right
    ax.set_ylim(0, y_top)
    ax.set_yticks([])                     # qualitative y-axis only
    ax.set_xlabel(r"time step  $\tau$")
    ax.set_ylabel(r"gradient magnitude  $\|\,\partial\mathcal{L}[t]/"
                  r"\partial U_i[\tau]\,\|$")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    _save(fig, "bptt_gradient_flow")
    plt.close(fig)


# =========================================================================== #
# CHAPTER 4 (DATASET) FIGURES
#
# These figures read the raw simulation data (vorticity .mat + lift .csv) from
# Datasets/Dataset_original/, unlike the analytic SNN-background figures above.
# =========================================================================== #

# Physical / grid constants (Section 4.3).
DOMAIN = (-1.4, 4.0, -1.2, 1.2)      # (x, y)/c extent of the stored vorticity box

# Colour tags shared between each snapshot badge and its C_l marker
# (colour-blind-safe: teal / orange / purple).
SNAP_COLORS = ["#1b9e77", "#d95f02", "#7570b3"]
SNAP_LABELS = ["pre-impingement", "impingement", "post-impingement"]

# Representative cases to render, chosen to span the pedagogy of the chapter:
# both gust-rotation signs, different vortex sizes R, and a vortex passing
# below / through / above the airfoil (y = -0.5, 0.0, +0.5).
# snap_idx = [pre, impingement, post] snapshot indices.
DATASET_CASES = [
    dict(aoa=20, case="RD4",  G=+3.80, R=1.00, y=-0.50, snap_idx=[160, 369, 600]),
    dict(aoa=40, case="RD29", G=-3.40, R=0.75, y=+0.00, snap_idx=[195, 315, 560]),
    dict(aoa=60, case="RD9",  G=-3.00, R=1.00, y=+0.50, snap_idx=[120, 320, 520]),
]


def _find_data_root():
    """Locate the Dataset_original folder (mirrors dataset_interpolation.py)."""
    env = os.environ.get("THESIS_DATA_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(here, "..")
    # Datasets live under the repo root, e.g. <repo>/datasets/Dataset_original/
    for name in ("Datasets", "DATASETS", "datasets"):
        candidate = os.path.normpath(os.path.join(base, name, "Dataset_original"))
        if os.path.isdir(candidate):
            return candidate
    # Legacy fallback: datasets live directly under the repo root (<repo>/Dataset/)
    for name in ("Dataset", "DATASET", "dataset"):
        legacy = os.path.normpath(os.path.join(base, name))
        if os.path.isdir(legacy):
            return legacy
    return os.path.normpath(os.path.join(base, "datasets", "Dataset_original"))


DATA_ROOT = _find_data_root()


def load_vorticity(aoa, case):
    """Return omega (Nt, Ny, Nx) float32 for one case."""
    path = os.path.join(
        DATA_ROOT, "vorticity", f"vort_a{aoa}", f"Vort_AoA{aoa}_{case}.mat")
    return scipy.io.loadmat(path)["omg_box"].astype(np.float32)


def load_lift(aoa, case):
    """Return Cl (Nt,) float32 for one case."""
    path = os.path.join(
        DATA_ROOT, "lift", f"lift_a{aoa}", f"Lift_AoA{aoa}_{case}.csv")
    return pd.read_csv(path, header=None).values.flatten().astype(np.float32)


def naca0012_polygon(aoa_deg, n=120):
    """
    Closed NACA 0012 outline, chord = 1, leading edge at the origin, rotated
    clockwise by the angle of attack (trailing edge pitched down) so the
    incidence matches a positive-lift configuration in a +x free stream.
    """
    x = (1.0 - np.cos(np.linspace(0.0, np.pi, n))) / 2.0   # cosine clustering
    yt = 5 * 0.12 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
        + 0.2843 * x ** 3 - 0.1036 * x ** 4
    )
    xc = np.concatenate([x, x[::-1]])
    yc = np.concatenate([yt, -yt[::-1]])

    a = np.deg2rad(aoa_deg)
    xr = xc * np.cos(a) + yc * np.sin(a)
    yr = -xc * np.sin(a) + yc * np.cos(a)
    return np.column_stack([xr, yr])


def _add_airfoil(ax, aoa_deg, zorder=5):
    ax.add_patch(Polygon(
        naca0012_polygon(aoa_deg), closed=True,
        facecolor="#d9d9d9", edgecolor="black", linewidth=1.1, zorder=zorder,
    ))


# --------------------------------------------------------------------------- #
# Figure 9 — Representative vortex–gust case (snapshots + lift history)
# --------------------------------------------------------------------------- #
def plot_dataset_case(spec):
    """
    Composite figure for one vortex--gust case: three vorticity snapshots
    (pre-impingement, impingement, post-impingement) with the airfoil overlaid,
    above the sectional lift history C_l(t) with those three instants marked.
    """
    aoa, case = spec["aoa"], spec["case"]
    snap_idx = spec["snap_idx"]

    omega = load_vorticity(aoa, case)
    cl = load_lift(aoa, case)
    T = omega.shape[0]
    t = np.arange(T)

    # Symmetric colour scale, robust to the extreme vortex-core peaks, shared
    # by the three snapshots so their colours are directly comparable.
    frames = np.stack([omega[i] for i in snap_idx])
    vmax = float(np.percentile(np.abs(frames), 99.0))

    # STIX math so the vorticity tensor Omega renders bold (Computer Modern has
    # no bold uppercase Greek); restored before returning.
    _prev_fs = plt.rcParams["mathtext.fontset"]
    plt.rcParams["mathtext.fontset"] = "stix"

    fig = plt.figure(figsize=(13.0, 7.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.30)
    top = outer[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.10)
    axes_snap = [fig.add_subplot(top[0, i]) for i in range(3)]
    cax = fig.add_subplot(top[0, 3])
    ax_cl = fig.add_subplot(outer[1])

    # ---- Top row: vorticity snapshots ------------------------------------- #
    im = None
    for k, (ax, idx) in enumerate(zip(axes_snap, snap_idx)):
        im = ax.imshow(
            omega[idx], origin="lower", extent=DOMAIN, aspect="equal",
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear",
        )
        _add_airfoil(ax, aoa)
        ax.set_title(rf"$t = {idx}/{T}$" + "\n" + SNAP_LABELS[k],
                     fontsize=12, pad=6)
        ax.set_xlabel(r"$x/c$")
        if k == 0:
            ax.set_ylabel(r"$y/c$")
        else:
            ax.set_yticklabels([])
        # Colour badge tying this snapshot to its marker on the Cl curve.
        ax.text(0.045, 0.90, f"{k + 1}", transform=ax.transAxes,
                fontsize=12, fontweight="bold", color="white",
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.28", fc=SNAP_COLORS[k],
                          ec="white", lw=1.2))

    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\boldsymbol{\Omega}$", rotation=0, labelpad=10)

    # ---- Bottom row: lift history (x-axis = timestep t) -------------------- #
    ax_cl.plot(t, cl, color="#222222", lw=1.8, zorder=3)

    y0, y1 = ax_cl.get_ylim()
    for k, idx in enumerate(snap_idx):
        # dotted guides down to the t-axis and across to the C_l-axis, so both
        # the timestep and the corresponding lift value can be read off.
        ax_cl.axvline(idx, color=SNAP_COLORS[k], ls=":", lw=1.1,
                      ymax=(cl[idx] - y0) / (y1 - y0), zorder=1, alpha=0.8)
        ax_cl.plot([0, idx], [cl[idx], cl[idx]], color=SNAP_COLORS[k],
                   ls=":", lw=1.1, alpha=0.8, zorder=1)
        ax_cl.plot(idx, cl[idx], "o", ms=11, color=SNAP_COLORS[k],
                   mec="white", mew=1.4, zorder=5)
        ax_cl.text(idx, cl[idx], f"{k + 1}", color="white",
                   fontsize=9, fontweight="bold", ha="center", va="center",
                   zorder=6)

    ax_cl.set_xlim(0, T - 1)
    ax_cl.set_xlabel(r"timestep $t$")
    ax_cl.set_ylabel(r"$C_l$")
    for sp in ("top", "right"):
        ax_cl.spines[sp].set_visible(False)

    # ---- Case parameters as the figure title ------------------------------ #
    fig.suptitle(
        rf"$\alpha = {aoa}^\circ$,  "
        rf"$G = {spec['G']:+.2f}$,  "
        rf"$R = {spec['R']:.2f}$,  "
        rf"$y = {spec['y']:+.2f}$",
        fontsize=15, y=0.99,
    )

    _save(fig, f"case_AoA{aoa}_{case}")
    plt.close(fig)
    plt.rcParams["mathtext.fontset"] = _prev_fs


# Reconstruction-overlay case (Chapter 7 style): same composite as
# plot_dataset_case, but the lift panel overlays the reference-SNN reconstructed
# C_l on top of the ground-truth C_l. The reconstructed trace is loaded from the
# .npy dumped alongside the model's per-case results (original-frame grid), so no
# re-inference is needed here.
RECON_CASE = dict(
    aoa=20, case="RD29", G=1.20, R=0.75, y=-0.10, snap_idx=[80, 141, 470],
    recon_npy=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results",
        "SNN_results", "RateOnly_InterpolatedN4_Stateful_Nz16_epoch100_Warmup_Best",
        "samples", "AoA20_RD29_G_1.2", "AoA20_RD29_reconstructed_cl_orig.npy")),
)
RECON_COLOR = "#d62728"          # reconstructed C_l (crimson dashed)


def plot_dataset_case_reconstructed(spec):
    """
    Composite figure for one vortex--gust case with the SNN reconstruction
    overlaid: three vorticity snapshots (pre-impingement, impingement,
    post-impingement) above the sectional lift history, where the reference-SNN
    reconstructed C_l(t) is drawn on the same axes as the ground-truth C_l(t).
    The snapshot badges/markers are placed on the ground-truth curve only.
    """
    aoa, case = spec["aoa"], spec["case"]
    snap_idx = spec["snap_idx"]

    omega = load_vorticity(aoa, case)
    cl = load_lift(aoa, case)
    cl_recon = np.load(spec["recon_npy"]).astype(np.float32)
    T = omega.shape[0]
    t = np.arange(T)
    # Align the reconstructed trace to the original-frame grid (guard against an
    # off-by-one from the interpolation subsampling).
    L = min(T, len(cl_recon))

    frames = np.stack([omega[i] for i in snap_idx])
    vmax = float(np.percentile(np.abs(frames), 99.0))

    _prev_fs = plt.rcParams["mathtext.fontset"]
    plt.rcParams["mathtext.fontset"] = "stix"

    fig = plt.figure(figsize=(13.0, 7.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.30)
    top = outer[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.10)
    axes_snap = [fig.add_subplot(top[0, i]) for i in range(3)]
    cax = fig.add_subplot(top[0, 3])
    ax_cl = fig.add_subplot(outer[1])

    # ---- Top row: vorticity snapshots ------------------------------------- #
    im = None
    for k, (ax, idx) in enumerate(zip(axes_snap, snap_idx)):
        im = ax.imshow(
            omega[idx], origin="lower", extent=DOMAIN, aspect="equal",
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear",
        )
        _add_airfoil(ax, aoa)
        ax.set_title(rf"$t = {idx}/{T}$" + "\n" + SNAP_LABELS[k],
                     fontsize=12, pad=6)
        ax.set_xlabel(r"$x/c$")
        if k == 0:
            ax.set_ylabel(r"$y/c$")
        else:
            ax.set_yticklabels([])
        ax.text(0.045, 0.90, f"{k + 1}", transform=ax.transAxes,
                fontsize=12, fontweight="bold", color="white",
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.28", fc=SNAP_COLORS[k],
                          ec="white", lw=1.2))

    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\boldsymbol{\Omega}$", rotation=0, labelpad=10)

    # ---- Bottom row: ground-truth vs reconstructed lift ------------------- #
    ax_cl.plot(t, cl, color="#222222", lw=1.8, zorder=3, label="Ground Truth")
    ax_cl.plot(t[:L], cl_recon[:L], color=RECON_COLOR, lw=1.7, ls="--",
               zorder=4, label="Reconstructed")

    y0, y1 = ax_cl.get_ylim()
    for k, idx in enumerate(snap_idx):
        # Snapshot markers are tied to the GROUND-TRUTH curve only.
        ax_cl.axvline(idx, color=SNAP_COLORS[k], ls=":", lw=1.1,
                      ymax=(cl[idx] - y0) / (y1 - y0), zorder=1, alpha=0.8)
        ax_cl.plot([0, idx], [cl[idx], cl[idx]], color=SNAP_COLORS[k],
                   ls=":", lw=1.1, alpha=0.8, zorder=1)
        ax_cl.plot(idx, cl[idx], "o", ms=11, color=SNAP_COLORS[k],
                   mec="white", mew=1.4, zorder=5)
        ax_cl.text(idx, cl[idx], f"{k + 1}", color="white",
                   fontsize=9, fontweight="bold", ha="center", va="center",
                   zorder=6)

    ax_cl.set_xlim(0, T - 1)
    ax_cl.set_xlabel(r"timestep $t$")
    ax_cl.set_ylabel(r"$C_l$")
    ax_cl.legend(loc="upper right", frameon=True, framealpha=0.95,
                 fontsize=12, ncol=1)
    for sp in ("top", "right"):
        ax_cl.spines[sp].set_visible(False)

    fig.suptitle(
        rf"$\alpha = {aoa}^\circ$,  "
        rf"$G = {spec['G']:+.2f}$,  "
        rf"$R = {spec['R']:.2f}$,  "
        rf"$y = {spec['y']:+.2f}$",
        fontsize=15, y=0.99,
    )

    _save(fig, f"case_AoA{aoa}_{case}_reconstructed")
    plt.close(fig)
    plt.rcParams["mathtext.fontset"] = _prev_fs


# --------------------------------------------------------------------------- #
# Figure 10 — Schematic of the physical setup
# --------------------------------------------------------------------------- #
def plot_setup_schematic():
    """
    Labelled sketch of the simulation configuration: the NACA 0012 airfoil at
    incidence with its leading edge at the origin, the incoming Taylor vortex
    released upstream at x_0/c = -2 and convected by the free stream, the gust
    parameters (G, R, y) annotated, the coordinate axes, and the stored data
    subdomain outlined.
    """
    aoa = 30.0
    x0, y_g, R = -2.0, 0.45, 0.5          # vortex: centre (x0, y_g), radius R
    x_ym = -2.75                          # x of the vertical y-offset measurement

    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    ax.set_xlim(-3.5, 4.35)
    ax.set_ylim(-2.05, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")

    dim = "#8a8a8a"

    # ---- Data subdomain outline ------------------------------------------- #
    ax.add_patch(Rectangle(
        (DOMAIN[0], DOMAIN[2]), DOMAIN[1] - DOMAIN[0], DOMAIN[3] - DOMAIN[2],
        fill=False, ls="--", ec=dim, lw=1.3, zorder=1))
    ax.text(DOMAIN[1] - 0.06, DOMAIN[3] - 0.11,
            r"data subdomain: $x/c\in[-1.4,\,4]$,  $y/c\in[-1.2,\,1.2]$"
            + "\n" + r"grid points: $240\times120$",
            ha="right", va="top", fontsize=9.5, color=dim, linespacing=1.5)

    # ---- y = 0 reference line upstream of the airfoil --------------------- #
    ax.plot([x_ym - 0.15, 0], [0, 0], color=dim, ls=":", lw=1.0, zorder=1)

    # ---- Coordinate axes at the leading edge ------------------------------ #
    _arrow(ax, (0, 0), (1.25, 0), "#333333", lw=1.4, mut=13)
    _arrow(ax, (0, 0), (0, 1.25), "#333333", lw=1.4, mut=13)
    ax.text(1.30, 0.0, r"$x/c$", ha="left", va="center", fontsize=12)
    ax.text(0.0, 1.32, r"$y/c$", ha="center", va="bottom", fontsize=12)

    # ---- Free-stream inflow ----------------------------------------------- #
    for yy in (-1.5, -0.8, 0.95, 1.6):
        _arrow(ax, (-3.40, yy), (-3.05, yy), "#3576b5", lw=1.8, mut=13)
    ax.text(-3.40, 1.95, r"$U_\infty$", color="#3576b5", fontsize=13,
            ha="left", va="center")

    # ---- Airfoil + angle of attack ---------------------------------------- #
    a = np.deg2rad(aoa)
    te = (np.cos(a), -np.sin(a))
    ax.plot([0, te[0]], [0, te[1]], color="#444444", ls="--", lw=1.0, zorder=4)
    ax.add_patch(Arc((0, 0), 1.2, 1.2, angle=0, theta1=-aoa, theta2=0,
                     color="#444444", lw=1.2, zorder=4))
    ax.text(0.74, -0.19, r"$\alpha$", fontsize=13, color="#444444",
            ha="center", va="center")
    _add_airfoil(ax, aoa)
    ax.text(te[0] + 0.12, te[1] - 0.2, "NACA 0012", fontsize=11,
            ha="left", va="center")

    # ---- Taylor vortex (positive gust: counter-clockwise, omega > 0) ------ #
    ax.add_patch(Circle((x0, y_g), R, facecolor="#fdece1",
                        edgecolor="#d9662a", lw=1.7, zorder=3))
    # counter-clockwise rotation arc (sweeps up the right-hand side so the
    # arrowhead at the top points in -x, matching the u_g arrow)
    swirl = FancyArrowPatch(
        (x0, y_g - 0.28), (x0, y_g + 0.28),
        connectionstyle="arc3,rad=0.7", arrowstyle="-|>", mutation_scale=15,
        color="#d9662a", lw=1.8, zorder=4)
    ax.add_patch(swirl)
    # tangential gust velocity u_g at the top of the core (points -x for CCW)
    _arrow(ax, (x0, y_g + R), (x0 - 0.55, y_g + R), "#d9662a", lw=1.8, mut=13)
    ax.text(x0 - 0.62, y_g + R + 0.02, r"$u_g$", color="#d9662a",
            fontsize=12, ha="right", va="center")
    # gust ratio, drawn for the positive case with its rotational sense
    ax.text(x0, y_g - R - 0.20, r"$G = u_g/U_\infty$", color="#d9662a",
            fontsize=12, ha="center", va="top")
    ax.text(x0, y_g - R - 0.46, r"(positive $\Rightarrow$ counter-clockwise)",
            color="#d9662a", fontsize=9, ha="center", va="top")

    # vortex core radius R (radius drawn to the upper-left, label outside the
    # core so it sits clear of every line)
    r_end = (x0 + R * np.cos(np.deg2rad(150)),
             y_g + R * np.sin(np.deg2rad(150)))
    _arrow(ax, (x0, y_g), r_end, "#8c3d10", lw=1.4, mut=11)
    ax.text(r_end[0] - 0.03, r_end[1] + 0.04, r"$R = r_g/c$", color="#8c3d10",
            fontsize=11, ha="right", va="bottom")

    # vertical position y_g, measured on a dedicated line to the left of the
    # core (guide from the core centre) so the label is legible and unambiguous
    ax.plot([x_ym, x0], [y_g, y_g], color=dim, ls=":", lw=0.9, zorder=1)
    ax.annotate("", xy=(x_ym, y_g), xytext=(x_ym, 0.0),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.3))
    ax.text(x_ym - 0.08, 0.5 * y_g, r"$y = y_g/c$", ha="right", va="center",
            fontsize=11)

    # streamwise release position x_0
    y_meas = -1.85
    ax.plot([x0, x0], [-0.90, y_meas], color=dim, ls=":", lw=0.9, zorder=1)
    ax.plot([0, 0], [0, y_meas], color=dim, ls=":", lw=0.9, zorder=1)
    ax.annotate("", xy=(x0, y_meas), xytext=(0, y_meas),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.3))
    ax.text(0.5 * x0, y_meas + 0.06, r"$x_0/c = -2$", ha="center", va="bottom",
            fontsize=11)

    # ---- Convection toward the airfoil ------------------------------------ #
    _arrow(ax, (x0 + R + 0.05, y_g), (-0.55, y_g), "#555555", lw=1.6,
           mut=15, ls="--")

    _save(fig, "setup_schematic")
    plt.close(fig)


# =========================================================================== #
# CHAPTER 5 (INTERPOLATION & SPIKE ENCODING) FIGURE
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Figure 11 — Preprocessing and spike-encoding pipeline overview
# --------------------------------------------------------------------------- #
def plot_encoding_pipeline():
    """
    Overview of the preprocessing / spike-encoding pipeline of Chapter 5:

        Omega_{1:T}, Cl   --interp-->   Omega_{1:T'}   --norm+clip-->
        Omega~_{1:T'}     --encode E--> S_{1:T'}  (rate channels),

    with the lift signal Cl kept time-aligned at every stage. Thumbnails and
    spike rasters are built from one real snapshot (AoA 20, RD4) so the encoding
    is illustrated on actual vorticity data.
    """
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(1)

    # ---- Build the visuals from one real vorticity snapshot --------------- #
    omega = load_vorticity(20, "RD4")            # (Nt, Ny, Nx)
    cl = load_lift(20, "RD4")
    t0 = 369
    scale = float(np.percentile(np.abs(omega[::15]), 99))   # P99 robust scale
    raw = omega[t0]
    vdisp = 3.0 * scale            # display cap revealing structure (raw units)
    norm = np.clip(raw / scale, -1.0, 1.0)                    # normalized + clip

    gamma = 0.6                                              # rate transfer
    rate_pos = (rng.random(norm.shape) < np.clip(norm, 0, 1) ** gamma)
    rate_neg = (rng.random(norm.shape) < np.clip(-norm, 0, 1) ** gamma)

    cmap_b = ListedColormap(["white", SPIKE_BLUE])

    # ---- Canvas ----------------------------------------------------------- #
    # Computer Modern math has no bold uppercase Greek, so \boldsymbol{\Omega}
    # silently falls back to regular. Use STIX math here so the vorticity
    # tensor Omega renders bold (C_l, not wrapped in \boldsymbol, stays upright).
    _prev_fontset = plt.rcParams["mathtext.fontset"]
    plt.rcParams["mathtext.fontset"] = "stix"
    # Smaller physical size (same aspect) so the text scales up more when the
    # figure is placed at \textwidth in the report.
    fig, ax = plt.subplots(figsize=(14.0, 5.35))
    ax.set_xlim(0, 18.9)
    ax.set_ylim(1.7, 8.05)
    ax.axis("off")

    xc = [3.1, 7.5, 11.7, 15.7]
    y_f = 5.6                                               # field row centre
    fw, fh = 2.6, 1.3
    y_title = 7.45
    y_lift = 3.9                                            # middle row (lift)
    y_cap = 2.45                                            # last row (labels)

    def card_stack(cx, img, cmap, vmin, vmax, ncards, ec="black"):
        for i in range(ncards, 0, -1):
            ax.add_patch(Rectangle((cx - fw / 2 + i * 0.09, y_f - fh / 2 + i * 0.11),
                                   fw, fh, facecolor="white", edgecolor=ec,
                                   lw=1.0, zorder=2))
        ax.imshow(img, extent=[cx - fw / 2, cx + fw / 2, y_f - fh / 2, y_f + fh / 2],
                  origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="bilinear", zorder=3)
        ax.add_patch(Rectangle((cx - fw / 2, y_f - fh / 2), fw, fh, fill=False,
                               edgecolor=ec, lw=1.4, zorder=4))

    def title(cx, s):
        ax.text(cx, y_title, s, ha="center", va="center", fontsize=16,
                fontweight="bold", color="#333333", linespacing=1.2)

    def caption(cx, lines):
        ax.text(cx, y_cap, "\n".join(lines), ha="center", va="center",
                fontsize=21, linespacing=1.5)

    # ---- Stage 1: raw simulation ------------------------------------------ #
    card_stack(xc[0], raw, "RdBu_r", -vdisp, vdisp, 2)
    title(xc[0], "Raw\nsimulation")
    caption(xc[0], [r"$\boldsymbol{\Omega}_{1:T},\ C_{l,\,1:T}$"])

    # ---- Stage 2: temporal interpolation ---------------------------------- #
    card_stack(xc[1], raw, "RdBu_r", -vdisp, vdisp, 4)
    title(xc[1], "Temporally\ninterpolated")
    caption(xc[1], [r"$\boldsymbol{\Omega}_{1:T'},\ C_{l,\,1:T'}$"])

    # ---- Stage 3: normalize + clip ---------------------------------------- #
    card_stack(xc[2], norm, "RdBu_r", -1, 1, 4)
    title(xc[2], "Normalized\n& clipped")
    caption(xc[2], [r"$\tilde{\boldsymbol{\Omega}}_{1:T'},\ C_{l,\,1:T'}$"])

    # ---- Stage 4: spike encoding (rate channel stack) --------------------- #
    # The reference model uses rate encoding only, so just the positive and
    # negative rate channels are shown, centred about the field row.
    sp_specs = [(rate_pos, cmap_b, r"rate$^{+}$"), (rate_neg, cmap_b, r"rate$^{-}$")]
    sw, sh = 2.5, 0.47
    ys = [y_f + 0.26, y_f - 0.26]
    cx3 = xc[3]
    for (img, cmap, lbl), yy in zip(sp_specs, ys):
        ax.imshow(img, extent=[cx3 - sw / 2, cx3 + sw / 2, yy - sh / 2, yy + sh / 2],
                  origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1, zorder=3)
        ax.add_patch(Rectangle((cx3 - sw / 2, yy - sh / 2), sw, sh, fill=False,
                               edgecolor="#555555", lw=1.1, zorder=4))
        ax.text(cx3 + sw / 2 + 0.16, yy, lbl, ha="left", va="center",
                fontsize=21, color=SPIKE_BLUE)
    title(cx3, "Spike tensor\nsequence")
    caption(cx3, [r"$\mathbf{S}_{1:T'},\ C_{l,\,1:T'}$"])

    # ---- Operation arrows between stages ---------------------------------- #
    lefts = [xc[0] + fw / 2, xc[1] + fw / 2, xc[2] + fw / 2]
    rights = [xc[1] - fw / 2, xc[2] - fw / 2, cx3 - sw / 2 - 0.1]
    for lft, rgt in zip(lefts, rights):
        _arrow(ax, (lft + 0.45, y_f), (rgt - 0.05, y_f), "#333333", lw=2.2,
               mut=21)

    # ---- Lift lane (middle row): Cl carried along ------------------------- #
    a, b = 150, 470
    clw = cl[a:b].astype(float)
    clw_n = (clw - clw.min()) / (clw.max() - clw.min())
    L = len(clw)

    def mini_lift(cx, dense):
        x = cx - 1.15 + np.arange(L) / (L - 1) * 2.3
        y = y_lift - 0.45 + 0.9 * clw_n
        ax.plot(x, y, color="#222222", lw=1.7, zorder=3)
        oi = np.linspace(0, L - 1, 9).astype(int)          # original samples
        ax.plot(x[oi], y[oi], "o", ms=6.5, color="#222222", zorder=4)
        if dense:
            ii = np.linspace(0, L - 1, 41).astype(int)     # interpolated
            ax.plot(x[ii], y[ii], "o", ms=3.6, mfc="white", mec="#d9662a",
                    mew=1.1, zorder=3)

    for k, cx in enumerate(xc):
        mini_lift(cx, dense=(k >= 1))

    # ---- Row labels on the left ------------------------------------------- #
    x_lbl = xc[0] - fw / 2 - 0.35
    ax.text(x_lbl, y_f, "vorticity" + "\n" + r"$\boldsymbol{\Omega}$", ha="right",
            va="center", fontsize=21, linespacing=1.3)
    ax.text(x_lbl, y_lift, "lift" + "\n" + r"$C_l$", ha="right",
            va="center", fontsize=21, linespacing=1.3)
    # marker legend, in the gap between the lift curve and the field above,
    # colour-matched to the markers (filled black = original, open orange =
    # interpolated)
    y_leg = 0.5 * ((y_lift + 0.45) + (y_f - fh / 2))
    ax.text(xc[1] - 0.15, y_leg, r"$\bullet$ original", ha="right",
            va="center", fontsize=14, color="black")
    ax.text(xc[1] + 0.15, y_leg, r"$\circ$ interpolated", ha="left",
            va="center", fontsize=14, color="#d9662a")

    _save(fig, "encoding_pipeline")
    plt.rcParams["mathtext.fontset"] = _prev_fontset
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figures 12–13 — Why the 99th percentile is the right normalization scale
# --------------------------------------------------------------------------- #
def plot_normalization_choice():
    """
    Justify robust percentile normalization over max-abs, and P99 over P95.
    Produces two figures:

      normalization_fields       — one real snapshot normalized+clipped by the
                                    global |Omega|_max, P99 and P95, with clipped
                                    pixels highlighted in gold.
      normalization_distribution — the distribution of |Omega| over a
                                    representative sample of the dataset, with
                                    the three candidate scales marked.
    """
    rng = np.random.default_rng(0)

    # ---- Sample |Omega| across a spread of angles and gust strengths ------ #
    sample_cases = [(20, "RD4"), (20, "RD8"), (30, "RD2"), (40, "RD16"),
                    (40, "RD6"), (50, "RD4"), (60, "RD9"), (60, "base")]
    samp, absmax = [], 0.0
    for a, c in sample_cases:
        o = load_vorticity(a, c)
        absmax = max(absmax, float(np.abs(o).max()))
        v = np.abs(o[::40]).ravel()
        samp.append(v[rng.choice(v.size, min(120000, v.size), replace=False)])
        del o
    s = np.concatenate(samp)
    p95, p99 = (float(x) for x in np.percentile(s, [95, 99]))

    frame = load_vorticity(40, "RD6")[300]     # moderate case, balanced at P99
    scales = [(r"|\boldsymbol{\Omega}|_{\max}", absmax, "washed out"),
              (r"P_{99}", p99, "structures preserved"),
              (r"P_{95}", p95, "over-clipped")]

    # STIX math so the vorticity tensor Omega renders bold (Computer Modern has
    # no bold uppercase Greek); restored at the end.
    _prev_fs = plt.rcParams["mathtext.fontset"]
    plt.rcParams["mathtext.fontset"] = "stix"

    # ================================================================= #
    # Figure A — fields under each scale (clipped pixels shown in gold)
    # ================================================================= #
    gold = "#f2b100"
    hl = plt.get_cmap("RdBu_r").copy()
    hl.set_over(gold)
    hl.set_under(gold)

    fig = plt.figure(figsize=(11.0, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12)
    axf = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    im = None
    for i, (name, sc, note) in enumerate(scales):
        sat = float((np.abs(frame) > sc).mean()) * 100.0
        im = axf[i].imshow(frame / sc, origin="lower", extent=DOMAIN,
                           aspect="equal", cmap=hl, vmin=-1, vmax=1,
                           interpolation="nearest")
        _add_airfoil(axf[i], 40)
        axf[i].set_title(rf"$\boldsymbol{{\Omega}}\,/\,{name}$"
                         + "\n" + f"{sat:.0f}% clipped"
                         + "\n" + note, fontsize=16, linespacing=1.45)
        axf[i].set_xlabel(r"$x/c$", fontsize=15)
        axf[i].tick_params(labelsize=13)
        axf[i].set_ylabel(r"$y/c$", fontsize=15) if i == 0 \
            else axf[i].set_yticklabels([])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(r"$\tilde{\boldsymbol{\Omega}}$", rotation=0, labelpad=12,
                 fontsize=16)
    cb.ax.tick_params(labelsize=13)
    fig.text(0.5, 0.02, r"gold $=$ pixels clipped to $\pm 1$", ha="center",
             va="bottom", fontsize=15, color="#9a7400")
    _save(fig, "normalization_fields")
    plt.close(fig)

    # ================================================================= #
    # Figure B — |Omega| distribution with the three candidate scales
    # ================================================================= #
    fig, axh = plt.subplots(figsize=(9.5, 4.5))
    bins = np.logspace(np.log10(3e-3), np.log10(absmax * 1.2), 90)
    axh.hist(s[s > bins[0]], bins=bins, color="#c9d7e8", edgecolor="#8ba7c8",
             linewidth=0.4)
    axh.set_xscale("log")
    axh.set_xlim(bins[0], absmax * 1.2)
    y1 = axh.get_ylim()[1]

    c95, c99, cmax = "#d9662a", "#2e8b57", "#c0392b"
    labels = [(p95, c95, "right", rf"$P_{{95}}={p95:.2f}$" + "\n5% clipped"),
              (p99, c99, "left", rf"$P_{{99}}={p99:.2f}$" + "\n1% clipped"),
              (absmax, cmax, "center",
               rf"$|\boldsymbol{{\Omega}}|_{{\max}}={absmax:.1f}$" + "\n0% clipped")]
    for xval, col, ha, lab in labels:
        axh.axvline(xval, color=col, ls="--", lw=1.8)
        axh.text(xval, y1 * 1.03, lab, color=col, ha=ha, va="bottom",
                 fontsize=15, linespacing=1.35)

    # outlier gap annotation (P99 -> max)
    axh.annotate("", xy=(absmax, y1 * 0.5), xytext=(p99, y1 * 0.5),
                 arrowprops=dict(arrowstyle="<->", color=cmax, lw=1.5))
    axh.text(np.sqrt(p99 * absmax), y1 * 0.54,
             rf"$\times{absmax / p99:.0f}$ — a rare outlier",
             color=cmax, ha="center", va="bottom", fontsize=15)

    axh.set_xlabel(r"vorticity magnitude $|\boldsymbol{\Omega}|$  (log scale)",
                   fontsize=16)
    axh.set_ylabel("pixel count", fontsize=16)
    axh.tick_params(labelsize=13)
    axh.set_ylim(0, y1 * 1.22)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)

    _save(fig, "normalization_distribution")
    plt.close(fig)
    plt.rcParams["mathtext.fontset"] = _prev_fs


# --------------------------------------------------------------------------- #
# Figures 14–15 — Rate-encoding transfer function (gamma and gain, separately)
# --------------------------------------------------------------------------- #
def plot_rate_transfer_curves():
    """
    The rate-encoding transfer function p = min(1, g |Omega~|^gamma), rendered
    as two standalone figures so each has room for its labels/legend:
        rate_gamma_curve — effect of the exponent gamma at fixed gain (g=1).
        rate_gain_curve  — effect of the gain g at fixed exponent (gamma=1),
                           showing the saturation plateau.
    """
    _prev_fs = plt.rcParams["mathtext.fontset"]
    plt.rcParams["mathtext.fontset"] = "stix"
    x = np.linspace(0.0, 1.0, 500)

    def _finish(ax, title):
        ax.set_title(title, fontsize=18)
        ax.set_xlabel(r"normalized vorticity magnitude "
                      r"$|\tilde{\boldsymbol{\Omega}}|$", fontsize=17)
        ax.set_ylabel(r"spike probability $p$", fontsize=17)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.07)
        ax.tick_params(labelsize=14)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # ---- Figure: effect of gamma (g = 1) ---------------------------------- #
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for gm, col in [(0.4, "#c0392b"), (0.6, "#e08214"), (1.0, "#000000"),
                    (2.0, "#2166ac")]:
        ls = "--" if gm == 1.0 else "-"
        lab = rf"$\gamma={gm}$" + (" (linear)" if gm == 1.0 else "")
        ax.plot(x, x ** gm, color=col, lw=2.8, ls=ls, label=lab)
    # white boxes keep the annotations legible where they cross a curve
    box = dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.8,
               alpha=0.92)
    ax.annotate(r"$\gamma<1$: weak / moderate" + "\n" + "vorticity boosted",
                xy=(0.20, 0.20 ** 0.4), xytext=(0.045, 0.88),
                fontsize=14.5, color="#c0392b", ha="left", va="center", bbox=box,
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))
    ax.annotate(r"$\gamma>1$: spikes concentrate" + "\n" + "on strong vorticity",
                xy=(0.52, 0.27), xytext=(0.30, 0.45),
                fontsize=14.5, color="#2166ac", ha="left", va="center", bbox=box,
                arrowprops=dict(arrowstyle="->", color="#2166ac", lw=1.5))
    _finish(ax, r"Effect of the exponent $\gamma$  (gain $g=1$)")
    ax.legend(loc="lower right", frameon=False, fontsize=16)
    fig.tight_layout()
    _save(fig, "rate_gamma_curve")
    plt.close(fig)

    # ---- Figure: effect of gain (gamma = 1) ------------------------------- #
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for g_, col in [(0.5, "#7fb3d5"), (1.0, "#000000"), (2.0, "#e08214"),
                    (4.0, "#c0392b")]:
        ls = "--" if g_ == 1.0 else "-"
        ax.plot(x, np.minimum(1.0, g_ * x), color=col, lw=2.8, ls=ls,
                label=rf"$g={g_}$")
    ax.axhline(1.0, color="#999999", ls=":", lw=1.3)
    ax.text(0.28, 1.015, r"saturation ($p=1$)", fontsize=15, color="#555555",
            ha="left", va="bottom")
    _finish(ax, r"Effect of the gain $g$  (exponent $\gamma=1$)")
    ax.legend(loc="lower right", frameon=False, fontsize=16)
    fig.tight_layout()
    _save(fig, "rate_gain_curve")
    plt.close(fig)

    plt.rcParams["mathtext.fontset"] = _prev_fs


# --------------------------------------------------------------------------- #
# Figure 15 — Rate-encoded vorticity field under gamma / gain sweeps
# --------------------------------------------------------------------------- #
def plot_rate_encoding_fields():
    """
    Visualize the rate-encoded spike field of one real snapshot. The signed
    normalized vorticity sets a per-pixel firing probability p = min(1, g|w|^g),
    which is sampled into positive (red) and negative (blue) polarity spike
    channels. Top row sweeps the exponent gamma (g=1); bottom row sweeps the
    gain g (gamma=0.6). Each panel reports the combined firing rate rho_comb.
    """
    rng = np.random.default_rng(1)
    o = load_vorticity(40, "RD6")
    scale = float(np.percentile(np.abs(o[::30]), 99))
    n = np.clip(o[300] / scale, -1.0, 1.0)
    mag = np.abs(n)
    pos, neg = n > 0, n < 0
    c_pos, c_neg = [0.80, 0.12, 0.12], [0.13, 0.33, 0.75]

    def spike_rgb(gm, g_):
        p = np.minimum(1.0, g_ * mag ** gm)
        draw = rng.random(mag.shape)
        sp, sn = (draw < p) & pos, (draw < p) & neg
        rho = (int(sp.sum()) + int(sn.sum())) / mag.size * 100.0
        rgb = np.ones(mag.shape + (3,))
        rgb[sp] = c_pos
        rgb[sn] = c_neg
        return rgb, rho

    def field_ax(ax, title):
        _add_airfoil(ax, 40)
        ax.set_title(title, fontsize=18, linespacing=1.35)
        ax.set_xticks([])
        ax.set_yticks([])

    _prev_fs = plt.rcParams["mathtext.fontset"]
    plt.rcParams["mathtext.fontset"] = "stix"

    fig = plt.figure(figsize=(13.6, 5.0))
    gs = fig.add_gridspec(2, 4, wspace=0.04, hspace=0.42,
                          left=0.055, right=0.995, top=0.9, bottom=0.07)

    # left reference column: input field (top) and probability map (bottom)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(n, origin="lower", extent=DOMAIN, aspect="equal", cmap="RdBu_r",
              vmin=-1, vmax=1, interpolation="bilinear")
    field_ax(ax, r"input $\tilde{\boldsymbol{\Omega}}$")
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(mag, origin="lower", extent=DOMAIN, aspect="equal", cmap="magma",
              vmin=0, vmax=1, interpolation="bilinear")
    field_ax(ax, r"firing prob. $p=|\tilde{\boldsymbol{\Omega}}|$" + "\n"
             + r"($\gamma=1,\ g=1$)")

    # top row: gamma sweep (g = 1)
    for j, gm in enumerate([0.4, 1.0, 2.0]):
        rgb, rho = spike_rgb(gm, 1.0)
        ax = fig.add_subplot(gs[0, j + 1])
        ax.imshow(rgb, origin="lower", extent=DOMAIN, aspect="equal",
                  interpolation="nearest")
        field_ax(ax, rf"$\gamma={gm}$" + "\n"
                 + rf"$\rho_{{\mathrm{{comb}}}}={rho:.0f}\%$")

    # bottom row: gain sweep (gamma = 0.6)
    for j, g_ in enumerate([0.5, 1.0, 2.0]):
        rgb, rho = spike_rgb(0.6, g_)
        ax = fig.add_subplot(gs[1, j + 1])
        ax.imshow(rgb, origin="lower", extent=DOMAIN, aspect="equal",
                  interpolation="nearest")
        field_ax(ax, rf"$g={g_}$" + "\n"
                 + rf"$\rho_{{\mathrm{{comb}}}}={rho:.0f}\%$")

    # row descriptors on the left margin (well separated so they read as two
    # distinct labels, not one sentence)
    fig.text(0.014, 0.71, r"vary $\gamma$  ($g=1$)", rotation=90, ha="center",
             va="center", fontsize=17, color="#333333")
    fig.text(0.014, 0.21, r"vary gain $g$  ($\gamma=0.6$)", rotation=90,
             ha="center", va="center", fontsize=17, color="#333333")
    # polarity legend (colour-matched to the spikes)
    fig.text(0.485, 0.006, r"$S^{r,+}$ (positive vorticity)", ha="right",
             va="bottom", fontsize=16, color="#b01818")
    fig.text(0.515, 0.006, r"$S^{r,-}$ (negative vorticity)", ha="left",
             va="bottom", fontsize=16, color="#1d3fa0")

    _save(fig, "rate_encoding_fields")
    plt.close(fig)
    plt.rcParams["mathtext.fontset"] = _prev_fs


# --------------------------------------------------------------------------- #
# Delta-encoding helpers (shared frame + statistics)
# --------------------------------------------------------------------------- #
_DELTA_CASE = (20, "RD4")          # strong convecting vortex, clear motion
_DELTA_FRAME = 700                 # relaxing wake: localized shed-vortex motion


def _delta_data():
    """Load one case, return (Omega~_{t-1}, Omega~_t, dOmega, mean|dOmega|,
    and the full-sequence |dOmega| sample) for the delta figures."""
    a, c = _DELTA_CASE
    o = load_vorticity(a, c)
    scale = float(np.percentile(np.abs(o[::30]), 99))
    on = np.clip(o / scale, -1.0, 1.0)
    d = np.abs(np.diff(on, axis=0))                     # consecutive-frame change
    mbar = float(d.mean())
    prev, cur = on[_DELTA_FRAME - 1], on[_DELTA_FRAME]
    ad = d[::3].ravel()                                 # subsample frames (gap 1)
    return prev, cur, cur - prev, mbar, ad, a


# --------------------------------------------------------------------------- #
# Figure — Delta-encoding operation (consecutive fields -> difference -> events)
# --------------------------------------------------------------------------- #
def plot_delta_mechanism():
    """
    The delta-encoding operation on one pair of consecutive normalized fields:
    subtract to get dOmega_t, then threshold at +-theta_Delta (f=1.4) to emit
    positive (red) and negative (blue) temporal-change events.
    """
    prev, cur, dom, mbar, _, aoa = _delta_data()
    theta = 1.4 * mbar
    vd = float(np.percentile(np.abs(dom), 99.5))

    sp = dom > theta
    sn = dom < -theta
    rgb = np.ones(dom.shape + (3,))
    rgb[sp] = [0.80, 0.12, 0.12]
    rgb[sn] = [0.13, 0.33, 0.75]

    _prev_fs = plt.rcParams["mathtext.fontset"]
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.7))
    panels = [
        (prev, "RdBu_r", -1, 1, r"$\tilde{\boldsymbol{\Omega}}_{t-1}$"),
        (cur, "RdBu_r", -1, 1, r"$\tilde{\boldsymbol{\Omega}}_{t}$"),
        (dom, "RdBu_r", -vd, vd,
         r"$\Delta\tilde{\boldsymbol{\Omega}}_t="
         r"\tilde{\boldsymbol{\Omega}}_t-\tilde{\boldsymbol{\Omega}}_{t-1}$"),
    ]
    for ax, (img, cmap, vmn, vmx, title) in zip(axes, panels):
        ax.imshow(img, origin="lower", extent=DOMAIN, aspect="equal", cmap=cmap,
                  vmin=vmn, vmax=vmx, interpolation="bilinear")
        _add_airfoil(ax, aoa)
        ax.set_title(title, fontsize=17, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[3].imshow(rgb, origin="lower", extent=DOMAIN, aspect="equal",
                   interpolation="nearest")
    _add_airfoil(axes[3], aoa)
    axes[3].set_title(r"delta events  ($f_\Delta=1.4$)", fontsize=17, pad=8)
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    # polarity legend under the events panel (colour-matched to the spikes)
    fig.text(0.808, 0.02, r"$S_t^{\Delta,+}$: increase", ha="right",
             va="bottom", fontsize=13.5, color="#b01818")
    fig.text(0.822, 0.02, r"$S_t^{\Delta,-}$: decrease", ha="left",
             va="bottom", fontsize=13.5, color="#1d3fa0")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.13,
                        wspace=0.08)
    _save(fig, "delta_mechanism")
    plt.close(fig)
    plt.rcParams["mathtext.fontset"] = _prev_fs


# --------------------------------------------------------------------------- #
# Figure — Effect of the delta threshold factor (distribution + thresholds)
# --------------------------------------------------------------------------- #
def plot_delta_threshold_effect():
    """
    Distribution of the frame-to-frame change magnitude |dOmega| over ALL pixels
    of the sequence (log axis, broken y so the huge static spike and the change
    distribution are both visible). Pixels that fire at f_Delta=1.4 (those with
    |dOmega| > theta_Delta) are shaded darker, so the fired fraction is visibly
    ~8% of the total. Thresholds for other factors are marked for comparison.
    """
    _, _, _, mbar, ad, _ = _delta_data()

    floor = 1e-5                       # below this = static / no meaningful change
    n_static = int((ad < floor).sum())
    change = ad[ad >= floor]
    top = float(ad.max())
    bins = np.logspace(np.log10(floor), np.log10(top), 56)
    counts, edges = np.histogram(change, bins=bins)
    widths = np.diff(edges)

    light, edgec = "#c9d7e8", "#8ba7c8"
    static_frac = n_static / ad.size * 100.0
    sx0, sx1 = 3e-6, 7e-6              # static-spike bar, left of the change dist.

    _prev_fs = plt.rcParams["mathtext.fontset"]
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, (axt, axb) = plt.subplots(
        2, 1, figsize=(11.0, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.6], "hspace": 0.09})
    for ax in (axt, axb):
        ax.bar(edges[:-1], counts, width=widths, align="edge",
               color=light, edgecolor=edgec, linewidth=0.3)
        ax.bar([sx0], [n_static], width=[sx1 - sx0], align="edge",
               color=light, edgecolor=edgec, linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlim(2.3e-6, top * 1.3)

    # broken y-axis: top shows the static spike, bottom the change distribution
    change_max = counts.max()
    axb.set_ylim(0, change_max * 1.22)
    axt.set_ylim(n_static * 0.80, n_static * 1.05)
    axt.spines["bottom"].set_visible(False)
    axb.spines["top"].set_visible(False)
    axt.tick_params(bottom=False, labelbottom=False, labelsize=12)
    axb.tick_params(labelsize=12)
    for ax in (axt, axb):
        ax.spines["right"].set_visible(False)
    axt.spines["top"].set_visible(False)
    dd = 0.015
    kw = dict(color="k", clip_on=False, lw=1.2)
    axt.plot((-dd, dd), (-dd * 2.6, dd * 2.6), transform=axt.transAxes, **kw)
    axb.plot((-dd, dd), (1 - dd, 1 + dd), transform=axb.transAxes, **kw)

    factors = [(0.7, "#2e8b57"), (1.4, "#c0392b"), (2.5, "#7030a0")]
    for ax in (axt, axb):
        for f_, col in factors:
            ax.axvline(f_ * mbar, color=col, ls="--", lw=2.0)

    axt.text(np.sqrt(sx0 * sx1), n_static * 1.0,
             rf"{static_frac:.0f}%" + "\n" + "static", ha="center", va="top",
             fontsize=13, color="#4a6785")

    # threshold-factor legend inside the bottom panel
    for i, (f_, col) in enumerate(factors):
        rho = (ad > f_ * mbar).mean() * 100.0
        axb.text(0.985, 0.95 - 0.10 * i,
                 rf"$f_\Delta={f_}$:  $\theta_\Delta={f_ * mbar:.4f}$,  "
                 rf"$\rho^{{\Delta}}_{{\mathrm{{comb}}}}={rho:.0f}\%$",
                 transform=axb.transAxes, ha="right", va="top", fontsize=14,
                 color=col)

    axb.set_xlabel(r"frame-to-frame change magnitude "
                   r"$|\Delta\tilde{\boldsymbol{\Omega}}_t|$  (log scale)",
                   fontsize=15)
    fig.text(0.032, 0.55, "pixel count", rotation=90, va="center",
             fontsize=15)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.97, bottom=0.10)
    _save(fig, "delta_threshold_effect")
    plt.close(fig)
    plt.rcParams["mathtext.fontset"] = _prev_fs


# --------------------------------------------------------------------------- #
# Figure — Stratified random-case dataset split
# --------------------------------------------------------------------------- #
def plot_random_case_split():
    """
    Schematic of the stratified random-case split built from the real dataset:
    the 31 cases of each angle of attack (30 disturbed + 1 baseline) are grouped
    by gust-intensity band (columns) and angle (rows). Each dot is one real
    simulation case, coloured by its split assignment following the actual
    per-group 70/15/15 holdout; the no-gust baseline of every angle (ringed)
    always stays in training.
    """
    xlsx = os.path.join(os.path.dirname(DATA_ROOT), "RD_list_FT2023.xlsx")
    if not os.path.isfile(xlsx):
        print(f"[SKIP] random_case_split: RD params not found ({xlsx})")
        return

    angles = [20, 30, 40, 50, 60]
    sheet = {20: "a20", 30: "a30", 40: "a40", 50: "a50", 60: "a60"}
    absG = {}
    for a in angles:
        df = pd.read_excel(xlsx, sheet_name=sheet[a], header=None)
        gs = []
        for _, row in df.iterrows():
            try:
                idx = int(row.iloc[0])
                if 1 <= idx <= 30:
                    gs.append(abs(float(row.iloc[1])))
            except (ValueError, TypeError):
                pass
        absG[a] = np.array(gs)
    band_defs = [(0.0, 1.0), (1.0, 2.5), (2.5, 1e9)]

    def split_counts(n):
        """(n_train, n_val, n_test) using the code's per-stratum 70/15/15 rule."""
        if n == 0:
            return 0, 0, 0
        n_test = max(1, int(round(0.15 * n))) if n >= 3 else 0
        n_val = max(1, int(round(0.15 * n))) if (n - n_test) >= 2 else 0
        return n - n_test - n_val, n_val, n_test

    c_train, c_val, c_test = "#4e79a7", "#f2a13b", "#e15759"
    cmap = {"train": c_train, "val": c_val, "test": c_test, "base": c_train}
    band_labels = [r"mild" + "\n" + r"$|G|\leq 1.0$",
                   r"medium" + "\n" + r"$1.0<|G|\leq 2.5$",
                   r"strong" + "\n" + r"$|G|>2.5$"]

    col_x = [3.5, 6.2, 8.9]
    row_y = [8.5, 7.0, 5.5, 4.0, 2.5]
    box_w, box_h = 2.05, 1.24
    ncols, dx, dy, r = 6, 0.31, 0.34, 0.14

    fig, ax = plt.subplots(figsize=(9.8, 10.2))
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.8, 10.9)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(6.2, 10.5, "Stratified random-case split", ha="center",
            va="center", fontsize=22, fontweight="bold", color="#333333")
    for x, bl in zip(col_x, band_labels):
        ax.text(x, 9.78, bl, ha="center", va="center", fontsize=17,
                linespacing=1.35, color="#333333")

    for i, (y, a) in enumerate(zip(row_y, angles)):
        ax.text(1.55, y, rf"$\alpha={a}^\circ$", ha="center", va="center",
                fontsize=18, color="#333333")
        for j, (lo, hi) in enumerate(band_defs):
            x = col_x[j]
            ax.add_patch(FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                facecolor="#f4f4f4", edgecolor="#cccccc", lw=1.0, zorder=1))
            g = absG[a]
            n = int((g <= hi).sum() if lo == 0 else ((g > lo) & (g <= hi)).sum())
            n_train, n_val, n_test = split_counts(n)
            toks = (["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
            if lo == 0:                         # baseline (G=0) lives in mild
                toks.append("base")
            np.random.default_rng(100 * i + j).shuffle(toks)

            m = len(toks)
            rows = int(np.ceil(m / ncols))
            idx = 0
            for rr in range(rows):
                nd = min(ncols, m - rr * ncols)
                for cc in range(nd):
                    tok = toks[idx]
                    idx += 1
                    cx = x + (cc - (nd - 1) / 2) * dx
                    cy = y + ((rows - 1) / 2 - rr) * dy
                    ec = "black" if tok == "base" else "white"
                    lw = 1.9 if tok == "base" else 0.8
                    ax.add_patch(Circle((cx, cy), r, facecolor=cmap[tok],
                                        edgecolor=ec, lw=lw, zorder=3))

    # legend
    ly = 1.35
    items = [(1.9, c_train, "train", "white", 0.8),
             (3.35, c_val, "validation", "white", 0.8),
             (5.5, c_test, "test", "white", 0.8),
             (6.75, c_train, "baseline (always train)", "black", 1.9)]
    for lx, fc, lab, ec, lw in items:
        ax.add_patch(Circle((lx, ly), r, facecolor=fc, edgecolor=ec, lw=lw,
                            zorder=3))
        ax.text(lx + 0.36, ly, lab, ha="left", va="center", fontsize=16,
                color="#333333")

    _save(fig, "random_case_split")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure — Network architecture (spiking Cl encoder)
# --------------------------------------------------------------------------- #
def plot_network_architecture():
    """
    Left-to-right data path of the spiking encoder: 2-channel spike input ->
    two spiking convolutional blocks (with channel expansion and spatial
    pooling) -> flatten -> LIF membrane read -> latent projection -> lift
    readout, applied per timestep with recurrent LIF membrane states.
    """
    # operation colours (each network step is a coloured slab; see legend)
    OP = {"conv": ("Conv 3$\\times$3", "#4e79a7"),
          "gn": ("GroupNorm", "#59a14f"),
          "lif": ("LIF", "#e15759"),
          "pool": ("MaxPool", "#f28e2b"),
          "flat": ("Flatten", "#9c755f"),
          "ln": ("LayerNorm", "#76b7b2")}
    fm_fc, fm_ec = "#c9ddf0", "#33507a"
    c_mem, c_z, ec_o = "#f7d9ae", "#f0bd7e", "#9a6a1f"
    c_cl, ec_cl = "#7fbf7f", "#3a7a3a"
    reg_ce, reg_ls, reg_hd = "#e9f1fa", "#eef7ee", "#f7edf3"

    def _lite(hx, f=0.5):
        r, g, b = (int(hx[i:i + 2], 16) for i in (1, 3, 5))
        return "#%02x%02x%02x" % (int(r + (255 - r) * f), int(g + (255 - g) * f),
                                  int(b + (255 - b) * f))

    cy = 5.2
    op_w, op_gap, op_h, gap = 0.24, 0.035, 2.05, 0.36
    gap_r, gap_h = 1.15, 1.4         # wider gaps around Latent Space / Cl Head

    fig, ax = plt.subplots(figsize=(13.8, 6.35))
    ax.set_xlim(0, 17.8)
    ax.set_ylim(1.4, 9.4)
    ax.set_aspect("equal")
    ax.axis("off")

    def stack(cx, w, h, n, ox, oy):
        """A multi-channel feature map drawn as a deck of offset planes."""
        fw = w + (n - 1) * ox
        bx, by = cx - fw / 2, cy - (h + (n - 1) * oy) / 2
        for i in range(n - 1, -1, -1):
            ax.add_patch(Rectangle((bx + i * ox, by + i * oy), w, h,
                                   facecolor=fm_fc if i == 0 else _lite(fm_fc),
                                   edgecolor=fm_ec, lw=1.0, zorder=10 + (n - i)))
        return fw

    def vbar(cx, w, h, fc, ncell):
        x0, y0 = cx - w / 2, cy - h / 2
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=fc, edgecolor=ec_o,
                               lw=1.4, zorder=12))
        for k in range(1, ncell):
            ax.plot([x0, x0 + w], [y0 + h * k / ncell] * 2, color=ec_o,
                    lw=0.5, zorder=13)

    def ops(x0, seq):
        for k, o in enumerate(seq):
            xx = x0 + k * (op_w + op_gap)
            ax.add_patch(FancyBboxPatch((xx, cy - op_h / 2), op_w, op_h,
                         boxstyle="round,pad=0.005,rounding_size=0.04",
                         facecolor=OP[o][1], edgecolor="#333333", lw=0.7,
                         zorder=11))
        return x0 + len(seq) * (op_w + op_gap) - op_gap

    def flow_arrow(x0, x1):
        _arrow(ax, (x0, cy), (x1, cy), "#555555", lw=1.6, mut=12)

    def shape(cx, txt):
        ax.text(cx, cy - 1.75, txt, ha="center", va="top", fontsize=14.5,
                linespacing=1.25)

    conv_block = ["conv", "gn", "lif", "conv", "gn", "lif", "pool"]

    # ------------------------------------------------------------------ #
    # Lay the flow out left to right, recording key x-positions.
    # ------------------------------------------------------------------ #
    x = 0.7
    fw = stack(x + (1.4 + 1 * 0.09) / 2, 1.4, 1.0, 2, 0.09, 0.18)
    x_in = x + fw / 2
    shape(x_in, r"$\mathbf{S}_t$" + "\n" r"$2\times120\times240$")
    x += fw
    flow_arrow(x + 0.05, x + gap - 0.05); x += gap
    x = ops(x, conv_block) + gap
    flow_arrow(x - gap + 0.05, x - 0.05)
    fw = stack(x + (1.0 + 5 * 0.075) / 2, 1.0, 0.72, 6, 0.075, 0.15)
    x_f1 = x + fw / 2
    shape(x_f1, r"$16\times30\times60$")
    x += fw
    flow_arrow(x + 0.05, x + gap - 0.05); x += gap
    x = ops(x, conv_block) + gap
    flow_arrow(x - gap + 0.05, x - 0.05)
    fw = stack(x + (0.62 + 7 * 0.055) / 2, 0.62, 0.46, 8, 0.055, 0.11)
    x_f2 = x + fw / 2
    shape(x_f2, r"$32\times6\times12$")
    x += fw
    flow_arrow(x + 0.05, x + gap - 0.05); x += gap
    x = ops(x, ["flat", "ln", "lif"]) + gap
    x_dom = x - gap / 2                          # spike -> continuous boundary
    flow_arrow(x - gap + 0.05, x - 0.05)
    vbar(x + 0.23, 0.46, 2.55, c_mem, 14)
    x_mem = x + 0.23
    shape(x_mem, "membrane\n2304")
    x += 0.46
    x_ce_end = x + gap_r / 2                    # Conv Encoder ends after membrane
    # Single arrow membrane -> LayerNorm; the linear projection from the 2304-d
    # membrane vector to R^16 is implied by the dimension change, not drawn.
    flow_arrow(x + 0.08, x + gap_r - 0.08); x += gap_r
    x = ops(x, ["ln"]) + gap
    flow_arrow(x - gap + 0.05, x - 0.05)
    vbar(x + 0.23, 0.46, 1.15, c_z, 8)
    x_z = x + 0.23
    shape(x_z, r"latent $\mathbf{z}\in\mathbb{R}^{16}$")
    x += 0.46
    x_ls_end = x + gap_h / 2 + 0.45              # Latent Space ends here
                                                 # (shifted right so the latent
                                                 #  label stays inside the region)
    # Single arrow straight from the latent vector to the Cl scalar: the linear
    # projection R^16 -> scalar is implied by the dimension change (no block).
    cl_gap = gap_h + op_w + gap                  # preserve the z -> Cl spacing
    flow_arrow(x + 0.08, x + cl_gap - 0.08); x += cl_gap
    ax.add_patch(Circle((x + 0.40, cy), 0.40, facecolor=c_cl,
                        edgecolor=ec_cl, lw=1.8, zorder=12))
    x_cl = x + 0.40
    ax.text(x_cl, cy - 0.62, r"$\hat{C}_l^{\,t}$", ha="center", va="top",
            fontsize=20)
    x_right = x_cl + 0.62

    # ------------------------------------------------------------------ #
    # Functional-region backgrounds + headers (drawn behind, then labels)
    # ------------------------------------------------------------------ #
    ry0, ry1 = 2.5, 8.35
    for (rx0, rx1, col, name) in [
            (0.25, x_ce_end, reg_ce, "Convolutional Encoder"),
            (x_ce_end, x_ls_end, reg_ls, "Latent Space"),
            (x_ls_end, x_right + 0.1, reg_hd, r"$C_l$ Head")]:
        ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                               facecolor=col, edgecolor="none", zorder=0))
        ax.text(0.5 * (rx0 + rx1), 9.0, name, ha="center", va="center",
                fontsize=16, fontweight="bold", color="#333333")

    # ------------------------------------------------------------------ #
    # Domain indicator bar (binary spikes vs continuous membrane)
    # ------------------------------------------------------------------ #
    dby = 7.85
    ax.add_patch(Rectangle((0.25, dby), x_dom - 0.25, 0.42, facecolor="#cfe0f2",
                           edgecolor="#7f9fc0", lw=0.8, zorder=1))
    ax.add_patch(Rectangle((x_dom, dby), x_right + 0.1 - x_dom, 0.42,
                           facecolor="#f6e0c0", edgecolor="#c9a06a", lw=0.8,
                           zorder=1))
    ax.text(0.5 * (0.25 + x_dom), dby + 0.21, "binary spike domain",
            ha="center", va="center", fontsize=14, color="#33507a")
    ax.text(0.5 * (x_dom + x_right + 0.1), dby + 0.21, "continuous (membrane)",
            ha="center", va="center", fontsize=14, color="#9a6a1f")

    # ------------------------------------------------------------------ #
    # Operation legend
    # ------------------------------------------------------------------ #
    leg = ["conv", "gn", "lif", "pool", "flat", "ln"]
    ly, x0, slot = 2.05, 0.4, 2.55           # fixed, equal spacing per item
    for i, o in enumerate(leg):
        name, col = OP[o]
        lx = x0 + i * slot
        ax.add_patch(FancyBboxPatch((lx, ly - 0.20), 0.38, 0.40,
                     boxstyle="round,pad=0.005,rounding_size=0.06",
                     facecolor=col, edgecolor="#333333", lw=0.8, zorder=3))
        ax.text(lx + 0.50, ly, name, ha="left", va="center", fontsize=14,
                color="#333333")

    _save(fig, "network_architecture")
    plt.close(fig)


def plot_parameter_matched_architecture():
    """
    Left-to-right data path of the parameter-matched non-spiking CNN: a single
    continuous vorticity frame -> two convolutional blocks (Conv+GN+ReLU x2 then
    MaxPool) -> flatten -> LayerNorm+ReLU -> latent projection -> lift readout.
    Same spatial structure and parameter budget as the reference SNN, but every
    LIF neuron is replaced by a ReLU, the input is one signed continuous channel
    (not two spike channels), and each frame is processed independently — no
    membrane state, no recurrence, no spike/continuous domain split.
    """
    OP = {"conv": ("Conv 3$\\times$3", "#4e79a7"),
          "gn": ("GroupNorm", "#59a14f"),
          "relu": ("ReLU", "#e15759"),
          "pool": ("MaxPool", "#f28e2b"),
          "flat": ("Flatten", "#9c755f"),
          "ln": ("LayerNorm", "#76b7b2")}
    fm_fc, fm_ec = "#c9ddf0", "#33507a"          # conv feature-map planes (blue)
    in_fc, in_ec = "#f6d9b8", "#c9a06a"          # continuous input field (peach)
    c_feat, c_z, ec_o = "#f7d9ae", "#f0bd7e", "#9a6a1f"
    c_cl, ec_cl = "#7fbf7f", "#3a7a3a"
    reg_ce, reg_ls, reg_hd = "#e9f1fa", "#eef7ee", "#f7edf3"

    def _lite(hx, f=0.5):
        r, g, b = (int(hx[i:i + 2], 16) for i in (1, 3, 5))
        return "#%02x%02x%02x" % (int(r + (255 - r) * f), int(g + (255 - g) * f),
                                  int(b + (255 - b) * f))

    cy = 5.2
    op_w, op_gap, op_h, gap = 0.24, 0.035, 2.05, 0.36
    gap_r, gap_h = 1.15, 1.4

    fig, ax = plt.subplots(figsize=(13.8, 6.35))
    ax.set_xlim(0, 17.8)
    ax.set_ylim(1.4, 9.4)
    ax.set_aspect("equal")
    ax.axis("off")

    def plane(cx, w, h, fc, ec):
        """A single feature map / field drawn as one plane."""
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                               facecolor=fc, edgecolor=ec, lw=1.1, zorder=12))
        return w

    def stack(cx, w, h, n, ox, oy):
        fw = w + (n - 1) * ox
        bx, by = cx - fw / 2, cy - (h + (n - 1) * oy) / 2
        for i in range(n - 1, -1, -1):
            ax.add_patch(Rectangle((bx + i * ox, by + i * oy), w, h,
                                   facecolor=fm_fc if i == 0 else _lite(fm_fc),
                                   edgecolor=fm_ec, lw=1.0, zorder=10 + (n - i)))
        return fw

    def vbar(cx, w, h, fc, ncell):
        x0, y0 = cx - w / 2, cy - h / 2
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=fc, edgecolor=ec_o,
                               lw=1.4, zorder=12))
        for k in range(1, ncell):
            ax.plot([x0, x0 + w], [y0 + h * k / ncell] * 2, color=ec_o,
                    lw=0.5, zorder=13)

    def ops(x0, seq):
        for k, o in enumerate(seq):
            xx = x0 + k * (op_w + op_gap)
            ax.add_patch(FancyBboxPatch((xx, cy - op_h / 2), op_w, op_h,
                         boxstyle="round,pad=0.005,rounding_size=0.04",
                         facecolor=OP[o][1], edgecolor="#333333", lw=0.7,
                         zorder=11))
        return x0 + len(seq) * (op_w + op_gap) - op_gap

    def flow_arrow(x0, x1):
        _arrow(ax, (x0, cy), (x1, cy), "#555555", lw=1.6, mut=12)

    def shape(cx, txt):
        ax.text(cx, cy - 1.75, txt, ha="center", va="top", fontsize=14.5,
                linespacing=1.25)

    conv_block = ["conv", "gn", "relu", "conv", "gn", "relu", "pool"]

    # ------------------------------------------------------------------ #
    # Lay the flow out left to right.
    # ------------------------------------------------------------------ #
    x = 0.7
    fw = plane(x + 0.7, 1.4, 1.0, in_fc, in_ec)   # single continuous input plane
    x_in = x + fw / 2
    shape(x_in, r"$\mathbf{\Omega}_t$" + "\n" r"$1\times120\times240$")
    x += fw
    flow_arrow(x + 0.05, x + gap - 0.05); x += gap
    x = ops(x, conv_block) + gap
    flow_arrow(x - gap + 0.05, x - 0.05)
    fw = stack(x + (1.0 + 5 * 0.075) / 2, 1.0, 0.72, 6, 0.075, 0.15)
    x_f1 = x + fw / 2
    shape(x_f1, r"$16\times30\times60$")
    x += fw
    flow_arrow(x + 0.05, x + gap - 0.05); x += gap
    x = ops(x, conv_block) + gap
    flow_arrow(x - gap + 0.05, x - 0.05)
    fw = stack(x + (0.62 + 7 * 0.055) / 2, 0.62, 0.46, 8, 0.055, 0.11)
    x_f2 = x + fw / 2
    shape(x_f2, r"$32\times6\times12$")
    x += fw
    flow_arrow(x + 0.05, x + gap - 0.05); x += gap
    x = ops(x, ["flat", "ln", "relu"]) + gap
    flow_arrow(x - gap + 0.05, x - 0.05)
    vbar(x + 0.23, 0.46, 2.55, c_feat, 14)
    x_feat = x + 0.23
    shape(x_feat, "features\n2304")
    x += 0.46
    x_ce_end = x + gap_r / 2
    # Single arrow features -> LayerNorm; the linear projection from the flattened
    # feature vector to R^16 is implied by the dimension change, not drawn.
    flow_arrow(x + 0.08, x + gap_r - 0.08); x += gap_r
    x = ops(x, ["ln", "relu"]) + gap
    flow_arrow(x - gap + 0.05, x - 0.05)
    vbar(x + 0.23, 0.46, 1.15, c_z, 8)
    x_z = x + 0.23
    shape(x_z, r"latent $\mathbf{z}\in\mathbb{R}^{16}$")
    x += 0.46
    x_ls_end = x + gap_h / 2 + 0.45
    # Single arrow straight from the latent vector to the Cl scalar: the linear
    # projection R^16 -> scalar is implied by the dimension change (no block).
    cl_gap = gap_h + op_w + gap                 # preserve the z -> Cl spacing
    flow_arrow(x + 0.08, x + cl_gap - 0.08); x += cl_gap
    ax.add_patch(Circle((x + 0.40, cy), 0.40, facecolor=c_cl,
                        edgecolor=ec_cl, lw=1.8, zorder=12))
    x_cl = x + 0.40
    ax.text(x_cl, cy - 0.62, r"$\hat{C}_l^{\,t}$", ha="center", va="top",
            fontsize=20)
    x_right = x_cl + 0.62

    # ------------------------------------------------------------------ #
    # Functional-region backgrounds + headers.
    # ------------------------------------------------------------------ #
    ry0, ry1 = 2.5, 8.35
    for (rx0, rx1, col, name) in [
            (0.25, x_ce_end, reg_ce, "Convolutional Encoder"),
            (x_ce_end, x_ls_end, reg_ls, "Latent Space"),
            (x_ls_end, x_right + 0.1, reg_hd, r"$C_l$ Head")]:
        ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                               facecolor=col, edgecolor="none", zorder=0))
        ax.text(0.5 * (rx0 + rx1), 9.0, name, ha="center", va="center",
                fontsize=16, fontweight="bold", color="#333333")

    # ------------------------------------------------------------------ #
    # Continuous-domain / per-frame note (replaces the SNN's spike-domain bar
    # and recurrent-state arrow — this network has neither).
    # ------------------------------------------------------------------ #
    dby = 7.85
    ax.add_patch(Rectangle((0.25, dby), x_right + 0.1 - 0.25, 0.42,
                           facecolor="#f6e0c0", edgecolor="#c9a06a", lw=0.8,
                           zorder=1))
    ax.text(0.5 * (0.25 + x_right + 0.1), dby + 0.21,
            "fully continuous, per-frame processing",
            ha="center", va="center", fontsize=13.5, color="#9a6a1f")

    # ------------------------------------------------------------------ #
    # Operation legend.
    # ------------------------------------------------------------------ #
    leg = ["conv", "gn", "relu", "pool", "flat", "ln"]
    ly, x0, slot = 2.05, 0.4, 2.55
    for i, o in enumerate(leg):
        name, col = OP[o]
        lx = x0 + i * slot
        ax.add_patch(FancyBboxPatch((lx, ly - 0.20), 0.38, 0.40,
                     boxstyle="round,pad=0.005,rounding_size=0.06",
                     facecolor=col, edgecolor="#333333", lw=0.8, zorder=3))
        ax.text(lx + 0.50, ly, name, ha="left", va="center", fontsize=14,
                color="#333333")

    _save(fig, "parameter_matched_architecture")
    plt.close(fig)


def plot_energy_matched_architecture():
    """
    Left-to-right data path of the energy-matched non-spiking CNN. Structurally
    identical to the parameter-matched CNN (single continuous input frame,
    Conv+GN+ReLU x2 + MaxPool per block, flatten -> LayerNorm+ReLU -> latent ->
    lift), but the convolutional block widths are cut from 16/32 to 4/4 so the
    dense per-step MAC energy approaches the reference SNN's budget. This shrinks
    the flattened feature dimension from 2304 to 288 and the parameter count from
    ~58k to ~5.8k, while the latent dimension (16) and the overall structure are
    unchanged.
    """
    OP = {"conv": ("Conv 3$\\times$3", "#4e79a7"),
          "gn": ("GroupNorm", "#59a14f"),
          "relu": ("ReLU", "#e15759"),
          "pool": ("MaxPool", "#f28e2b"),
          "flat": ("Flatten", "#9c755f"),
          "ln": ("LayerNorm", "#76b7b2")}
    fm_fc, fm_ec = "#c9ddf0", "#33507a"
    in_fc, in_ec = "#f6d9b8", "#c9a06a"
    c_feat, c_z, ec_o = "#f7d9ae", "#f0bd7e", "#9a6a1f"
    c_cl, ec_cl = "#7fbf7f", "#3a7a3a"
    reg_ce, reg_ls, reg_hd = "#e9f1fa", "#eef7ee", "#f7edf3"

    def _lite(hx, f=0.5):
        r, g, b = (int(hx[i:i + 2], 16) for i in (1, 3, 5))
        return "#%02x%02x%02x" % (int(r + (255 - r) * f), int(g + (255 - g) * f),
                                  int(b + (255 - b) * f))

    cy = 5.2
    op_w, op_gap, op_h, gap = 0.24, 0.035, 2.05, 0.36
    gap_r, gap_h = 1.15, 1.4

    fig, ax = plt.subplots(figsize=(13.8, 6.35))
    ax.set_xlim(0, 17.8)
    ax.set_ylim(1.4, 9.4)
    ax.set_aspect("equal")
    ax.axis("off")

    def plane(cx, w, h, fc, ec):
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                               facecolor=fc, edgecolor=ec, lw=1.1, zorder=12))
        return w

    def stack(cx, w, h, n, ox, oy):
        fw = w + (n - 1) * ox
        bx, by = cx - fw / 2, cy - (h + (n - 1) * oy) / 2
        for i in range(n - 1, -1, -1):
            ax.add_patch(Rectangle((bx + i * ox, by + i * oy), w, h,
                                   facecolor=fm_fc if i == 0 else _lite(fm_fc),
                                   edgecolor=fm_ec, lw=1.0, zorder=10 + (n - i)))
        return fw

    def vbar(cx, w, h, fc, ncell):
        x0, y0 = cx - w / 2, cy - h / 2
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=fc, edgecolor=ec_o,
                               lw=1.4, zorder=12))
        for k in range(1, ncell):
            ax.plot([x0, x0 + w], [y0 + h * k / ncell] * 2, color=ec_o,
                    lw=0.5, zorder=13)

    def ops(x0, seq):
        for k, o in enumerate(seq):
            xx = x0 + k * (op_w + op_gap)
            ax.add_patch(FancyBboxPatch((xx, cy - op_h / 2), op_w, op_h,
                         boxstyle="round,pad=0.005,rounding_size=0.04",
                         facecolor=OP[o][1], edgecolor="#333333", lw=0.7,
                         zorder=11))
        return x0 + len(seq) * (op_w + op_gap) - op_gap

    def flow_arrow(x0, x1):
        _arrow(ax, (x0, cy), (x1, cy), "#555555", lw=1.6, mut=12)

    def shape(cx, txt):
        ax.text(cx, cy - 1.75, txt, ha="center", va="top", fontsize=14.5,
                linespacing=1.25)

    conv_block = ["conv", "gn", "relu", "conv", "gn", "relu", "pool"]

    # ------------------------------------------------------------------ #
    # Lay the flow out left to right. Feature-map decks are drawn THIN (few
    # planes) to convey the narrowed 4-channel blocks.
    # ------------------------------------------------------------------ #
    x = 0.7
    fw = plane(x + 0.7, 1.4, 1.0, in_fc, in_ec)
    x_in = x + fw / 2
    shape(x_in, r"$\mathbf{\Omega}_t$" + "\n" r"$1\times120\times240$")
    x += fw
    flow_arrow(x + 0.05, x + gap - 0.05); x += gap
    x = ops(x, conv_block) + gap
    flow_arrow(x - gap + 0.05, x - 0.05)
    fw = stack(x + (1.0 + 2 * 0.075) / 2, 1.0, 0.72, 3, 0.075, 0.15)
    x_f1 = x + fw / 2
    shape(x_f1, r"$4\times30\times60$")
    x += fw
    flow_arrow(x + 0.05, x + gap - 0.05); x += gap
    x = ops(x, conv_block) + gap
    flow_arrow(x - gap + 0.05, x - 0.05)
    fw = stack(x + (0.62 + 2 * 0.055) / 2, 0.62, 0.46, 3, 0.055, 0.11)
    x_f2 = x + fw / 2
    shape(x_f2, r"$4\times6\times12$")
    x += fw
    flow_arrow(x + 0.05, x + gap - 0.05); x += gap
    x = ops(x, ["flat", "ln", "relu"]) + gap
    flow_arrow(x - gap + 0.05, x - 0.05)
    vbar(x + 0.23, 0.46, 1.65, c_feat, 9)
    x_feat = x + 0.23
    shape(x_feat, "features\n288")
    x += 0.46
    x_ce_end = x + gap_r / 2
    # Single arrow features -> LayerNorm; the linear projection from the flattened
    # feature vector to R^16 is implied by the dimension change, not drawn.
    flow_arrow(x + 0.08, x + gap_r - 0.08); x += gap_r
    x = ops(x, ["ln", "relu"]) + gap
    flow_arrow(x - gap + 0.05, x - 0.05)
    vbar(x + 0.23, 0.46, 1.15, c_z, 8)
    x_z = x + 0.23
    shape(x_z, r"latent $\mathbf{z}\in\mathbb{R}^{16}$")
    x += 0.46
    x_ls_end = x + gap_h / 2 + 0.45
    # Single arrow straight from the latent vector to the Cl scalar: the linear
    # projection R^16 -> scalar is implied by the dimension change (no block).
    cl_gap = gap_h + op_w + gap                 # preserve the z -> Cl spacing
    flow_arrow(x + 0.08, x + cl_gap - 0.08); x += cl_gap
    ax.add_patch(Circle((x + 0.40, cy), 0.40, facecolor=c_cl,
                        edgecolor=ec_cl, lw=1.8, zorder=12))
    x_cl = x + 0.40
    ax.text(x_cl, cy - 0.62, r"$\hat{C}_l^{\,t}$", ha="center", va="top",
            fontsize=20)
    x_right = x_cl + 0.62

    # ------------------------------------------------------------------ #
    # Functional-region backgrounds + headers.
    # ------------------------------------------------------------------ #
    ry0, ry1 = 2.5, 8.35
    for (rx0, rx1, col, name) in [
            (0.25, x_ce_end, reg_ce, "Convolutional Encoder"),
            (x_ce_end, x_ls_end, reg_ls, "Latent Space"),
            (x_ls_end, x_right + 0.1, reg_hd, r"$C_l$ Head")]:
        ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                               facecolor=col, edgecolor="none", zorder=0))
        ax.text(0.5 * (rx0 + rx1), 9.0, name, ha="center", va="center",
                fontsize=16, fontweight="bold", color="#333333")

    # ------------------------------------------------------------------ #
    # Continuous / per-frame + energy-matching note.
    # ------------------------------------------------------------------ #
    dby = 7.85
    ax.add_patch(Rectangle((0.25, dby), x_right + 0.1 - 0.25, 0.42,
                           facecolor="#f6e0c0", edgecolor="#c9a06a", lw=0.8,
                           zorder=1))
    ax.text(0.5 * (0.25 + x_right + 0.1), dby + 0.21,
            "fully continuous, per-frame processing",
            ha="center", va="center", fontsize=13.5, color="#9a6a1f")

    # ------------------------------------------------------------------ #
    # Operation legend.
    # ------------------------------------------------------------------ #
    leg = ["conv", "gn", "relu", "pool", "flat", "ln"]
    ly, x0, slot = 2.05, 0.4, 2.55
    for i, o in enumerate(leg):
        name, col = OP[o]
        lx = x0 + i * slot
        ax.add_patch(FancyBboxPatch((lx, ly - 0.20), 0.38, 0.40,
                     boxstyle="round,pad=0.005,rounding_size=0.06",
                     facecolor=col, edgecolor="#333333", lw=0.8, zorder=3))
        ax.text(lx + 0.50, ly, name, ha="left", va="center", fontsize=14,
                color="#333333")

    _save(fig, "energy_matched_architecture")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure — Dataset scatter (G, R, y) coloured by split, shaped by angle
# --------------------------------------------------------------------------- #
# Test cases of the current baseline (mirrors TEST_CASES in train_SNN.py). Every
# other case is training (the baseline merges the validation set into training).
_SCATTER_TEST_CASES = {
    "AoA20_RD5_G_n2.2", "AoA20_RD8_G_0.8", "AoA20_RD17_G_2.8", "AoA20_RD29_G_1.2",
    "AoA30_RD2_G_3.4", "AoA30_RD4_G_n1.2", "AoA30_RD11_G_1.2", "AoA30_RD13_G_n3.6",
    "AoA30_RD22_G_n0.4",
    "AoA40_RD1_G_n2.2", "AoA40_RD6_G_1.8", "AoA40_RD12_G_n2.0", "AoA40_RD16_G_4.0",
    "AoA40_RD19_G_0.8",
    "AoA50_RD4_G_n3.0", "AoA50_RD7_G_0.4", "AoA50_RD8_G_n2.0", "AoA50_RD18_G_2.2",
    "AoA50_RD21_G_n4.0",
    "AoA60_RD6_G_n3.0", "AoA60_RD9_G_n3.0", "AoA60_RD10_G_n0.4", "AoA60_RD24_G_n1.8",
    "AoA60_RD26_G_2.0",
}

# Marker per angle of attack.
_SCATTER_ANGLE_MARKER = {20: "o", 30: "s", 40: "^", 50: "D", 60: "v"}


def _scatter_case_tag(angle, case_id, g_val):
    """Reproduce the canonical case tag, e.g. 'AoA20_RD5_G_n2.2'."""
    g_str = f"{abs(g_val):.1f}" if g_val >= 0 else f"n{abs(g_val):.1f}"
    return f"AoA{angle}_{case_id}_G_{g_str}"


def plot_dataset_scatter():
    """
    3D scatter of every dataset case in the (G, R, y) parameter space, where G
    is the gust ratio, R the vortex core radius and y the vertical position.
    Points are blue for the training set and red for the test set (with case
    counts in the legend); the marker shape encodes the angle of attack.
    """
    xlsx = os.path.join(os.path.dirname(DATA_ROOT), "RD_list_FT2023.xlsx")
    if not os.path.isfile(xlsx):
        print(f"[SKIP] dataset_scatter: RD params not found ({xlsx})")
        return

    angles = [20, 30, 40, 50, 60]
    sheet = {20: "a20", 30: "a30", 40: "a40", 50: "a50", 60: "a60"}

    # points[angle] = list of (is_test, G, R, y); base case is G=R=y=0.
    points = {a: [(False, 0.0, 0.0, 0.0)] for a in angles}
    for a in angles:
        df = pd.read_excel(xlsx, sheet_name=sheet[a], header=None)
        for _, row in df.iterrows():
            try:
                idx = int(row.iloc[0])
                G, R, y = float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3])
            except (ValueError, TypeError):
                continue
            if not 1 <= idx <= 30:
                continue
            tag = _scatter_case_tag(a, f"RD{idx}", G)
            points[a].append((tag in _SCATTER_TEST_CASES, G, R, y))

    n_train = sum(1 for a in angles for t, *_ in points[a] if not t)
    n_test = sum(1 for a in angles for t, *_ in points[a] if t)

    train_color, test_color = SPIKE_BLUE, C_RESET

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    for is_test in (False, True):
        color = test_color if is_test else train_color
        for a in angles:
            xs = [G for t, G, R, y in points[a] if t == is_test]
            ys = [R for t, G, R, y in points[a] if t == is_test]
            zs = [y for t, G, R, y in points[a] if t == is_test]
            if not xs:
                continue
            ax.scatter(xs, ys, zs, marker=_SCATTER_ANGLE_MARKER[a], c=color,
                       s=42, edgecolors="white", linewidths=0.4, depthshade=False)

    ax.set_xlabel(r"$G$", labelpad=8)
    ax.set_ylabel(r"$R$", labelpad=8)
    # Keep the z-label upright; padded and pulled inside the right margin so the
    # global tight bbox does not clip it (a known matplotlib 3D quirk).
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$y$", labelpad=10, rotation=0)
    ax.grid(True)

    # Colour legend: train / test with case counts (no legend title).
    split_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=train_color, markeredgecolor="white",
                   label=f"train ({n_train})"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=test_color, markeredgecolor="white",
                   label=f"test ({n_test})"),
    ]
    # Shape legend: angle of attack in neutral grey (no legend title).
    angle_handles = [
        plt.Line2D([0], [0], marker=_SCATTER_ANGLE_MARKER[a], linestyle="none",
                   markersize=8, markerfacecolor="0.35", markeredgecolor="white",
                   label=rf"$\alpha = {a}^\circ$")
        for a in angles
    ]

    leg1 = ax.legend(handles=split_handles, loc="upper left",
                     bbox_to_anchor=(0.0, 1.0), framealpha=0.9)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=angle_handles, loc="upper right",
                     bbox_to_anchor=(1.0, 1.0), framealpha=0.9)

    # Enlarge the 3D axes within the figure, then let the tight-bbox crop trim
    # the border. The z-axis "$y$" label is passed as an extra artist so the
    # tight crop keeps it (the 3D tight bbox omits it — a known matplotlib quirk).
    fig.subplots_adjust(left=0.0, right=0.98, bottom=0.0, top=1.0)
    _save(fig, "dataset_scatter",
          bbox_extra_artists=[ax.zaxis.label, leg1, leg2])
    plt.close(fig)


if __name__ == "__main__":
    plot_dirac_delta()
    plot_spike_train()
    plot_lif_neuron()
    plot_rate_encoding()
    plot_delta_encoding()
    plot_latency_encoding()
    plot_non_differentiability()
    plot_surrogate_gradient()
    plot_recurrence_unrolling()
    plot_bptt_gradient_flow()

    # Chapter 5 (Rate encoding) transfer-function figure — analytic, no data.
    plot_rate_transfer_curves()

    # Dataset-split schematic — needs the RD params xlsx (self-skips if absent).
    plot_random_case_split()

    # Dataset scatter in (G, R, y) space — needs the RD params xlsx (self-skips).
    plot_dataset_scatter()

    # Network architecture — analytic schematic, no data.
    plot_network_architecture()
    plot_parameter_matched_architecture()
    plot_energy_matched_architecture()

    # Chapter 4/5 figures — require the raw simulation data on disk.
    if os.path.isdir(DATA_ROOT):
        plot_setup_schematic()
        plot_encoding_pipeline()
        plot_normalization_choice()
        plot_rate_encoding_fields()
        plot_delta_mechanism()
        plot_delta_threshold_effect()
        for spec in DATASET_CASES:
            plot_dataset_case(spec)
        plot_dataset_case_reconstructed(RECON_CASE)
    else:
        print(f"[SKIP] Dataset figures: data root not found ({DATA_ROOT})")

    print(f"\nFigures written to: {os.path.abspath(OUT_DIR)}")
