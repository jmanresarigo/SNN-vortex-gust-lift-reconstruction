"""
loss.py — Loss function for the spiking Cl encoder.

The model is decoder-free: it predicts the lift coefficient Cl directly from the
spike-encoded vorticity field. The training objective is the mean squared error
(MSE) between the reconstructed Cl and the ground-truth Cl.

Note on nomenclature: throughout this thesis "MSE loss" refers to the mean
squared difference between the reconstructed Cl and the ground-truth Cl — NOT a
reconstruction loss on the vorticity field.
"""

import torch.nn as nn


def mse_loss(cl_pred, cl_true):
    """
    MSE between the reconstructed Cl and the ground-truth Cl.

    Args:
        cl_pred: (T, B, 1) Cl predictions from the spiking encoder
        cl_true: (T, B, 1) ground-truth Cl values
    Returns:
        scalar loss
    """
    return nn.functional.mse_loss(cl_pred, cl_true)


def combined_loss(cl_pred, cl_true, cfg=None):
    """
    Total training loss = MSE(Cl_pred, Cl_true).

    Args:
        cl_pred: (T, B, 1) Cl predictions
        cl_true: (T, B, 1) ground-truth Cl values
        cfg:     kept for interface compatibility; unused.

    Returns:
        total_loss:      scalar tensor (for backprop)
        loss_components: dict with the individual loss values (for logging)
    """
    loss = mse_loss(cl_pred, cl_true)
    return loss, {"MSE": loss.item()}
