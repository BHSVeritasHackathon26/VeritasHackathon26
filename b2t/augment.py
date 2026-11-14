"""Data augmentation: small random changes applied to each training batch so the
model cannot memorise exact recordings and copes better with day-to-day drift.
Used only in training, never when scoring. Settings follow the Brain-to-Text '25
baseline.
"""

import random

import torch


def augment(x, lengths, white_noise_std=1.0, offset_std=0.2, max_cut=3):
    # White noise: a different random nudge for every value.
    x = x + torch.randn_like(x) * white_noise_std
    # Constant offset: one random shift per feature per trial, held for the whole
    # trial. This imitates an electrode's baseline drifting between days.
    x = x + torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * offset_std
    # Random cut: drop up to 3 time steps from the start, so the windows the model
    # sees do not always line up the same way.
    cut = random.randint(0, max_cut)
    return x[:, cut:], lengths - cut
