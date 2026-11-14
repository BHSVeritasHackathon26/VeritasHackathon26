"""Models. Every model takes neural features and returns, for each output time step,
a score for each of the 41 phoneme classes.

All models share the same Frontend, so the only difference between the simple
baseline and the real model is what sits between the frontend and the output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Frontend(nn.Module):
    """Smooths the signal over time, then cuts it into overlapping windows.

    With the defaults and 20 ms bins, each window covers 14 bins (280 ms) and a new
    window starts every 4 bins (80 ms). A window is flattened into one long vector,
    so the model sees a short stretch of activity at a time, not a single instant.
    These settings follow the Brain-to-Text '25 baseline.
    """

    def __init__(self, smooth_std=2.0, patch_size=14, patch_stride=4):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        # Bell-curve weights for averaging each time step with its neighbours.
        radius = int(4 * smooth_std)
        t = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (t / smooth_std) ** 2)
        self.register_buffer("kernel", (kernel / kernel.sum()).view(1, 1, -1))

    def forward(self, x, lengths):
        batch, time, n_features = x.shape
        # Smooth each feature along time.
        x = x.transpose(1, 2).reshape(batch * n_features, 1, time)
        x = F.conv1d(x, self.kernel, padding=self.kernel.shape[-1] // 2)
        x = x.reshape(batch, n_features, time).transpose(1, 2)
        # Cut into windows: (batch, n_windows, patch_size * n_features).
        x = x.unfold(1, self.patch_size, self.patch_stride)   # (batch, n, features, patch)
        x = x.permute(0, 1, 3, 2).flatten(2)
        out_lengths = (lengths - self.patch_size) // self.patch_stride + 1
        return x, out_lengths


class LinearBaseline(nn.Module):
    """The simple baseline: one linear layer from each window straight to phonemes.

    It has no memory of earlier windows and no per-session layer, so it shows how
    far you get without the recurrent network.
    """

    def __init__(self, n_features, n_sessions, n_classes, patch_size=14, patch_stride=4):
        super().__init__()
        self.frontend = Frontend(patch_size=patch_size, patch_stride=patch_stride)
        self.out = nn.Linear(n_features * patch_size, n_classes)

    def forward(self, x, lengths, session_idx):
        x, out_lengths = self.frontend(x, lengths)
        return self.out(x), out_lengths


MODELS = {"linear": LinearBaseline}


def build_model(name, n_features, n_sessions, n_classes, **kwargs):
    return MODELS[name](n_features, n_sessions, n_classes, **kwargs)
