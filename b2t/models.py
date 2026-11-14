"""Models. Every model takes neural features and returns, for each output time step,
a score for each of the 41 phoneme classes.

All models share the same Frontend, so the only difference between the simple
baseline and the real model is what sits around it.
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

    def smooth(self, x):
        batch, time, n_features = x.shape
        x = x.transpose(1, 2).reshape(batch * n_features, 1, time)
        x = F.conv1d(x, self.kernel.to(x.dtype), padding=self.kernel.shape[-1] // 2)
        return x.reshape(batch, n_features, time).transpose(1, 2)

    def patch(self, x, lengths):
        """(batch, time, features) -> (batch, n_windows, patch_size * features)"""
        x = x.unfold(1, self.patch_size, self.patch_stride)   # (batch, n, features, patch)
        x = x.permute(0, 1, 3, 2).flatten(2)
        out_lengths = (lengths - self.patch_size) // self.patch_stride + 1
        return x, out_lengths


class LinearBaseline(nn.Module):
    """The simple baseline: one linear layer from each window straight to phonemes.

    It has no memory of earlier windows and no per-session layer, so it shows how
    far you get without the recurrent network.
    """

    def __init__(self, n_features, n_sessions, n_classes):
        super().__init__()
        self.frontend = Frontend()
        self.out = nn.Linear(n_features * self.frontend.patch_size, n_classes)

    def forward(self, x, lengths, session_idx):
        x, out_lengths = self.frontend.patch(self.frontend.smooth(x), lengths)
        return self.out(x), out_lengths


class SessionLayers(nn.Module):
    """One small input layer per recording session: the personal part of the model.

    The same electrodes pick up slightly different signals from day to day, so each
    session gets its own (n_features x n_features) layer that re-maps that day's
    signals into a common form. Each starts as "change nothing" and learns from there.
    Fitting a brand-new layer is also how the model adapts to a new day or person.
    """

    def __init__(self, n_features, n_sessions, dropout=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(n_features).repeat(n_sessions, 1, 1))
        self.bias = nn.Parameter(torch.zeros(n_sessions, n_features))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, session_idx):
        w = self.weight[session_idx]                 # (batch, features, features)
        b = self.bias[session_idx].unsqueeze(1)      # (batch, 1, features)
        x = torch.einsum("btf,bkf->btk", x, w) + b
        return self.dropout(F.softsign(x))


class GRUDecoder(nn.Module):
    """The real model, following the Brain-to-Text '25 baseline design:

        smooth -> per-session layer -> windows -> GRU -> phoneme scores

    The GRU is a recurrent network: it carries a memory forward through time, so its
    guess at each step can use everything it has seen so far in the sentence. It only
    looks backwards, which means it could run live while someone is speaking.
    Everything after the per-session layer is shared across all sessions.
    """

    def __init__(self, n_features, n_sessions, n_classes,
                 units=768, layers=5, dropout=0.4, session_dropout=0.2):
        super().__init__()
        self.frontend = Frontend()
        self.session_layers = SessionLayers(n_features, n_sessions, session_dropout)
        self.gru = nn.GRU(
            n_features * self.frontend.patch_size, units, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0, batch_first=True,
        )
        self.out = nn.Linear(units, n_classes)
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x, lengths, session_idx):
        x = self.frontend.smooth(x)
        x = self.session_layers(x, session_idx)
        x, out_lengths = self.frontend.patch(x, lengths)
        x, _ = self.gru(x)
        return self.out(x), out_lengths


MODELS = {"linear": LinearBaseline, "gru": GRUDecoder}


def build_model(name, n_features, n_sessions, n_classes, **kwargs):
    return MODELS[name](n_features, n_sessions, n_classes, **kwargs)
