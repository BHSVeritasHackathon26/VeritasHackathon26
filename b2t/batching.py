"""Turns a list of Trials into one batch of tensors the model can train on.

Trials have different lengths, so each batch is padded with zeros up to the longest
trial in it, and the true lengths are passed along so the padding can be ignored.
"""

import numpy as np
import torch


def make_batch(trials, device):
    lengths = [t.features.shape[0] for t in trials]
    label_lengths = [len(t.phoneme_ids) for t in trials]
    n_features = trials[0].features.shape[1]

    x = np.zeros((len(trials), max(lengths), n_features), dtype=np.float32)
    labels = np.zeros((len(trials), max(label_lengths)), dtype=np.int64)
    for i, t in enumerate(trials):
        x[i, : lengths[i]] = t.features
        labels[i, : label_lengths[i]] = t.phoneme_ids

    return {
        "x": torch.from_numpy(x).to(device),                  # (batch, time, features)
        "lengths": torch.tensor(lengths, device=device),      # real time steps per trial
        "labels": torch.from_numpy(labels).to(device),        # (batch, max phonemes)
        "label_lengths": torch.tensor(label_lengths, device=device),
        "session_idx": torch.tensor([t.session_idx for t in trials], device=device),
    }
