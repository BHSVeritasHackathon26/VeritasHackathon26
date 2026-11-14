"""The data contract: every recording, from any participant, becomes a list of Trial
objects. All later code (models, decoders, evaluation) only ever sees Trials, so
supporting a new person means writing a new loader, not changing anything else.

Splits used in this project:
    train - official training trials, minus the tuning trials
    tune  - ~10% of official training trials, for early stopping and picking settings
    val   - official validation trials: our final, held-out test set
    test  - official test trials: no labels, only useful as demo inputs
"""

import json
from dataclasses import dataclass

import h5py
import numpy as np

SPLITS = ("train", "tune", "val", "test")


@dataclass
class Trial:
    participant: str
    session: str          # recording day, e.g. "t15.2023.08.13"
    session_idx: int      # position of the session in the participant's session list
    block: int
    trial: int
    key: str              # name of the trial inside its file, e.g. "trial_0000"
    features: np.ndarray  # shape (time steps, n_features), float32; None if not loaded
    n_steps: int          # number of time steps
    phoneme_ids: np.ndarray  # shape (n phonemes,), ints from b2t.phonemes; None for test
    sentence: str         # what the participant was trying to say; None for test


def list_sessions(cfg):
    """All session folders that contain a training file, in date order."""
    return sorted(
        d.name for d in cfg.data_dir.iterdir()
        if (d / cfg.files["train"]).exists()
    )


def load_splits(cfg):
    if not cfg.splits_file.exists():
        raise FileNotFoundError(
            f"{cfg.splits_file} not found. Run: python -m b2t.splits <config file>"
        )
    with open(cfg.splits_file) as f:
        return json.load(f)


def load_trials(cfg, split, sessions=None, load_features=True):
    """Load every trial of one split as a list of Trial objects.

    sessions:      only load these sessions (default: all of them).
    load_features: set False to read just labels and metadata, which is fast and
                   uses almost no memory.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    splits = load_splits(cfg)
    all_sessions = splits["sessions"]
    # "train" and "tune" both come out of the official training file.
    file_name = cfg.files["train" if split in ("train", "tune") else split]

    trials = []
    for session in sessions or all_sessions:
        path = cfg.data_dir / session / file_name
        if not path.exists():  # a few sessions have no val/test file
            continue
        tune_keys = set(splits["tune"].get(session, []))

        with h5py.File(path, "r") as f:
            for key in sorted(f.keys()):
                if split == "train" and key in tune_keys:
                    continue
                if split == "tune" and key not in tune_keys:
                    continue
                g = f[key]

                features = None
                if load_features:
                    features = g["input_features"][:]
                    if features.shape[1] != cfg.n_features:
                        raise ValueError(
                            f"{session}/{key}: expected {cfg.n_features} features, "
                            f"got {features.shape[1]}"
                        )

                phoneme_ids, sentence = None, None
                if "seq_class_ids" in g:
                    # Stored padded with zeros to length 500; keep only the real part.
                    phoneme_ids = g["seq_class_ids"][: int(g.attrs["seq_len"])]
                    sentence = str(g.attrs["sentence_label"])

                trials.append(Trial(
                    participant=cfg.participant,
                    session=session,
                    session_idx=all_sessions.index(session),
                    block=int(g.attrs["block_num"]),
                    trial=int(g.attrs["trial_num"]),
                    key=key,
                    features=features,
                    n_steps=int(g.attrs["n_time_steps"]),
                    phoneme_ids=phoneme_ids,
                    sentence=sentence,
                ))
    return trials
