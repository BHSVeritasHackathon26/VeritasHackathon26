"""Makes the frozen split list for a participant. Run once, then commit the result:

    python -m b2t.splits configs/t15.toml

The output file records (1) the session order, which fixes which per-session input
layer each session uses, (2) which training trials are set aside for tuning, and
(3) which sessions are held out for the calibration experiment. Because the file is
saved in the repo, every result we report uses exactly the same split.
"""

import json
import random
import sys

import h5py

from b2t.config import load_config
from b2t.data import list_sessions


def make_splits(cfg):
    sessions = list_sessions(cfg)
    unknown = set(cfg.holdout_sessions) - set(sessions)
    if unknown:
        raise ValueError(f"holdout_sessions not found in data: {sorted(unknown)}")

    rng = random.Random(cfg.seed)
    tune = {}
    for session in sessions:
        with h5py.File(cfg.data_dir / session / cfg.files["train"], "r") as f:
            keys = sorted(f.keys())
        n_tune = max(1, round(len(keys) * cfg.tune_fraction))
        tune[session] = sorted(rng.sample(keys, n_tune))

    return {
        "participant": cfg.participant,
        "sessions": sessions,
        "holdout_sessions": list(cfg.holdout_sessions),
        "tune": tune,
    }


if __name__ == "__main__":
    cfg = load_config(sys.argv[1])
    splits = make_splits(cfg)
    cfg.splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.splits_file, "w") as f:
        json.dump(splits, f, indent=1)
    n_tune = sum(len(v) for v in splits["tune"].values())
    print(f"Wrote {cfg.splits_file}")
    print(f"{len(splits['sessions'])} sessions, {n_tune} tuning trials, "
          f"{len(splits['holdout_sessions'])} held-out sessions")
