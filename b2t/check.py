"""Sanity checks for the data layer. Run after making the splits:

    python -m b2t.check configs/t15.toml

It reads labels and metadata for every trial (fast), loads the neural features for
one session only, and tests the error-rate functions on examples with known answers.
"""

import sys
from collections import Counter

from b2t.config import load_config
from b2t.data import SPLITS, load_splits, load_trials
from b2t.metrics import normalize_text, phoneme_error_rate, word_error_rate
from b2t.phonemes import N_CLASSES, ids_to_phonemes


def main(config_path):
    cfg = load_config(config_path)
    splits = load_splits(cfg)

    print(f"Participant {cfg.participant}: {len(splits['sessions'])} sessions")
    meta = {s: load_trials(cfg, s, load_features=False) for s in SPLITS}
    for split, trials in meta.items():
        hours = sum(t.n_steps for t in trials) * cfg.bin_ms / 1000 / 3600
        print(f"  {split:5s} {len(trials):5d} trials  {hours:5.2f} hours")

    # No trial may be in both train and tune.
    ids = lambda trials: {(t.session, t.key) for t in trials}
    assert not ids(meta["train"]) & ids(meta["tune"]), "train and tune overlap"

    # Labels must be valid phoneme ids, and test trials must have no labels.
    for split in ("train", "tune", "val"):
        for t in meta[split]:
            assert t.sentence and len(t.phoneme_ids) > 0
            assert 0 < t.phoneme_ids.min() and t.phoneme_ids.max() < N_CLASSES
    assert all(t.sentence is None for t in meta["test"])

    # How many val sentences were also seen in training? (Reported separately later.)
    seen = {normalize_text(t.sentence) for t in meta["train"] + meta["tune"]}
    repeats = sum(normalize_text(t.sentence) in seen for t in meta["val"])
    print(f"  {repeats} of {len(meta['val'])} val sentences also appear in training")

    held = Counter(t.session for t in meta["val"] if t.session in splits["holdout_sessions"])
    print(f"  held-out sessions have {sum(held.values())} val trials")

    # Load real neural features for one session and look at one trial.
    session = splits["sessions"][1]
    trials = load_trials(cfg, "train", sessions=[session])
    t = trials[0]
    assert t.features.shape == (t.n_steps, cfg.n_features)
    print(f"\nExample trial: {t.session} block {t.block} trial {t.trial}")
    print(f"  features  {t.features.shape} {t.features.dtype} "
          f"= {t.n_steps * cfg.bin_ms / 1000:.1f} seconds")
    print(f"  sentence  {t.sentence}")
    print(f"  phonemes  {' '.join(ids_to_phonemes(t.phoneme_ids))}")

    # Error-rate functions on cases where we know the right answer.
    assert word_error_rate(["The cat sat."], ["the cat sat"]) == 0
    assert word_error_rate(["the cat sit"], ["the cat sat"]) == 1 / 3
    assert word_error_rate(["the cat"], ["the cat sat"]) == 1 / 3
    assert word_error_rate(["dont go"], ["don't go"]) == 1 / 2
    assert phoneme_error_rate([[1, 2, 3]], [[1, 2, 3]]) == 0
    assert phoneme_error_rate([[1, 2], [5]], [[1, 2, 3], [4]]) == 2 / 4
    print("\nAll checks passed.")


if __name__ == "__main__":
    main(sys.argv[1])
