"""Runs a saved model on a few trials and prints what it predicted. Example:

    python -m b2t.predict --run runs/linear --n 5

For each trial it shows the true sentence, the true phonemes, the phonemes the model
predicted, and that trial's phoneme error rate. Use --split test to try trials that
have no labels (only the prediction is shown).
"""

import argparse
import json
import random
from pathlib import Path

import torch

from b2t.batching import make_batch
from b2t.config import load_config
from b2t.data import SPLITS, load_splits, load_trials
from b2t.metrics import edit_distance
from b2t.models import build_model
from b2t.phonemes import ids_to_phonemes
from b2t.train import greedy_decode


def load_bundle(run_dir, device):
    """Rebuild a trained model from its output bundle folder."""
    run_dir = Path(run_dir)
    with open(run_dir / "run.json") as f:
        run = json.load(f)
    model = build_model(run["model"], run["n_features"], len(run["all_sessions"]),
                        run["n_classes"], **run.get("model_kwargs", {}))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    return model.to(device).eval(), run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="bundle folder, e.g. runs/linear")
    p.add_argument("--split", default="val", choices=SPLITS)
    p.add_argument("--session", default=None, help="e.g. t15.2023.08.13 (default: random)")
    p.add_argument("--n", type=int, default=5, help="how many trials to show")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, run = load_bundle(args.run, device)
    cfg = load_config(Path(args.run) / "config.toml")

    # Pick the trials first using labels only, then load features for just those.
    rng = random.Random(args.seed)
    session = args.session
    if session is None:
        has_split = [s for s in load_splits(cfg)["sessions"]
                     if (cfg.data_dir / s / cfg.files["train" if args.split in ("train", "tune") else args.split]).exists()]
        session = rng.choice(has_split)
    trials = load_trials(cfg, args.split, sessions=[session])
    trials = rng.sample(trials, min(args.n, len(trials)))

    print(f"Model '{run['model']}' from {args.run}, {args.split} trials from {session}\n")
    with torch.no_grad():
        for t in trials:
            if t.phoneme_ids is None:  # test trials: make_batch needs some label array
                t.phoneme_ids = []
            batch = make_batch([t], device)
            logits, out_lengths = model(batch["x"], batch["lengths"], batch["session_idx"])
            pred = greedy_decode(logits, out_lengths)[0]

            print(f"block {t.block} trial {t.trial} ({t.n_steps * cfg.bin_ms / 1000:.1f} s)")
            if t.sentence is not None:
                print(f"  sentence   {t.sentence}")
                print(f"  true       {' '.join(ids_to_phonemes(t.phoneme_ids))}")
            print(f"  predicted  {' '.join(ids_to_phonemes(pred))}")
            if t.sentence is not None:
                rate = edit_distance(pred, list(t.phoneme_ids)) / len(t.phoneme_ids)
                print(f"  phoneme error rate {rate:.0%}")
            print()


if __name__ == "__main__":
    main()
