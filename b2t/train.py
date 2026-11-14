"""Trains a model and saves an output bundle. Example:

    python -m b2t.train --config configs/t15.toml --model linear --out runs/linear

The bundle folder (--out) holds everything needed to reuse or audit the run:
    model.pt   - weights from the step with the best phoneme error rate on tune
    run.json   - model type, settings, session list, phoneme vocabulary, metrics
    config.toml - copy of the participant config that was used

Training uses CTC loss, which lets the model learn phoneme sequences without being
told when each phoneme happened.
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from b2t.batching import make_batch
from b2t.config import load_config
from b2t.data import load_splits, load_trials
from b2t.metrics import phoneme_error_rate
from b2t.models import MODELS, build_model
from b2t.phonemes import BLANK_ID, N_CLASSES, PHONEMES


def greedy_decode(logits, out_lengths):
    """Pick the top class at each step, merge repeats, then drop blanks."""
    best = logits.argmax(-1).cpu().numpy()
    decoded = []
    for seq, n in zip(best, out_lengths.cpu().numpy()):
        seq = seq[:n]
        keep = [int(p) for i, p in enumerate(seq) if i == 0 or p != seq[i - 1]]
        decoded.append([p for p in keep if p != BLANK_ID])
    return decoded


@torch.no_grad()
def evaluate(model, trials, device, batch_size=64):
    """Phoneme error rate of greedy decoding. Returns (rate, predictions)."""
    model.eval()
    order = sorted(range(len(trials)), key=lambda i: trials[i].n_steps)  # less padding
    preds = [None] * len(trials)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = make_batch([trials[i] for i in idx], device)
        logits, out_lengths = model(batch["x"], batch["lengths"], batch["session_idx"])
        for i, p in zip(idx, greedy_decode(logits, out_lengths)):
            preds[i] = p
    model.train()
    return phoneme_error_rate(preds, [t.phoneme_ids for t in trials]), preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--out", required=True, help="folder for the output bundle")
    p.add_argument("--batches", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-holdout", action="store_true",
                   help="leave out the config's holdout_sessions (Run B)")
    p.add_argument("--limit-sessions", type=int, default=None,
                   help="only use the first N sessions (quick tests on a laptop)")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(args.config)
    splits = load_splits(cfg)
    sessions = splits["sessions"]
    if args.exclude_holdout:
        sessions = [s for s in sessions if s not in splits["holdout_sessions"]]
    if args.limit_sessions:
        sessions = sessions[: args.limit_sessions]

    print(f"Loading {len(sessions)} sessions into memory...")
    t0 = time.time()
    train = load_trials(cfg, "train", sessions=sessions)
    tune = load_trials(cfg, "tune", sessions=sessions)
    print(f"  {len(train)} train and {len(tune)} tune trials in {time.time() - t0:.0f}s")

    # Sessions are numbered by their place in the full list, so a model always has
    # one slot per session even when some are left out of this run.
    model = build_model(args.model, cfg.n_features, len(splits["sessions"]), N_CLASSES)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model '{args.model}' with {n_params:,} parameters on {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.toml")

    history, best_per, losses = [], float("inf"), []
    t0 = time.time()
    for step in range(1, args.batches + 1):
        batch = make_batch(random.sample(train, args.batch_size), device)
        logits, out_lengths = model(batch["x"], batch["lengths"], batch["session_idx"])
        loss = F.ctc_loss(
            logits.log_softmax(-1).transpose(0, 1),  # CTC wants (time, batch, classes)
            batch["labels"], out_lengths, batch["label_lengths"],
            blank=BLANK_ID, zero_infinity=True,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(loss.item())

        if step % args.log_every == 0:
            speed = step / (time.time() - t0)
            print(f"step {step:6d}  loss {sum(losses) / len(losses):.3f}  "
                  f"{speed:.1f} batches/s")
            losses = []

        if step % args.eval_every == 0 or step == args.batches:
            per, _ = evaluate(model, tune, device)
            history.append({"step": step, "tune_per": per})
            marker = ""
            if per < best_per:
                best_per = per
                torch.save(model.state_dict(), out / "model.pt")
                marker = "  (best, saved)"
            print(f"step {step:6d}  tune phoneme error rate {per:.1%}{marker}")

    with open(out / "run.json", "w") as f:
        json.dump({
            "participant": cfg.participant,
            "model": args.model,
            "n_features": cfg.n_features,
            "n_classes": N_CLASSES,
            "phonemes": PHONEMES,
            "all_sessions": splits["sessions"],
            "trained_on_sessions": sessions,
            "args": vars(args),
            "n_parameters": n_params,
            "best_tune_per": best_per,
            "history": history,
            "train_minutes": (time.time() - t0) / 60,
        }, f, indent=1)
    print(f"Done. Best tune phoneme error rate {best_per:.1%}. Bundle saved in {out}/")


if __name__ == "__main__":
    main()
