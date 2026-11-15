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
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from b2t.augment import augment
from b2t.batching import make_batch
from b2t.config import load_config
from b2t.data import load_splits, load_trials
from b2t.metrics import phoneme_error_rate
from b2t.models import MODELS, build_model
from b2t.phonemes import BLANK_ID, N_CLASSES, PHONEMES


# Training settings per model, used unless overridden on the command line.
# The GRU values follow the Brain-to-Text '25 baseline.
DEFAULTS = {
    "linear": dict(lr=1e-3, eps=1e-8, weight_decay=0.01, cosine=False, augment=False),
    "gru": dict(lr=5e-3, eps=0.1, weight_decay=1e-3, cosine=True, augment=True),
}


def learning_rate_scale(step, total, warmup=1000, floor=0.02):
    """Ramp up for the first steps, then glide down along a cosine curve."""
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def keep_fraction(trials, fraction):
    """Keep a random share of each session's trials, for the data-amount experiment.

    Every session keeps at least one trial, so each per-session layer still gets
    some training. The shuffle uses a fixed seed, so a smaller fraction is always a
    subset of a larger one (the 10% trials are all inside the 25% trials).
    """
    by_session = {}
    for t in trials:
        by_session.setdefault(t.session, []).append(t)
    kept = []
    for session in sorted(by_session):
        group = by_session[session]
        random.Random(f"fraction-{session}").shuffle(group)
        kept.extend(group[: max(1, round(len(group) * fraction))])
    return kept


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
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--units", type=int, default=768, help="GRU size (gru only)")
    p.add_argument("--layers", type=int, default=5, help="GRU depth (gru only)")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-holdout", action="store_true",
                   help="leave out the config's holdout_sessions (unused by our reported models)")
    p.add_argument("--train-until", default=None, metavar="SESSION",
                   help="train only on sessions up to and including this one, "
                        "e.g. t15.2023.12.29 (for the drift experiment)")
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="train on only this share of the training sentences (0 to 1)")
    p.add_argument("--limit-sessions", type=int, default=None,
                   help="only use the first N sessions (quick tests on a laptop)")
    args = p.parse_args()
    settings = dict(DEFAULTS[args.model])
    if args.lr is not None:
        settings["lr"] = args.lr

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(args.config)
    splits = load_splits(cfg)
    sessions = splits["sessions"]
    if args.exclude_holdout:
        sessions = [s for s in sessions if s not in splits["holdout_sessions"]]
    if args.train_until:
        # Session names contain their date, so they sort in time order.
        if args.train_until not in splits["sessions"]:
            raise SystemExit(f"--train-until: no session called {args.train_until}")
        sessions = [s for s in sessions if s <= args.train_until]
    if args.limit_sessions:
        sessions = sessions[: args.limit_sessions]

    print(f"Loading {len(sessions)} sessions into memory...")
    t0 = time.time()
    train = load_trials(cfg, "train", sessions=sessions)
    tune = load_trials(cfg, "tune", sessions=sessions)
    n_train_available = len(train)
    if args.train_fraction < 1.0:
        # Only the training sentences are cut. Tune stays whole, so every run is
        # scored on the same sentences.
        train = keep_fraction(train, args.train_fraction)
        print(f"  keeping {args.train_fraction:.0%} of training sentences: "
              f"{len(train)} of {n_train_available}")
    print(f"  {len(train)} train and {len(tune)} tune trials in {time.time() - t0:.0f}s")

    # Sessions are numbered by their place in the full list, so a model always has
    # one slot per session even when some are left out of this run.
    model_kwargs = {"units": args.units, "layers": args.layers} if args.model == "gru" else {}
    model = build_model(args.model, cfg.n_features, len(splits["sessions"]), N_CLASSES,
                        **model_kwargs)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model '{args.model}' with {n_params:,} parameters on {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
                                  eps=settings["eps"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        (lambda s: learning_rate_scale(s, args.batches)) if settings["cosine"] else (lambda s: 1.0),
    )
    # Mixed precision: do most of the maths in 16-bit on the GPU, which is faster
    # and uses less memory. The loss itself is still computed in 32-bit.
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.toml")

    history, best_per, losses = [], float("inf"), []
    t0 = time.time()
    for step in range(1, args.batches + 1):
        batch = make_batch(random.sample(train, min(args.batch_size, len(train))), device)
        x, lengths = batch["x"], batch["lengths"]
        if settings["augment"]:
            x, lengths = augment(x, lengths)
        with torch.autocast(device_type=device, enabled=use_amp):
            logits, out_lengths = model(x, lengths, batch["session_idx"])
        loss = F.ctc_loss(
            logits.float().log_softmax(-1).transpose(0, 1),  # (time, batch, classes)
            batch["labels"], out_lengths, batch["label_lengths"],
            blank=BLANK_ID, zero_infinity=True,
        )
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
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
            "train_fraction": args.train_fraction,
            "n_train_trials": len(train),
            "n_train_trials_available": n_train_available,
            "model_kwargs": model_kwargs,
            "settings": settings,
            "args": vars(args),
            "n_parameters": n_params,
            "best_tune_per": best_per,
            "history": history,
            "train_minutes": (time.time() - t0) / 60,
        }, f, indent=1)
    print(f"Done. Best tune phoneme error rate {best_per:.1%}. Bundle saved in {out}/")


if __name__ == "__main__":
    main()
