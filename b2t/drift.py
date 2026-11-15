"""The drift experiment: how fast does accuracy fall over time, and does a quick
recalibration bring it back? Example:

    python -m b2t.drift --run runs/drift

Takes a model trained only on sessions up to some date (made with --train-until).
For every LATER session it scores the model on that session's val trials:

    0 sentences  - no recalibration. The later session has no input layer of its
                   own, so it uses a copy of the last training day's layer.
    N sentences  - recalibrated: starting from that copy, only this session's small
                   input layer is trained on N sentences recorded that day. The
                   shared network is locked.

Plotting error against time since the last training day shows the drift, and the
gap between the lines shows what recalibration buys.

This is one participant, so it measures change over time within a person. It says
nothing about a different person.

Results are printed and saved to results/drift_<run name>.json.
"""

import argparse
import datetime
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from b2t.augment import augment
from b2t.batching import make_batch
from b2t.config import REPO_ROOT, load_config
from b2t.data import load_trials
from b2t.decode import DictionaryDecoder, count_words
from b2t.metrics import phoneme_error_rate, word_error_rate
from b2t.phonemes import BLANK_ID
from b2t.predict import load_bundle
from b2t.train import evaluate


def session_date(session):
    """'t15.2023.12.29' -> the date 2023-12-29 (session names end in year.month.day)."""
    year, month, day = session.split(".")[-3:]
    return datetime.date(int(year), int(month), int(day))


def fit_session_layer(model, sentences, device, steps, batch_size, lr):
    """Train only the per-session input layers on the given sentences.

    Everything else in the model is locked. Only the layer for these sentences'
    session receives any signal, so the other sessions' layers stay as they were.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    personal = [model.session_layers.weight, model.session_layers.bias]
    for p in personal:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(personal, lr=lr, weight_decay=0.0)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    model.train()
    for _ in range(steps):
        batch = make_batch(random.sample(sentences, min(batch_size, len(sentences))), device)
        x, lengths = augment(batch["x"], batch["lengths"])
        with torch.autocast(device_type=device, enabled=use_amp):
            logits, out_lengths = model(x, lengths, batch["session_idx"])
        loss = F.ctc_loss(
            logits.float().log_softmax(-1).transpose(0, 1),
            batch["labels"], out_lengths, batch["label_lengths"],
            blank=BLANK_ID, zero_infinity=True,
        )
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="bundle trained with --train-until")
    p.add_argument("--sizes", type=int, nargs="+", default=[0, 10, 50],
                   help="numbers of recalibration sentences to try (0 = none)")
    p.add_argument("--steps", type=int, default=200, help="training steps per recalibration")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-sessions", type=int, default=None,
                   help="only test the first N later sessions (quick tests)")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, run = load_bundle(args.run, device)
    if not hasattr(model, "session_layers"):
        raise SystemExit("This model has no per-session layers, so it cannot be recalibrated.")
    cfg = load_config(Path(args.run) / "config.toml")

    all_sessions = run["all_sessions"]
    trained_on = run["trained_on_sessions"]
    last_trained = trained_on[-1]
    later = [s for s in all_sessions if s > last_trained]
    if not later:
        raise SystemExit("This model was trained on every session; use a run made "
                         "with --train-until.")
    if args.limit_sessions:
        later = later[: args.limit_sessions]
    last_idx = all_sessions.index(last_trained)
    print(f"Model from {args.run} on {device}, trained on {len(trained_on)} sessions "
          f"up to {last_trained}.")
    print(f"Testing on {len(later)} later sessions.\n")

    # The saved weights, kept aside so every test starts from the same place.
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    decoder = DictionaryDecoder(
        count_words(t.sentence for t in load_trials(cfg, "train", sessions=trained_on,
                                                    load_features=False)))

    rows = []
    collected = {}  # size -> predictions and truths across all later sessions
    for session in later:
        test = load_trials(cfg, "val", sessions=[session])
        if not test:
            print(f"{session}: no val trials, skipped")
            continue
        pool = load_trials(cfg, "train", sessions=[session])
        random.Random(args.seed).shuffle(pool)  # so the 10 are inside the 50
        session_idx = all_sessions.index(session)
        days = (session_date(session) - session_date(last_trained)).days
        print(f"{session}: {days} days after training, {len(test)} val trials, "
              f"{len(pool)} sentences available for recalibration")

        for n in args.sizes:
            if n > len(pool):
                continue
            model.load_state_dict(saved)
            with torch.no_grad():  # start from the last training day's layer
                model.session_layers.weight[session_idx] = model.session_layers.weight[last_idx]
                model.session_layers.bias[session_idx] = model.session_layers.bias[last_idx]
            if n > 0:
                fit_session_layer(model, pool[:n], device, args.steps, args.batch_size, args.lr)

            per, preds = evaluate(model, test, device, batch_size=32)
            decoded = [decoder.decode(ids) for ids in preds]
            wer = word_error_rate(decoded, [t.sentence for t in test])
            rows.append({"session": session, "days_since_training": days, "n_sentences": n,
                         "phoneme_error_rate": per, "word_error_rate_dictionary": wer,
                         "n_val_trials": len(test)})
            bucket = collected.setdefault(n, {"p": [], "t": [], "d": [], "s": [], "sessions": 0})
            bucket["p"] += preds
            bucket["t"] += [t.phoneme_ids for t in test]
            bucket["d"] += decoded
            bucket["s"] += [t.sentence for t in test]
            bucket["sessions"] += 1
            print(f"  {n:3d} recalibration sentences: phoneme error {per:6.1%}   "
                  f"word error {wer:6.1%}")

    summary = []
    print("\nAll later sessions together:")
    print("recalibration sentences | sessions | phoneme error | word error (dictionary)")
    for n, b in sorted(collected.items()):
        per = phoneme_error_rate(b["p"], b["t"])
        wer = word_error_rate(b["d"], b["s"])
        summary.append({"n_sentences": n, "n_sessions": b["sessions"], "n_val_trials": len(b["p"]),
                        "phoneme_error_rate": per, "word_error_rate_dictionary": wer})
        print(f"{n:23d} | {b['sessions']:8d} | {per:13.1%} | {wer:.1%}")

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"drift_{Path(args.run).name}.json"
    out_path.write_text(json.dumps({
        "run": args.run,
        "trained_on_sessions": trained_on,
        "last_training_session": last_trained,
        "settings": {"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
                     "seed": args.seed, "sizes": args.sizes},
        "summary": summary,
        "per_session": rows,
    }, indent=1))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
