"""The calibration experiment: how many sentences does a new day need? Example:

    python -m b2t.calibrate --run runs/run_b

Takes a model that was trained WITHOUT some sessions (the held-out sessions). For
each held-out session it locks the shared network, trains only that session's small
input layer on N sentences from that session, and scores the result on that
session's val trials. It repeats this for several values of N, starting fresh from
the saved model every time.

This is our stand-in for a new user: a recording day the model has never seen.

Two starting points for the new session's input layer are compared:
    identity - the layer starts as "change nothing"
    recent   - the layer starts as a copy of the most recent trained session's layer

Results are printed and saved to results/calibration_<run name>.json.
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from b2t.augment import augment
from b2t.batching import make_batch
from b2t.config import REPO_ROOT, load_config
from b2t.data import load_splits, load_trials
from b2t.decode import DictionaryDecoder, count_words
from b2t.metrics import phoneme_error_rate, word_error_rate
from b2t.phonemes import BLANK_ID
from b2t.predict import load_bundle
from b2t.train import evaluate


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
    p.add_argument("--run", required=True, help="bundle trained with --exclude-holdout")
    # The held-out sessions have between 53 and 149 training sentences each, so sizes
    # up to 50 can be tried on all of them. Larger sizes only cover some sessions.
    p.add_argument("--sizes", type=int, nargs="+", default=[0, 5, 10, 25, 50],
                   help="numbers of calibration sentences to try")
    p.add_argument("--steps", type=int, default=200, help="training steps per calibration")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, run = load_bundle(args.run, device)
    if not hasattr(model, "session_layers"):
        raise SystemExit("This model has no per-session layers, so it cannot be calibrated.")
    cfg = load_config(Path(args.run) / "config.toml")
    splits = load_splits(cfg)

    trained_on = run["trained_on_sessions"]
    new_sessions = [s for s in splits["holdout_sessions"] if s not in trained_on]
    if not new_sessions:
        raise SystemExit("This model was trained on every session; use a run made "
                         "with --exclude-holdout.")
    all_sessions = run["all_sessions"]
    recent_idx = all_sessions.index(trained_on[-1])
    print(f"Model from {args.run} on {device}, trained on {len(trained_on)} sessions.")
    print(f"New sessions: {', '.join(new_sessions)}")
    print(f"'recent' start copies the layer of {trained_on[-1]}\n")

    # The saved weights, kept aside so every calibration starts from the same place.
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    decoder = DictionaryDecoder(
        count_words(t.sentence for t in load_trials(cfg, "train", sessions=trained_on,
                                                    load_features=False)))

    rows = []       # one row per (start, session, size)
    collected = {}  # (start, size) -> predictions and truths across all new sessions
    for session in new_sessions:
        pool = load_trials(cfg, "train", sessions=[session])
        test = load_trials(cfg, "val", sessions=[session])
        if not test:
            continue
        random.Random(args.seed).shuffle(pool)  # so the 10 are inside the 25, and so on
        session_idx = all_sessions.index(session)
        sizes = [n for n in args.sizes if n <= len(pool)]
        print(f"{session}: {len(pool)} sentences available, {len(test)} val trials")

        for start in ("identity", "recent"):
            for n in sizes:
                model.load_state_dict(saved)
                if start == "recent":
                    with torch.no_grad():
                        model.session_layers.weight[session_idx] = model.session_layers.weight[recent_idx]
                        model.session_layers.bias[session_idx] = model.session_layers.bias[recent_idx]
                if n > 0:
                    fit_session_layer(model, pool[:n], device, args.steps, args.batch_size, args.lr)

                per, preds = evaluate(model, test, device, batch_size=32)
                truth_ids = [t.phoneme_ids for t in test]
                truth_text = [t.sentence for t in test]
                decoded = [decoder.decode(ids) for ids in preds]
                wer = word_error_rate(decoded, truth_text)
                rows.append({"start": start, "session": session, "n_sentences": n,
                             "phoneme_error_rate": per, "word_error_rate_dictionary": wer,
                             "n_val_trials": len(test)})
                bucket = collected.setdefault((start, n), {"p": [], "t": [], "d": [], "s": [], "sessions": 0})
                bucket["p"] += preds
                bucket["t"] += truth_ids
                bucket["d"] += decoded
                bucket["s"] += truth_text
                bucket["sessions"] += 1
                print(f"  {start:8s} {n:4d} sentences: phoneme error {per:6.1%}   word error {wer:6.1%}")

    # Combine the new sessions into one curve per starting point.
    summary = []
    print("\nAll new sessions together:")
    print("start    | sentences | sessions | phoneme error | word error (dictionary)")
    for (start, n), b in sorted(collected.items()):
        per = phoneme_error_rate(b["p"], b["t"])
        wer = word_error_rate(b["d"], b["s"])
        summary.append({"start": start, "n_sentences": n, "n_sessions": b["sessions"],
                        "n_val_trials": len(b["p"]), "phoneme_error_rate": per,
                        "word_error_rate_dictionary": wer})
        print(f"{start:8s} | {n:9d} | {b['sessions']:8d} | {per:13.1%} | {wer:.1%}")
    print("\nA row covering fewer sessions than the others (a session did not have that "
          "many sentences) is not directly comparable with the rest.")

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"calibration_{Path(args.run).name}.json"
    out_path.write_text(json.dumps({
        "run": args.run,
        "trained_on_sessions": trained_on,
        "new_sessions": new_sessions,
        "recent_start_copied_from": trained_on[-1],
        "settings": {"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
                     "seed": args.seed, "sizes": args.sizes},
        "summary": summary,
        "per_session": rows,
    }, indent=1))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
