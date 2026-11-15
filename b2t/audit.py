"""Checks that no test data leaked into training, and saves a report. Example:

    python -m b2t.audit configs/t15.toml

"val" is our final test set. The model learns from "train" and is tuned on "tune".
This script checks, in increasing strength:

  1. No trial (session, block, trial number) is in more than one split.
  2. No recording block is shared between val and train/tune. A block is one
     continuous stretch of recording, so this means the model never saw even the
     neighbouring trials of a test trial.
  3. No two splits contain the same neural data. Every trial's recording is
     fingerprinted (a SHA-256 hash of its numbers); identical recordings would have
     identical fingerprints, whatever they were named.
  4. The training script only ever asks for the train and tune splits.
  5. Which sessions each saved run was trained on.

It also lists val sentences whose wording appears in training. Those are different
recordings of the same text, so they are not leaked data, but they are easier and
must be reported separately.

The report is written to results/audit_<participant>.json.
"""

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from b2t.config import REPO_ROOT, load_config
from b2t.data import SPLITS, load_splits, load_trials
from b2t.metrics import normalize_text


def fingerprint(features):
    """A short code that is the same only if the recording's numbers are identical."""
    h = hashlib.sha256()
    h.update(str(features.shape).encode())
    h.update(features.tobytes())
    return h.hexdigest()


def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return out.stdout.strip() or None
    except OSError:
        return None


def main(config_path):
    cfg = load_config(config_path)
    splits = load_splits(cfg)
    checks = []  # (name, passed, detail)

    # Labels and identifiers only; fast.
    meta = {s: load_trials(cfg, s, load_features=False) for s in SPLITS}
    trial_ids = {s: {(t.session, t.block, t.trial) for t in ts} for s, ts in meta.items()}
    blocks = {s: {(t.session, t.block) for t in ts} for s, ts in meta.items()}
    learn_ids = trial_ids["train"] | trial_ids["tune"]
    learn_blocks = blocks["train"] | blocks["tune"]

    # 1. Trials
    for a, b in (("train", "tune"), ("train", "val"), ("tune", "val"),
                 ("train", "test"), ("tune", "test"), ("val", "test")):
        shared = len(trial_ids[a] & trial_ids[b])
        checks.append((f"no trial is in both {a} and {b}", shared == 0, f"{shared} shared"))

    # 2. Blocks
    shared = len(blocks["val"] & learn_blocks)
    checks.append(("no recording block is shared between val and train/tune",
                   shared == 0, f"{shared} shared of {len(blocks['val'])} val blocks"))

    # 3. The recordings themselves, one session at a time to keep memory use low.
    print("Fingerprinting every recording (reads all the data; takes a few minutes)...")
    prints = {s: {} for s in SPLITS}  # fingerprint -> "session/key"
    for session in splits["sessions"]:
        for split in SPLITS:
            for t in load_trials(cfg, split, sessions=[session]):
                prints[split][fingerprint(t.features)] = f"{t.session}/{t.key}"
    for split in SPLITS:
        n = len(meta[split])
        checks.append((f"every {split} recording is unique within {split}",
                       len(prints[split]) == n, f"{len(prints[split])} fingerprints for {n} trials"))
    learn_prints = set(prints["train"]) | set(prints["tune"])
    for a, b, pa, pb in (("val", "train/tune", set(prints["val"]), learn_prints),
                         ("test", "train/tune", set(prints["test"]), learn_prints),
                         ("train", "tune", set(prints["train"]), set(prints["tune"])),
                         ("val", "test", set(prints["val"]), set(prints["test"]))):
        shared = len(pa & pb)
        checks.append((f"no identical recording in both {a} and {b}", shared == 0,
                       f"{shared} identical"))

    # 4. What the training script loads.
    source = (REPO_ROOT / "b2t" / "train.py").read_text()
    loaded = sorted(set(re.findall(r'load_trials\(\s*cfg\s*,\s*"(\w+)"', source)))
    checks.append(("the training script loads only train and tune",
                   set(loaded) <= {"train", "tune"}, f"it loads: {', '.join(loaded)}"))

    # 5. What each saved run was trained on.
    runs = []
    run_files = sorted(REPO_ROOT.glob("runs/*/run.json")) + sorted(REPO_ROOT.glob("results/*_run.json"))
    for path in run_files:
        run = json.loads(path.read_text())
        trained = run.get("trained_on_sessions", [])
        held_out_used = sorted(set(trained) & set(splits["holdout_sessions"]))
        runs.append({
            "file": str(path.relative_to(REPO_ROOT)),
            "model": run.get("model"),
            "n_sessions_trained_on": len(trained),
            "holdout_sessions_trained_on": held_out_used,
            "best_tune_per": run.get("best_tune_per"),
        })

    # Not leakage, but disclosed: val sentences whose wording was seen in training.
    seen_text = {normalize_text(t.sentence) for t in meta["train"] + meta["tune"]}
    repeated = [t for t in meta["val"] if normalize_text(t.sentence) in seen_text]

    passed = all(ok for _, ok, _ in checks)
    report = {
        "participant": cfg.participant,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": git("rev-parse", "HEAD"),
        "splits_file_first_committed": git("log", "--diff-filter=A", "--format=%cI %H", "--",
                                           str(cfg.splits_file.relative_to(REPO_ROOT))),
        "all_checks_passed": passed,
        "trial_counts": {s: len(ts) for s, ts in meta.items()},
        "checks": [{"check": name, "passed": ok, "detail": detail} for name, ok, detail in checks],
        "runs": runs,
        "val_sentences_with_wording_seen_in_training": {
            "count": len(repeated),
            "of": len(meta["val"]),
            "note": "Different recordings of the same text. Not leaked data, but easier; "
                    "report results on these separately.",
            "trials": [{"session": t.session, "block": t.block, "trial": t.trial,
                        "sentence": t.sentence} for t in repeated],
        },
    }

    print(f"\nParticipant {cfg.participant}: " +
          ", ".join(f"{len(ts)} {s}" for s, ts in meta.items()))
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    print("\nSaved runs:")
    for r in runs or [{"file": "(none found)"}]:
        if "model" in r:
            # Only the calibration-experiment run (Run B) is meant to leave out the
            # held-out sessions; every other run is meant to use all sessions.
            n_held = len(r["holdout_sessions_trained_on"])
            extra = (f", including the {n_held} calibration sessions (expected, unless "
                     f"this is the calibration run)" if n_held
                     else ", with all calibration sessions left out (the calibration run)")
            print(f"  {r['file']}: {r['model']}, trained on {r['n_sessions_trained_on']} sessions{extra}")
        else:
            print(f"  {r['file']}")
    print(f"\nDisclosed, not leakage: {len(repeated)} of {len(meta['val'])} val sentences "
          f"have wording that also appears in training.")

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"audit_{cfg.participant}.json"
    out_path.write_text(json.dumps(report, indent=1))
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}. Report saved to {out_path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main(sys.argv[1])
