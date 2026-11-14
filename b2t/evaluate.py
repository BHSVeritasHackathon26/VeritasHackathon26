"""Scores a trained model end to end: neural data -> phonemes -> words. Example:

    python -m b2t.evaluate --run runs/run_a --split tune
    python -m b2t.evaluate --run runs/run_a --split tune --llm --limit 100

It reports the phoneme error rate of the model, then the word error rate using the
dictionary decoder and, with --llm, the language-model decoder. Every prediction is
saved to a JSON file in results/ so it can be inspected and graphed later.

Use --split tune while developing. --split val is the final test: run it once, at
the end, after every setting has been decided.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from b2t.config import REPO_ROOT, load_config
from b2t.data import load_trials
from b2t.decode import DictionaryDecoder, count_words
from b2t.llm_decode import DEFAULT_MODEL, llm_decode
from b2t.metrics import normalize_text, phoneme_error_rate, word_error_rate
from b2t.phonemes import ids_to_phonemes
from b2t.predict import load_bundle
from b2t.train import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="bundle folder, e.g. runs/run_a")
    p.add_argument("--split", default="tune", choices=("tune", "val"))
    p.add_argument("--sessions", nargs="*", default=None, help="only these sessions")
    p.add_argument("--limit", type=int, default=None,
                   help="score a random sample of this many sentences")
    p.add_argument("--llm", action="store_true", help="also run the language-model decoder")
    p.add_argument("--llm-model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, run = load_bundle(args.run, device)
    cfg = load_config(Path(args.run) / "config.toml")

    trials = load_trials(cfg, args.split, sessions=args.sessions, load_features=False)
    if args.limit and args.limit < len(trials):
        trials = random.Random(args.seed).sample(trials, args.limit)
    # Load neural features only for the sessions we actually need.
    wanted = {(t.session, t.key) for t in trials}
    trials = [t for t in load_trials(cfg, args.split, sessions=sorted({t.session for t in trials}))
              if (t.session, t.key) in wanted]
    print(f"Scoring {len(trials)} {args.split} sentences with model from {args.run} on {device}")

    per, preds = evaluate(model, trials, device, batch_size=32)
    truths = [t.sentence for t in trials]

    # Word frequencies come from training sentences only.
    train = load_trials(cfg, "train", load_features=False)
    decoder = DictionaryDecoder(count_words(t.sentence for t in train))
    dict_sentences = [decoder.decode(ids) for ids in preds]
    dict_wer = word_error_rate(dict_sentences, truths)

    llm_sentences, llm_wer, fallbacks = None, None, 0
    if args.llm:
        llm_sentences, fallbacks = llm_decode(decoder, preds, model=args.llm_model)
        llm_wer = word_error_rate(llm_sentences, truths)

    print(f"\nPhoneme error rate (model alone)          {per:.1%}")
    print(f"Word error rate, dictionary decoder       {dict_wer:.1%}")
    if args.llm:
        print(f"Word error rate, language-model decoder   {llm_wer:.1%}"
              f"   ({fallbacks} sentences fell back to the dictionary)")

    print("\nExamples:")
    for i in range(min(5, len(trials))):
        print(f"  truth       {normalize_text(truths[i])}")
        print(f"  phonemes    {' '.join(ids_to_phonemes(preds[i]))}")
        print(f"  dictionary  {dict_sentences[i]}")
        if args.llm:
            print(f"  language    {llm_sentences[i]}")
        print()

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{Path(args.run).name}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump({
            "run": args.run,
            "split": args.split,
            "n_sentences": len(trials),
            "phoneme_error_rate": per,
            "word_error_rate_dictionary": dict_wer,
            "word_error_rate_llm": llm_wer,
            "llm_model": args.llm_model if args.llm else None,
            "llm_fallbacks": fallbacks,
            "sentences": [{
                "session": t.session, "block": t.block, "trial": t.trial,
                "truth": t.sentence,
                "true_phonemes": ids_to_phonemes(t.phoneme_ids),
                "predicted_phonemes": ids_to_phonemes(preds[i]),
                "dictionary": dict_sentences[i],
                "llm": llm_sentences[i] if args.llm else None,
            } for i, t in enumerate(trials)],
        }, f, indent=1)
    print(f"Saved every prediction to {out_path}")


if __name__ == "__main__":
    main()
