"""Compares three ways of turning the model's phonemes into a sentence with a
language model, on the same sentences. Example:

    python -m b2t.llm_compare --run runs/run_a --split tune --llm-model gemini-3.1-flash-lite

    candidates    today's method: for each word, the phonemes plus the dictionary's
                  five closest words
    phonemes      no dictionary: just the phoneme sequence
    alternatives  no dictionary: the phonemes with the model's runner-up choices
                  and how confident it was at each one (see b2t/alternatives.py)

All three use the same language model, so the only difference is what it is shown.
The dictionary decoder with no language model is reported too, as the reference.

Use --split tune to decide which is best. Score val only for the one you pick.

Results are printed and saved to results/llm_compare_<run>_<split>.json.
Needs GEMINI_API_KEY. Each method costs about one request per 10 sentences;
answers are cached, so re-running costs nothing.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from b2t.alternatives import describe, greedy_ids, slots_from_probabilities
from b2t.batching import make_batch
from b2t.config import REPO_ROOT, load_config
from b2t.data import load_trials
from b2t.decode import DictionaryDecoder, count_words
from b2t.llm_decode import (DEFAULT_MODEL, INSTRUCTIONS_ALTERNATIVES, INSTRUCTIONS_PHONEMES,
                            ask_gemini, llm_decode, llm_decode_lines)
from b2t.metrics import edit_distance, normalize_text, phoneme_error_rate, word_error_rate
from b2t.phonemes import ids_to_phonemes
from b2t.predict import load_bundle

METHODS = ("candidates", "phonemes", "alternatives")


@torch.no_grad()
def class_probabilities(model, trials, device, batch_size=32):
    """The model's probability for every class at every output step, per trial."""
    model.eval()
    order = sorted(range(len(trials)), key=lambda i: trials[i].n_steps)
    out = [None] * len(trials)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = make_batch([trials[i] for i in idx], device)
        logits, out_lengths = model(batch["x"], batch["lengths"], batch["session_idx"])
        probs = logits.float().softmax(-1).cpu().numpy()
        for row, i in enumerate(idx):
            out[i] = probs[row, : int(out_lengths[row])]
    return out


def main(ask=ask_gemini):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="tune", choices=("tune", "val"))
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--llm-model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=None, help="use a random sample of this many sentences")
    p.add_argument("--min-prob", type=float, default=0.1,
                   help="alternatives: smallest probability worth showing as a runner-up")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, run = load_bundle(args.run, device)
    cfg = load_config(Path(args.run) / "config.toml")

    trials = load_trials(cfg, args.split, load_features=False)
    if args.limit and args.limit < len(trials):
        trials = random.Random(args.seed).sample(trials, args.limit)
    wanted = {(t.session, t.key) for t in trials}
    trials = [t for t in load_trials(cfg, args.split, sessions=sorted({t.session for t in trials}))
              if (t.session, t.key) in wanted]
    truths = [t.sentence for t in trials]
    print(f"{len(trials)} {args.split} sentences, model from {args.run} on {device}, "
          f"language model {args.llm_model}")

    slots = [slots_from_probabilities(pr, min_prob=args.min_prob)
             for pr in class_probabilities(model, trials, device)]
    ids = [greedy_ids(s) for s in slots]
    per = phoneme_error_rate(ids, [t.phoneme_ids for t in trials])
    uncertain = sum(sum(len(slot) > 1 for slot in s) for s in slots) / sum(len(s) for s in slots)
    print(f"Phoneme error rate {per:.1%}; {uncertain:.0%} of sounds come with a runner-up")

    decoder = DictionaryDecoder(count_words(t.sentence for t in load_trials(cfg, "train", load_features=False)))
    dictionary = [decoder.decode(i) for i in ids]

    sentences, fallbacks = {"dictionary": dictionary}, {"dictionary": 0}
    for method in args.methods:
        print(f"\nMethod: {method}")
        if method == "candidates":
            sentences[method], fallbacks[method] = llm_decode(decoder, ids, model=args.llm_model, ask=ask)
        elif method == "phonemes":
            lines = [" ".join(ids_to_phonemes(i)) for i in ids]
            sentences[method], fallbacks[method] = llm_decode_lines(
                lines, INSTRUCTIONS_PHONEMES, dictionary, model=args.llm_model, ask=ask, label=method)
        else:
            lines = [describe(s) for s in slots]
            sentences[method], fallbacks[method] = llm_decode_lines(
                lines, INSTRUCTIONS_ALTERNATIVES, dictionary, model=args.llm_model, ask=ask, label=method)

    print(f"\nWord error rate on {len(trials)} {args.split} sentences")
    summary = {}
    for method, out in sentences.items():
        wer = word_error_rate(out, truths)
        perfect = sum(edit_distance(normalize_text(o).split(), normalize_text(t).split()) == 0
                      for o, t in zip(out, truths))
        summary[method] = {"word_error_rate": wer, "sentences_all_correct": perfect,
                           "fell_back_to_dictionary": fallbacks[method]}
        note = f"   ({fallbacks[method]} fell back to the dictionary)" if fallbacks[method] else ""
        label = "dictionary only, no language model" if method == "dictionary" else method
        print(f"  {label:36s} {wer:6.1%}   {perfect} sentences fully correct{note}")

    print("\nExamples:")
    for i in range(min(4, len(trials))):
        print(f"  truth         {normalize_text(truths[i])}")
        print(f"  shown as      {describe(slots[i])}")
        for method, out in sentences.items():
            print(f"  {method:13s} {out[i]}")
        print()

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"llm_compare_{Path(args.run).name}_{args.split}.json"
    out_path.write_text(json.dumps({
        "run": args.run, "split": args.split, "n_sentences": len(trials),
        "llm_model": args.llm_model, "min_prob": args.min_prob,
        "phoneme_error_rate": per, "summary": summary,
        "sentences": [{"session": t.session, "block": t.block, "trial": t.trial, "truth": t.sentence,
                       "predicted_phonemes": ids_to_phonemes(ids[i]),
                       "with_alternatives": describe(slots[i]),
                       **{m: out[i] for m, out in sentences.items()}}
                      for i, t in enumerate(trials)],
    }, indent=1))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
