"""Tests the dictionary decoder without needing a trained model. Example:

    python -m b2t.decode_check configs/t15.toml

It takes the TRUE phonemes of tune sentences, damages them on purpose at several
error levels, decodes them to words, and reports the word error rate. This shows how
much word accuracy is lost for a given amount of phoneme error. The 0% row is the
decoder's own floor: mistakes it makes even from perfect phonemes.

The damage is random, which is not exactly how a real model errs, so treat these
numbers as a guide, not as results.
"""

import argparse
import random

from b2t.config import load_config
from b2t.data import load_trials
from b2t.decode import DictionaryDecoder, count_words
from b2t.metrics import normalize_text, phoneme_error_rate, word_error_rate
from b2t.phonemes import N_CLASSES, ids_to_phonemes


def corrupt(ids, rate, rng):
    """Randomly substitute, delete or insert phonemes at roughly the given rate."""
    out = []
    for p in ids:
        if rng.random() < rate:
            kind = rng.choice(("substitute", "delete", "insert"))
            if kind == "substitute":
                out.append(rng.randrange(1, N_CLASSES))
            elif kind == "insert":
                out.extend((int(p), rng.randrange(1, N_CLASSES)))
        else:
            out.append(int(p))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--n", type=int, default=300, help="how many tune sentences to use")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(args.seed)
    train = load_trials(cfg, "train", load_features=False)
    tune = load_trials(cfg, "tune", load_features=False)
    tune = rng.sample(tune, min(args.n, len(tune)))
    truths = [t.sentence for t in tune]

    decoder = DictionaryDecoder(count_words(t.sentence for t in train))

    # How many words in these sentences does the dictionary not contain at all?
    known = {w for words in decoder.words_by_pron.values() for w in words}
    words = [w for s in truths for w in normalize_text(s).split()]
    missing = [w for w in words if w not in known]
    print(f"{len(tune)} tune sentences, {len(words)} words, "
          f"{len(missing)} not in the dictionary ({len(missing) / len(words):.1%})")

    print("\nphoneme errors added | actual phoneme error rate | word error rate")
    examples = None
    for rate in (0.0, 0.05, 0.10, 0.20):
        noisy = [corrupt(t.phoneme_ids, rate, rng) for t in tune]
        decoded = [decoder.decode(ids) for ids in noisy]
        per = phoneme_error_rate(noisy, [t.phoneme_ids for t in tune])
        wer = word_error_rate(decoded, truths)
        print(f"{rate:20.0%} | {per:25.1%} | {wer:.1%}")
        if rate == 0.10:
            examples = list(zip(tune, noisy, decoded))[:4]

    print("\nExamples at 10% added phoneme errors:")
    for t, ids, decoded in examples:
        print(f"  truth    {normalize_text(t.sentence)}")
        print(f"  phonemes {' '.join(ids_to_phonemes(ids))}")
        print(f"  decoded  {decoded}\n")


if __name__ == "__main__":
    main()
