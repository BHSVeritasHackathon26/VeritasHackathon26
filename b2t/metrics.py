"""Error rates. Both are "edit distance" scores: the number of substitutions,
insertions and deletions needed to turn the prediction into the truth, divided by
the length of the truth. 0% is perfect; it can go above 100%.

    phoneme error rate (PER) - scores the neural network on its own
    word error rate (WER)    - scores the whole system, neural data to sentence

Rates are aggregated the same way as the Brain-to-Text '25 benchmark: total edits
over all sentences divided by total true length, not an average of per-sentence rates.
"""

import re


def edit_distance(pred, truth):
    """Fewest single-item edits to turn the sequence `pred` into `truth`."""
    # prev[j] = distance between the part of pred seen so far and truth[:j]
    prev = list(range(len(truth) + 1))
    for i, p in enumerate(pred, start=1):
        cur = [i]
        for j, t in enumerate(truth, start=1):
            cur.append(min(
                prev[j] + 1,              # delete p
                cur[j - 1] + 1,           # insert t
                prev[j - 1] + (p != t),   # substitute (free if they match)
            ))
        prev = cur
    return prev[-1]


def error_rate(preds, truths):
    """Aggregate error rate over a list of (prediction, truth) sequences."""
    if len(preds) != len(truths):
        raise ValueError("preds and truths must be the same length")
    edits = sum(edit_distance(p, t) for p, t in zip(preds, truths))
    total = sum(len(t) for t in truths)
    return edits / total


def normalize_text(sentence):
    """Lower-case and strip punctuation, keeping apostrophes ("don't" stays one word).

    Hyphens become spaces. The benchmark does not score punctuation.
    """
    sentence = sentence.lower().replace("-", " ")
    sentence = re.sub(r"[^a-z0-9' ]", "", sentence)
    return " ".join(sentence.split())


def phoneme_error_rate(pred_ids, true_ids):
    """pred_ids, true_ids: lists of phoneme id sequences, one per sentence."""
    return error_rate([list(p) for p in pred_ids], [list(t) for t in true_ids])


def word_error_rate(pred_sentences, true_sentences):
    """pred_sentences, true_sentences: lists of strings, one per sentence."""
    return error_rate(
        [normalize_text(s).split() for s in pred_sentences],
        [normalize_text(s).split() for s in true_sentences],
    )
