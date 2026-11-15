"""Keeps the model's uncertainty instead of throwing it away.

At every output step the model gives a probability for each of the 41 classes.
Normally we keep only the most likely class at each step ("greedy decoding"), so
if the model was 55% sure of D and 40% sure of T, everything after it only sees D.

This module turns those step-by-step probabilities into a short list of "slots",
one per sound the model heard, each with its runner-up choices and how confident
the model was. A language model can then use context to choose between them.

Sending every raw step would be far too long (most steps are "blank", meaning
nothing new) and hard to reason about, so steps are merged the same way greedy
decoding merges them, and alternatives are attached to each merged sound.
"""

from b2t.phonemes import BLANK_ID, PHONEMES


def name(class_id):
    """How a class is written in the prompt."""
    return "nothing" if class_id == BLANK_ID else PHONEMES[class_id]


def slots_from_probabilities(probs, min_prob=0.1, max_alternatives=2, missed_prob=0.2):
    """probs: array of shape (steps, 41), each row summing to 1.

    Returns a list of slots. Each slot is a list of (class id, probability),
    most likely first:
      - a sound the model heard, with any runner-up above min_prob (a runner-up of
        "nothing" means the sound may not really be there);
      - or a "maybe missed" slot, [nothing, some phoneme], where the model said
        nothing but a phoneme came close (above missed_prob).
    """
    best = probs.argmax(axis=1)
    slots, start = [], 0
    while start < len(best):
        cls = int(best[start])
        end = start
        while end < len(best) and best[end] == cls:
            end += 1
        run = probs[start:end]  # the steps where the model kept giving this class

        if cls != BLANK_ID:
            frame = run[run[:, cls].argmax()]  # the step where it was most sure
            others = sorted(((float(frame[k]), k) for k in range(len(frame)) if k != cls), reverse=True)
            slot = [(cls, float(frame[cls]))]
            slot += [(k, p) for p, k in others if p >= min_prob][:max_alternatives]
            slots.append(slot)
        else:
            # A stretch of "nothing": did any phoneme nearly break through?
            heard = run.copy()
            heard[:, BLANK_ID] = 0.0
            step, k = divmod(int(heard.argmax()), heard.shape[1])
            # A phoneme's probability rises just before it is given and falls just
            # after, so a near-miss of the SAME phoneme next door is not a second sound.
            before = int(best[start - 1]) if start > 0 else None
            after = int(best[end]) if end < len(best) else None
            if heard[step, k] >= missed_prob and k not in (before, after):
                slots.append([(BLANK_ID, float(run[step, BLANK_ID])), (k, float(heard[step, k]))])
        start = end
    return slots


def greedy_ids(slots):
    """The plain greedy phoneme sequence these slots came from."""
    return [slot[0][0] for slot in slots if slot[0][0] != BLANK_ID]


def describe(slots):
    """One line of text for the prompt.

    A confident sound is written plainly:           K
    An uncertain one lists its options and odds:    [D .55, T .40]
    A sound that may not be there:                  [L .48, nothing .41]
    A possibly missed sound:                        [nothing .70, T .25]
    """
    parts = []
    for slot in slots:
        cls, p = slot[0]
        if len(slot) == 1:
            parts.append(name(cls))
        else:
            parts.append("[" + ", ".join(f"{name(k)} {q:.2f}".replace(" 0.", " .") for k, q in slot) + "]")
    return " ".join(parts)
