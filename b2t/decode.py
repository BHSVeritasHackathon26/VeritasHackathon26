"""Turns a phoneme sequence into English words using a pronunciation dictionary.

The model outputs phonemes with a silence marker between words. This decoder cuts
the sequence at each marker and, for every chunk, finds the dictionary word whose
pronunciation is closest. It looks at one word at a time and knows nothing about
grammar or meaning; that is the job of a language model, added on top later.

Pronunciations come from the CMU Pronouncing Dictionary (about 126,000 words),
which uses the same 39 phonemes as our labels.
"""

import re
from collections import Counter, defaultdict

import cmudict

from b2t.metrics import normalize_text
from b2t.phonemes import split_into_words

WORD_PATTERN = re.compile(r"[a-z']+")


def count_words(sentences):
    """How often each word appears in a list of sentences (use training data only)."""
    counts = Counter()
    for s in sentences:
        counts.update(normalize_text(s).split())
    return counts


def bounded_distance(a, b, limit):
    """Edit distance between a and b, or limit + 1 as soon as it must exceed limit.

    Giving up early makes searching the whole dictionary fast.
    """
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def one_removed(seq):
    """Every version of seq with exactly one item taken out."""
    return {seq[:i] + seq[i + 1:] for i in range(len(seq))}


class DictionaryDecoder:
    def __init__(self, word_counts=None):
        """word_counts: how common each word is, used only to break ties between
        words that sound equally close (for example "to", "too" and "two")."""
        self.word_counts = word_counts or {}
        # pronunciation (tuple of phonemes) -> every word pronounced that way
        self.words_by_pron = defaultdict(list)
        for word, prons in cmudict.dict().items():
            if not WORD_PATTERN.fullmatch(word):
                continue
            for pron in prons:
                # The dictionary marks stress with a digit ("AE1"); our labels do not.
                key = tuple(p.rstrip("012") for p in pron)
                if word not in self.words_by_pron[key]:
                    self.words_by_pron[key].append(word)
        # Group pronunciations by length so a search can skip ones that are far
        # too long or too short to be close.
        self.prons_by_length = defaultdict(list)
        for pron in self.words_by_pron:
            self.prons_by_length[len(pron)].append(pron)
        # Shortcut for finding pronunciations one edit away, the most common case.
        # Two sequences are one edit apart only if they are equal after removing at
        # most one phoneme from each, so we file every pronunciation under itself
        # and under each "one removed" version, and later just look those up.
        self.near_index = defaultdict(list)
        for pron in self.words_by_pron:
            for key in one_removed(pron) | {pron}:
                self.near_index[key].append(pron)
        self._cache = {}

    def candidates(self, chunk, k=5):
        """The k best words for one chunk of phonemes, as (word, distance) pairs.

        distance = number of phoneme edits between the chunk and the word's
        pronunciation; 0 means an exact match. Closest first, then most common first.
        """
        chunk = tuple(chunk)
        if (chunk, k) in self._cache:
            return self._cache[(chunk, k)]

        scored = []
        if k == 1 and chunk in self.words_by_pron:
            # Exact match and only one word wanted: no need to search.
            scored = [(0, w) for w in self.words_by_pron[chunk]]
        if not scored:
            # Everything within one edit, found through the shortcut index.
            nearby = set()
            for key in one_removed(chunk) | {chunk}:
                nearby.update(self.near_index.get(key, ()))
            for pron in nearby:
                d = bounded_distance(chunk, pron, 1)
                if d <= 1:
                    scored.extend((d, w) for w in self.words_by_pron[pron])
        # Still nothing: search the dictionary, allowing more and more mistakes.
        for limit in (2, 3, len(chunk)):
            if scored:
                break
            for length in range(len(chunk) - limit, len(chunk) + limit + 1):
                for pron in self.prons_by_length.get(length, ()):
                    d = bounded_distance(chunk, pron, limit)
                    if d <= limit:
                        scored.extend((d, w) for w in self.words_by_pron[pron])

        scored.sort(key=lambda dw: (dw[0], -self.word_counts.get(dw[1], 0), dw[1]))
        result, seen = [], set()
        for d, w in scored:
            if w not in seen:
                seen.add(w)
                result.append((w, d))
            if len(result) == k:
                break
        self._cache[(chunk, k)] = result
        return result

    def decode(self, phoneme_ids):
        """Phoneme ids for a whole sentence -> a sentence of best-guess words."""
        words = []
        for chunk in split_into_words(phoneme_ids):
            best = self.candidates(chunk, k=1)
            if best:
                words.append(best[0][0])
        return " ".join(words)
