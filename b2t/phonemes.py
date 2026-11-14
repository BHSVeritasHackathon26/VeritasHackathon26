"""The phoneme vocabulary. This is the same for every English speaker, so it is
shared code, not part of a participant's config.

The model outputs one of these 41 classes at each time step. Index 0 is the CTC
"blank" (meaning "no new phoneme here"), and index 40 is the silence between words.
Order comes from the Brain-to-Text '25 dataset description.
"""

PHONEMES = [
    "BLANK",
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH",
    "|",
]

BLANK_ID = 0
SIL_ID = PHONEMES.index("|")
N_CLASSES = len(PHONEMES)


def ids_to_phonemes(ids):
    """[36, 17, 8, 40] -> ["W", "IH", "CH", "|"]"""
    return [PHONEMES[int(i)] for i in ids]


def split_into_words(ids):
    """Split a phoneme id sequence at the silence token, one list per word.

    [36, 17, 8, 40, 17, 38, 40] -> [["W", "IH", "CH"], ["IH", "Z"]]
    """
    words, current = [], []
    for i in ids:
        if int(i) == SIL_ID:
            if current:
                words.append(current)
            current = []
        else:
            current.append(PHONEMES[int(i)])
    if current:
        words.append(current)
    return words
