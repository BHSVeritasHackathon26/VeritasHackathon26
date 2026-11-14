"""The language-model decoder: uses a large language model to choose words in context.

The dictionary decoder judges each word alone, so it cannot tell that "i know this"
is more likely than "i dau this". Here we give a language model, for every word
slot, the phonemes the neural network produced plus the dictionary's closest
candidate words, and ask it for the most likely sentence.

The language model never sees the true sentence. It is general-purpose and not
trained on any participant, so this stage is reused unchanged for a new person.

Needs a Gemini API key in the GEMINI_API_KEY environment variable.
"""

import hashlib
import json
import time
from pathlib import Path

from b2t.config import REPO_ROOT
from b2t.metrics import normalize_text
from b2t.phonemes import split_into_words

DEFAULT_MODEL = "gemini-4-hydrogen"
CACHE_DIR = REPO_ROOT / "cache" / "llm"

INSTRUCTIONS = """\
You are part of a brain-computer interface that helps a person who cannot speak.
A neural network turned their brain activity into phonemes (ARPAbet symbols), and it
makes some mistakes. Your job is to work out the sentence they were trying to say.

Each numbered sentence below is a list of word slots, in order. Each slot shows the
phonemes the network produced, then candidate dictionary words with the number of
phoneme edits needed to match (0 = exact match).

Rules:
- Normally pick one candidate per slot, choosing the words that make the most
  sensible sentence together. Prefer candidates with fewer edits when context
  does not decide.
- You may use a word that is not listed if it fits the phonemes better.
- Word boundaries are sometimes wrong: you may merge two slots into one word or
  split one slot into two words when the phonemes support it.
- Do not add, drop or reorder words to improve grammar. Stay faithful to the phonemes.
- Write each sentence in lower case with no punctuation except apostrophes.

Return one decoded sentence for every numbered sentence, in the same order.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"sentences": {"type": "array", "items": {"type": "string"}}},
    "required": ["sentences"],
}


def describe_sentence(decoder, phoneme_ids, k=5):
    """One sentence's word slots as text for the prompt."""
    lines = []
    for chunk in split_into_words(phoneme_ids):
        options = ", ".join(f"{w} ({d})" for w, d in decoder.candidates(chunk, k=k))
        lines.append(f"  [{' '.join(chunk)}] -> {options}")
    return "\n".join(lines)


def build_prompt(decoder, batch_of_ids, k=5):
    parts = [INSTRUCTIONS]
    for i, ids in enumerate(batch_of_ids, start=1):
        parts.append(f"Sentence {i}:\n{describe_sentence(decoder, ids, k)}")
    return "\n\n".join(parts)


def ask_gemini(prompt, model=DEFAULT_MODEL):
    """Send one prompt to Gemini and return its list of sentences."""
    from google import genai  # imported here so the rest of the project works without it

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    interaction = client.interactions.create(
        model=model,
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": RESPONSE_SCHEMA,
        },
    )
    return json.loads(interaction.output_text)["sentences"]


def ask_with_cache(prompt, model, ask, retries=4):
    """Ask the model, saving each answer to disk so a re-run costs no extra requests."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()[:24]
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text()), True

    for attempt in range(retries):
        try:
            sentences = ask(prompt, model)
            path.write_text(json.dumps(sentences))
            return sentences, False
        except Exception as error:  # rate limit or network trouble: wait and retry
            wait = 15 * (attempt + 1)
            print(f"  request failed ({type(error).__name__}: {error}); retrying in {wait}s")
            time.sleep(wait)
    return None, False


def llm_decode(decoder, all_ids, model=DEFAULT_MODEL, batch_size=20, pause=4.0,
               ask=ask_gemini):
    """Decode many sentences. Returns (sentences, number that fell back).

    Sentences are sent in batches to stay inside free-tier request limits. If the
    model fails or returns the wrong number of sentences for a batch, that batch
    falls back to the dictionary decoder's answer, and we count how often.
    """
    results, fallbacks = [], 0
    for start in range(0, len(all_ids), batch_size):
        batch = all_ids[start : start + batch_size]
        answer, was_cached = ask_with_cache(build_prompt(decoder, batch), model, ask)
        if answer is None or len(answer) != len(batch):
            answer = [decoder.decode(ids) for ids in batch]
            fallbacks += len(batch)
        results.extend(normalize_text(s) for s in answer)
        print(f"  language model: {len(results)}/{len(all_ids)} sentences")
        if not was_cached:
            time.sleep(pause)  # stay under the requests-per-minute limit
    return results, fallbacks
