"""
model_adapter.py  --  connects the trained model to the demo server.

The server calls two functions:

    load_model()                    -> called ONCE when the server starts
    predict(model, features, info)  -> called once per trial

This version loads our trained GRU ("Run A") from its output bundle folder,
runs/run_a/, which holds model.pt (weights), run.json (settings) and config.toml.
That folder is too large for GitHub, so it has to be copied onto the computer by
hand. If it is not there, load_model() returns None and the app runs in DEMO MODE.

What predict() does, using the code in the b2t package:
    1. the GRU turns the neural recording into phonemes        (b2t.models)
    2. repeated phonemes are merged and blanks dropped          (b2t.train.greedy_decode)
    3. the phonemes are looked up in a pronunciation dictionary (b2t.decode)

The language-model decoder (Gemini) is not run here: it needs an API key and an
internet connection, and has a daily request limit.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_DIR = REPO / "runs" / "run_a"

# Shown in the app's status pill.
MODEL_NAME = "GRU (Run A) + dictionary decoder"


def load_model():
    """Load the trained model once. Returns None (demo mode) if it is not on this computer."""
    if not (RUN_DIR / "model.pt").exists():
        print(f"  No trained model at {RUN_DIR}; starting in demo mode.")
        return None

    sys.path.insert(0, str(REPO))  # so the b2t package can be imported
    import torch

    from b2t.config import load_config
    from b2t.data import load_trials
    from b2t.decode import DictionaryDecoder, count_words
    from b2t.predict import load_bundle

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, run = load_bundle(RUN_DIR, device)  # also switches the model to eval mode

    # Word frequencies (used only to break ties between sound-alike words) come
    # from the training sentences. Without the data folder, the decoder still works.
    try:
        cfg = load_config(RUN_DIR / "config.toml")
        counts = count_words(t.sentence for t in load_trials(cfg, "train", load_features=False))
    except Exception as e:
        print(f"  Training sentences not available ({type(e).__name__}); decoding without word frequencies.")
        counts = None

    return {"net": net, "run": run, "device": device, "decoder": DictionaryDecoder(counts)}


def predict(model, features, info):
    """Turn one trial of neural data into a sentence.

    model     what load_model() returned
    features  numpy float32 array, shape (time_steps, n_features)
    info      dict with "session" (e.g. "t15.2023.08.13"), "block_num", "trial_num"

    Returns {"text": sentence, "phonemes": "AY | N OW | ...", "phoneme_list": [...]}.
    """
    import torch

    from b2t.phonemes import ids_to_phonemes
    from b2t.train import greedy_decode

    net, run, device = model["net"], model["run"], model["device"]

    # Each recording session has its own input layer, so we need to know which
    # session this trial came from. If the file does not say, or it is a session
    # the model never saw, fall back to the most recent session's layer.
    sessions = run["all_sessions"]
    session = info.get("session")
    session_idx = sessions.index(session) if session in sessions else len(sessions) - 1

    x = torch.from_numpy(features).unsqueeze(0).to(device)           # (1, time, features)
    lengths = torch.tensor([features.shape[0]], device=device)
    with torch.no_grad():
        logits, out_lengths = net(x, lengths, torch.tensor([session_idx], device=device))
    ids = greedy_decode(logits, out_lengths)[0]

    phoneme_list = ids_to_phonemes(ids)
    return {"text": model["decoder"].decode(ids), "phonemes": " ".join(phoneme_list),
            "phoneme_list": phoneme_list}
