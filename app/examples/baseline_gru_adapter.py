"""
STARTING POINT, NOT TESTED -- adapt it to your own training code.

For teams who trained with the Brain-to-Text '25 baseline repo (GRUDecoder in
model_training/rnn_model.py). Written from memory of that repo without a
checkpoint to run against, so check every name against your code: argument
names, checkpoint keys, the smoothing step and the phoneme table.

The baseline GRU outputs PHONEMES. To get a sentence you still need your own
phoneme-to-text step (the baseline uses an n-gram language model); plug it in
at the marked line.

To use: copy load_model() and predict() into model_adapter.py and set the paths.
"""
import sys

import numpy as np
import torch
import yaml

REPO_DIR = "/path/to/your/repo/model_training"       # folder containing rnn_model.py
MODEL_DIR = "/path/to/trained_model"                 # contains checkpoint/ with args.yaml + best_checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PHONEMES = ["BLANK", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G",
            "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
            "UH", "UW", "V", "W", "Y", "Z", "ZH", " | "]


def load_model():
    sys.path.insert(0, REPO_DIR)
    from rnn_model import GRUDecoder

    with open(f"{MODEL_DIR}/checkpoint/args.yaml") as fh:
        args = yaml.safe_load(fh)
    m = args["model"]
    net = GRUDecoder(
        neural_dim=m["n_input_features"], n_units=m["n_units"], n_days=len(args["dataset"]["sessions"]),
        n_classes=args["dataset"]["n_classes"], rnn_dropout=m["rnn_dropout"], input_dropout=m["input_network"]["input_layer_dropout"],
        n_layers=m["n_layers"], patch_size=m["patch_size"], patch_stride=m["patch_stride"],
    )
    ckpt = torch.load(f"{MODEL_DIR}/checkpoint/best_checkpoint", map_location=DEVICE, weights_only=False)
    state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    net.load_state_dict(state)
    net.to(DEVICE).eval()
    return {"net": net, "args": args}


def predict(model, features, info):
    net, args = model["net"], model["args"]
    day = args["dataset"]["sessions"].index(info["session"])  # per-day input layer

    x = torch.from_numpy(features).unsqueeze(0).to(DEVICE)   # (1, T, 512)
    # NOTE: the baseline Gaussian-smooths the input before the model (gauss_smooth in
    # data_augmentations.py, using smooth_kernel_std / smooth_kernel_size from args).
    # Apply the same preprocessing you used at evaluation time here.
    with torch.no_grad():
        logits = net(x, torch.tensor([day], device=DEVICE))  # (1, T', n_classes)

    ids = logits[0].argmax(-1).cpu().numpy()
    ids = ids[np.insert(np.diff(ids) != 0, 0, True)]         # collapse repeats (CTC)
    phonemes = " ".join(PHONEMES[i] for i in ids if i != 0)  # drop blanks

    text = phonemes  # <-- replace with your phoneme-to-sentence step (language model)
    return {"text": text, "phonemes": phonemes}
