"""Make a small fake HDF5 file in the Brain-to-Text '25 layout (unlabeled, like the
test split), for trying the UI.

    python tools/make_sample_hdf5.py sample.hdf5

The neural data is random noise -- only useful for demo mode / UI testing.
"""
import sys

import h5py
import numpy as np

SENTENCES = [
    "I would like a glass of water please",
    "The weather is beautiful today",
    "Can you turn the lights off",
    "Thank you for coming to see me",
    "What time does the game start tonight",
    "I am feeling much better this morning",
]

out = sys.argv[1] if len(sys.argv) > 1 else "sample.hdf5"
rng = np.random.default_rng(0)
with h5py.File(out, "w") as f:
    for i, s in enumerate(SENTENCES):
        g = f.create_group(f"trial_{i:04d}")
        T = int(rng.integers(150, 400))
        t = np.linspace(0, 6, T)[:, None]
        x = rng.normal(size=(T, 512)) + 2 * np.sin(t * rng.uniform(1, 4, size=(1, 512)))
        g.create_dataset("input_features", data=x.astype(np.float32))
        g.attrs["n_time_steps"] = T
        g.attrs["session"] = "t15.2023.08.13"
        g.attrs["block_num"] = 1
        g.attrs["trial_num"] = i
print("wrote", out)
