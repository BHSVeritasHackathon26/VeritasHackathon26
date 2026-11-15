# Thought → Text demo app

> **This is the earlier version of the demo app**, kept as it was in commit `b082bbc`
> ("Connect the trained model to the app; decode held-out sentences live"): the dark
> page with the recording preview, the result list and the held-out sentence explorer.
> The newer, simpler page lives in `app/`. Run this one from the repo root with
> `python app_v2/server.py`. To run both at once, give this one another port:
> `PORT=8001 python app_v2/server.py`.

Drop an HDF5 file of neural recordings into a web page; the trained model decodes
every trial in it and the page shows (and reads aloud) the predicted sentences.

The app has two parts that live together in this folder:

- `server.py` runs **on the computer that has the trained model**. It loads the
  model once, accepts HDF5 uploads and streams predictions back.
- `static/index.html` is the page. The server serves it, so there is nothing to
  install on the computers that only *use* the app: they just open a link.

## 1. On the computer with the model (one time)

Copy this folder there, then, in the same Python environment used for training:

```bash
pip install -r requirements.txt
```

Open `model_adapter.py` and fill in the two functions:

- `load_model()` loads the checkpoint and returns the model.
- `predict(model, features, info)` gets one trial's `input_features` array
  (time × 512) and returns the sentence.

If you trained with the Brain-to-Text '25 baseline GRU, `examples/baseline_gru_adapter.py`
is a starting point (untested; check it against your code).

Until `load_model()` returns a model the app runs in clearly labelled **demo mode**.

## 2. Start it

```bash
python server.py
```

It prints two links. Use the `localhost` one on that computer, or the
"same Wi-Fi" one from any other laptop or phone on the same network.

Not on the same network (or the Wi-Fi blocks device-to-device traffic, as many
school and event networks do)? Run a tunnel on the model computer and share the
link it prints, for example `cloudflared tunnel --url http://localhost:8000`.
Anyone with a link can use the app, so stop the server when you are done.

## 3. Use it

Drag a `.hdf5` file anywhere onto the page. Results appear one trial at a time as
the model finishes them. Click a row (or use ↑ / ↓) to show it large and hear it.

- The app is built for the unlabeled test files: it never reads sentence labels
  and has no accuracy scoring.
- "Read aloud" uses the browser's built-in voices; pick the voice and speed
  under the result.

## Browsing the held-out sentences

When the repo's `results/run_a_val_gemini31.json` and the `hdf5_data_final/` data folder are
present, the top of the page also lets you browse the 1,426 held-out (val)
sentences: pick a session and sentence, or press **Random sentence**. For each one
it shows the neural recording, what the participant was trying to say, the phonemes
the model produced, and the sentence from the dictionary decoder and from the
language-model decoder, with wrong words highlighted and error counts.

These predictions were saved earlier by `python -m b2t.evaluate`, so this part works
on a computer that does not have the trained model. The page says so. Start the
server from the repo root so it can find those files:

```bash
python app_v2/server.py
```

## It cannot train the model

The server only runs `predict()`. It contains no training code, runs the model
with gradients switched off (`torch.inference_mode()`), and never writes to the
checkpoint. Each uploaded file is a temporary copy that is deleted the moment
decoding ends, so dropped files never land in your training data folder.

## Trying the UI without real data

```bash
python tools/make_sample_hdf5.py sample.hdf5
```

makes a small fake file in the same layout (random noise, for demo mode only).

## Expected file layout

One group per trial (`trial_0000`, …), each with an `input_features` dataset.
Optional per-trial attributes: `session`, `block_num`, `trial_num`.
This is the Brain-to-Text '25 format.
