# Thought → Text demo app

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

The front page is a stack of boxes, one per stage:

1. **Input**: drag a `.hdf5` file onto the page (or click the box to browse).
2. **Phonemes**: the sounds the model decoded, as a row of pills (gaps = word breaks),
   with the name of the model underneath (`MODEL_NAME` in `model_adapter.py`).
3. **Words**: what happens between the sounds and the sentence:
   - **After the dictionary**: each group of sounds becomes the closest real word.
     This runs live for every file.
   - **After the language model**: Gemini picks the most sensible words in context.
4. **Sentence**: the final sentence, read aloud automatically (the speaker button
   replays it), with the **Attempted sentence** (what the participant was trying to say)
   underneath when it is known.

A file holds many sentences. The first one shows right away and the rest keep
decoding behind it; use the ‹ › buttons (or the arrow keys) in the Input box to
step through them.

Box 2 shows the `phonemes` that `predict()` returns (a string such as `AY | N OW`, or a
list; `|` is a word break). The dictionary sentence is the `text` that `predict()` returns.

The language-model step is **not run live**: it needs a Gemini API key and internet, so
`model_adapter.py` skips it. Instead the page shows the sentence saved for that recording in
`results/run_a_val.json` (the held-out sentences), but only when this recording's phonemes
are identical to the saved ones. Any other file, such as a brand-new test file, shows
"Not run for this file" in that row, and the final sentence is the dictionary sentence
(the box says which one it is).

When the sentence the participant was trying to say is known, it appears under the
final sentence as a grey **Attempted sentence**. It comes from the file's own label if it
has one (`sentence_label` or `transcription`), otherwise from the saved held-out
results in `results/run_a_val.json` (matched by session, block and trial). Unlabeled
test files simply show nothing there. The target is for display only: it is never
given to the model, and there is no accuracy scoring.

## Browsing the held-out sentences

When the repo's `results/run_a_val.json` and the `hdf5_data_final/` data folder are
present, the explorer page at `/explorer` (for example http://localhost:8000/explorer) lets you
browse the 1,426 held-out (val) sentences: pick a session and sentence, or press **Random sentence**. For each one
it shows the neural recording, what the participant was trying to say, the phonemes
the model produced, and the sentence from the dictionary decoder and from the
language-model decoder, with wrong words highlighted and error counts.

These predictions were saved earlier by `python -m b2t.evaluate`, so this part works
on a computer that does not have the trained model. The page says so. Start the
server from the repo root so it can find those files:

```bash
python app/server.py
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
