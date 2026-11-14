# Lab notebook

A running record of what we did, what we measured and what we decided. Newest entries
go at the bottom. This feeds the write-up (results, what didn't work, sources).

## Saturday, November 14

### Project decision
- Goal: decode attempted speech from neural recordings into English text, for people
  with motor speech disabilities.
- Track: TODO (leaning AI + Languages: "help people ... be understood").
- User: TODO, name one specific person and the moment they would use it.

### Data
- Brain-to-Text '25 dataset, participant T15, 45 recording sessions (Aug 2023 to Apr 2025).
- Each trial: neural features of shape (time, 512), one row per 20 ms, plus the phoneme
  sequence and sentence the participant was attempting to say.
- No cleaning was needed; features arrive already normalised.

### Splits (frozen in `splits/t15.json`)
| Split | Trials | Role |
|---|---|---|
| train | 7,264 | model learns from these |
| tune | 808 | early stopping and choosing settings (random 10% of each session's official train trials, seed 0) |
| val | 1,426 | final held-out test; not used by any code so far |
| test | 1,450 | no labels; demo inputs only |

- The training script loads only train and tune.
- 162 of the 1,426 val sentences have the same wording as a training sentence
  (different recordings). To be reported separately.
- Held-out sessions for the calibration experiment: the six 2025 sessions, which
  have 173 val trials between them.

### Results so far
| Run | Model | Sessions | Batches | Tune phoneme error rate | Notes |
|---|---|---|---|---|---|
| linear | one linear layer + CTC, 293,929 parameters | all 45 | 5,000 | 61.1% | simple baseline; 14.1 batches/s |
| run_a | GRU, 5 layers x 768 units, 44.3M parameters | all 45 | 120,000 at batch size 32 | TODO | first attempt (60,000 at batch size 64, 3.4-3.5 batches/s) ran out of memory; see below |
| run_b | same GRU | 39 (2025 held out) | 40,000 | TODO | for the calibration experiment |
| run_a_long | same GRU | all 45 | 120,000 | TODO | the reference recipe's full length |

Reference point: the dataset authors report 10.1% phoneme error rate on val for their
baseline, trained for 120,000 batches.

### Decisions and why
- Kept the dataset's official train/val split instead of re-splitting 80/20, so our
  final score is comparable with the published one.
- Predict phonemes, not words: the recorded brain area controls speech movements,
  and 41 phoneme classes are far easier to learn than a whole vocabulary.
- Model design follows the dataset authors' baseline (per-session input layer + GRU +
  CTC loss). We wrote our own implementation.
- Everything specific to the participant is in `configs/t15.toml`, so the same code
  can be retrained on another person's data.
- Skipped the authors' n-gram language models: they need about 60 GB to 300 GB of RAM.

### What didn't work / problems hit
- **Out of memory with the reference batch size (Sat, about 2 PM).**
  - What we tried: Run A with batches of 64 sentences, the dataset authors' setting.
  - What happened: it crashed between batches 1,500 and 1,600 with
    `torch.OutOfMemoryError`: "Tried to allocate 1.93 GiB. GPU 0 has a total capacity
    of 10.54 GiB of which 638.69 MiB is free."
  - Why we think it happened: sentences range from about 3 to about 50 seconds, and
    each batch is padded to its longest sentence. A batch that happens to include a
    very long sentence needs far more memory than a typical one. A short speed test
    (a few hundred batches) never drew such a batch, so it did not show the problem.
    The authors trained on a 24 GB card; ours has 11 GB.
  - What we changed: batch size 64 -> 32 and batches 60,000 -> 120,000, so the model
    still sees the same total number of sentences. We also set
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation.
  - Open question: the learning rate was tuned for batches of 64. We have not
    measured whether batches of 32 change the final score.
  - Lesson: a short test run only proves the typical case works. Rare inputs (here,
    the longest sentences) need checking too.
- TODO, add more as they happen.

### Hardware
- Training: Linux desktop, GeForce RTX 2080 Ti (11 GB), 64 GB RAM.
- Development and testing of code: MacBook (Apple M5, 16 GB RAM).

## Sources
- Dataset: Card et al., Brain-to-Text '25, https://www.kaggle.com/competitions/brain-to-text-25 (CC0 licence)
- Paper: Card et al., "An Accurate and Rapidly Calibrating Speech Neuroprosthesis" (2024), New England Journal of Medicine
- Reference code and settings: https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text
- Dataset archive: https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85
- Libraries: PyTorch, NumPy, h5py

## AI tools used
- Claude Code (Claude Fable 5.2, Anthropic): project planning, explanations, and writing
  the code in `b2t/` with us. TODO: note which parts each of us can explain.
