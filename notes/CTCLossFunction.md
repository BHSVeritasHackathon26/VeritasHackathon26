# CTC (Connectionist Temporal Classification)

**What it is:** A loss for training sequence models when the input and target have different lengths and you don't know the alignment between them (Graves et al., 2006). Example: an audio clip with 500 frames and the transcript "hello" with no per-frame labels.

## How it works

The model emits a distribution over `vocab + blank` at every input frame (T frames). Any frame-level labeling π of length T is a possible **alignment**. A collapse function `B` maps alignments to label sequences by (1) merging consecutive repeats, then (2) deleting blanks:

- `h h e - l l - l o o` → `hello`
- `h e l l o` → `helo` (the blank between the two l's is what makes "ll" representable)

The probability of target `y` is the sum over all alignments that collapse to it:

```
P(y | x) = Σ_{π ∈ B⁻¹(y)}  Π_t  p_t(π_t | x)
Loss     = -log P(y | x)
```

This sum is exponentially large, so it's computed with a forward-backward dynamic program (like an HMM) over the target with blanks interleaved (`-h-e-l-l-o-`, length 2U+1). Allowed transitions at each step are: stay, advance one state, or skip a blank (only when the neighboring labels differ). Cost is O(T·U), fully differentiable, and the gradients come from the same recursion.

Key assumptions and constraints:
- **Conditional independence** of outputs across frames given the input. There is no internal language model, so this is the main weakness.
- **Monotonic alignment** (left to right, no reordering).
- **T ≥ U + (number of adjacent repeated labels)**, or the loss is infinite.

## Decoding

- **Greedy:** argmax per frame, then collapse. Fast and usually decent.
- **Prefix beam search:** merges paths that collapse to the same prefix. Better, and it can fuse an external LM.

## When to use it

- Speech recognition (DeepSpeech, wav2vec 2.0 fine-tuning)
- OCR / handwriting recognition (the classic CRNN pipeline)
- Lip reading, sign language recognition, keyword spotting, music transcription
- Generally: monotonic alignment, input at least as long as output, no alignment labels available

## When not to

- **Non-monotonic tasks** (translation, summarization): use attention seq2seq.
- **Outputs that depend strongly on each other** (the independence assumption hurts): use RNN-T or an attention decoder, or add an external LM.
- **Output longer than input:** CTC can't represent it. Upsample the input.
- **If you have forced alignments:** plain cross-entropy per frame can be simpler and more stable.

## Practical notes (PyTorch)

```python
ctc = nn.CTCLoss(blank=0, zero_infinity=True)

log_probs = model(x).log_softmax(-1)          # (T, N, C), must be log-probs
loss = ctc(log_probs, targets, input_lengths, target_lengths)
```

Common pitfalls:
- Shape is `(T, N, C)` by default, and inputs must be **log_softmax**ed, not raw logits.
- Don't include the blank index in targets, and pad targets consistently with `target_lengths`.
- If you downsample in the network (strided convs, pooling), `input_lengths` must reflect the reduced T.
- `zero_infinity=True` avoids NaNs on samples where T is too short for the target.
- `reduction='mean'` divides each sample's loss by its target length before averaging, which matters if you compare loss values across setups.
- CTC models tend to become "peaky": mostly blanks with sharp spikes at label positions. That's normal, not a bug, but it means frame-level outputs aren't reliable for timing.
