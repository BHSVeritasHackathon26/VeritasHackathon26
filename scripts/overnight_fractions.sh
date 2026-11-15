#!/usr/bin/env bash
# The data-amount experiment: how does accuracy change with the number of training
# sentences?
#
# Run from the project folder, with the Python environment active, when nothing
# else is using the GPU:
#
#     bash scripts/overnight_fractions.sh
#
# It trains the same model four times, one after another, on 10%, 25%, 50% and 100%
# of the training sentences, for 40,000 batches each (about 1.85 hours each on the
# 2080 Ti, so roughly 7.5 hours in total). All four are scored on the same 808 tune
# sentences. If one run fails, the script carries on with the next.
# Everything it prints is also saved to runs/fractions_log.txt.

mkdir -p runs
LOG=runs/fractions_log.txt

for FRACTION in 0.10 0.25 0.50 1.00; do
    echo "$(date)  Starting fraction $FRACTION" | tee -a "$LOG"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m b2t.train \
        --config configs/t15.toml --model gru --out "runs/frac_$FRACTION" \
        --train-fraction "$FRACTION" --batches 40000 --batch-size 32 --eval-every 4000 \
        2>&1 | tee -a "$LOG"
done
echo "$(date)  All fraction runs finished." | tee -a "$LOG"
