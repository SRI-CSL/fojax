#!/usr/bin/env bash

METHODS=(
  "make_mala_step"
  "make_fmala_step"
  "make_line_fmala_step"
  "make_precon_fmala_step"
  "make_precon_line_fmala_step"
)

# Example default settings
EPSILON=0.0001
NUM_SAMPLES=100
NUM_REPEATS=5
SUB_SAMPLE=1

for METHOD in "${METHODS[@]}"; do
    echo "Running cifar10_time.py with sampler=${METHOD}"
    export CUDA_VISIBLE_DEVICES=0
    python cifar10_time.py \
      --sampler "${METHOD}" \
      --epsilon "${EPSILON}" \
      --num_samples "${NUM_SAMPLES}" \
      --n_time_iterations "${NUM_REPEATS}" \
      --subsample "${SUB_SAMPLE}" \
      --force
    echo ""
done
