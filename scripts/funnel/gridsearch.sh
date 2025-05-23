#!/bin/bash
# Usage: ./gridsearch.sh <sampler> <funnel_D> <num_samples> <eps_min>
# Example: ./gridsearch.sh make_fmala_step 100 10000 0.01 #0.1

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <sampler> <funnel_D> <num_samples> <eps_min>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

sampler=$1
funnel_D=$2
num_samples=$3
eps_min=$4

# Loop over the desired seed values
for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "Running with seed $seed..."
    python funnel_grid.py --sampler "$sampler" --num_samples "$num_samples" --funnel_D "$funnel_D" --seed "$seed" --bashscript --eps_min "$eps_min" --log --correction --eps_max 2.0 # 0.1 for the other ones 0.01 for D=100
done