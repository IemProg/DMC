#!/bin/bash

# DPP + VA: Full pipeline (train + eval base + eval new) for all 3 seeds
# Usage: bash run_all_seeds.sh <DATASET> <W_GEN> <W_LMC> <VA_W> <CFG> [VA_TAU]

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

VA_TAU=${6:-2.0}

for SEED in 1 2 3
do
    echo "======================================================"
    echo "  DPP+VA: $1 seed=${SEED} (W_GEN=$2, W_LMC=$3, VA_W=$4)"
    echo "======================================================"

    echo "--- Training ---"
    bash "${HERE}/train.sh" "$1" "$2" "$3" "$4" ${SEED} "$5" "${VA_TAU}"

    echo "--- Base eval ---"
    bash "${HERE}/test.sh" "$1" "$2" "$3" "$4" ${SEED} "$5" base

    echo "--- New eval ---"
    bash "${HERE}/test.sh" "$1" "$2" "$3" "$4" ${SEED} "$5" new
    echo ""
done
