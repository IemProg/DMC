#!/bin/bash

# DPP: Sweep W_LMC for Flowers (W_GEN=8.0 fixed, W_LMC=1.0 already done)
# Testing whether stronger LMC path constraint reduces seed variance

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

for W_LMC in 2.0 4.0
do
    for SEED in 1 2 3
    do
        echo "======================================================"
        echo "  DPP: oxford_flowers W_LMC=${W_LMC} seed=${SEED}"
        echo "======================================================"
        bash "${HERE}/dmc_no_va_run.sh" oxford_flowers 8.0 ${W_LMC} ${SEED} vit_b16_ep100_ctxv1
        echo ""
    done
done
