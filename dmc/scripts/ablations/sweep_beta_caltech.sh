#!/bin/bash

# DPP: Sweep W_LMC for Caltech (W_GEN=8.0 fixed, W_LMC=1.0 already done)

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

for W_LMC in 2.0 4.0
do
    for SEED in 1 2 3
    do
        echo "======================================================"
        echo "  DPP: caltech101 W_LMC=${W_LMC} seed=${SEED}"
        echo "======================================================"
        bash "${HERE}/dmc_no_va_run.sh" caltech101 8.0 ${W_LMC} ${SEED} vit_b16_ep100_ctxv1
        echo ""
    done
done
