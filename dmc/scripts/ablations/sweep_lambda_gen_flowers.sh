#!/bin/bash

# DPP: Sweep W_GEN for Flowers (seed 1, W_GEN=8.0 already done)

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

for W_GEN in 4.0 16.0
do
    echo "======================================================"
    echo "  DPP: oxford_flowers W_GEN=${W_GEN}"
    echo "======================================================"
    bash "${HERE}/dmc_no_va_run.sh" oxford_flowers ${W_GEN} 1.0 1 vit_b16_ep100_ctxv1
    echo ""
done
