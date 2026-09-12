#!/bin/bash

# DPP: Full pipeline (train + eval base + eval new)
# Usage: bash dmc_no_va_run.sh <DATASET> <W_GEN> <W_LMC> <SEED> <CFG>


HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

echo "=== DPP: Training (W_GEN=$2, W_LMC=$3) ==="
bash "${HERE}/dmc_no_va_train.sh" "$1" "$2" "$3" "$4" "$5"

echo ""
echo "=== DPP: Base class alpha sweep ==="
bash "${HERE}/dmc_no_va_test.sh" "$1" "$2" "$3" "$4" "$5" base

echo ""
echo "=== DPP: New class alpha sweep ==="
bash "${HERE}/dmc_no_va_test.sh" "$1" "$2" "$3" "$4" "$5" new
