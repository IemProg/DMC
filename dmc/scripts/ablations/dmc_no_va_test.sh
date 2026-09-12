#!/bin/bash

# DPP: Alpha sweep evaluation on base or new classes
# Usage: bash dmc_no_va_test.sh <DATASET> <W_GEN> <W_LMC> <SEED> <CFG> [SUB]
#
# SUB defaults to "new". Pass "base" to evaluate on base classes.

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
DATASET=$1
W_GEN=$2
W_LMC=$3
SEED=$4
CFG=$5
SUB=${6:-new}
CTP=end
NCTX=4
SHOTS=16
CSC=False

for SEED in ${SEED}
do
    COMMON_DIR=${DATASET}/dpp_wgen${W_GEN}_wlmc${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base_dpp/${COMMON_DIR}
    DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/evaluate/dpp_test_${SUB}/${COMMON_DIR}
    RESUME_COOP=None

    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run DPP alpha sweep (sub=${SUB}) and save to ${DIR}"
        python "${TRAIN_PY}" \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --eval-only-dpp \
            --resume-coop ${RESUME_COOP} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            TRAINER.COOP.DPP True \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
