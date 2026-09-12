#!/bin/bash
#
# Test KgCoOp_COOP_LMC trained with Proposal A (prompt-space Fisher) on new classes.
#
# Usage:
#   bash prompt_ewc_test.sh <DATASET> <CLIP_W> <W_LMC> <PROMPT_FISHER_W> <FISHER_NORM> <SEED> <CFG>

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
DATASET=$1
Clip_WEIGHT=$2
W_LMC=$3
PROMPT_FISHER_W=$4
FISHER_NORM=$5
SEED=$6
CFG=$7

CTP=end
NCTX=4
SHOTS=16
CSC=False
SUB=new

for SEED in ${SEED}
do
    COMMON_DIR=${DATASET}/prompt_fisher/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}_pf_${PROMPT_FISHER_W}_fn_${FISHER_NORM}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base/${COMMON_DIR}
    DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/evaluate/test_${SUB}/${COMMON_DIR}
    RESUME_COOP=None

    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python "${TRAIN_PY}" \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --eval-only \
            --resume-coop ${RESUME_COOP} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
