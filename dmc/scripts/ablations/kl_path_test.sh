#!/bin/bash
#
# Test KgCoOp_COOP_LMC trained with Proposal B (KL path) on new classes.
#
# Usage:
#   bash kl_path_test.sh <DATASET> <CLIP_W> <W_LMC> <KL_PATH_W> <SEED> <CFG>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
DATASET=$1
Clip_WEIGHT=$2
W_LMC=$3
KL_PATH_W=$4
SEED=$5
CFG=$6

CTP=end
NCTX=4
SHOTS=16
CSC=False
SUB=new

for SEED in ${SEED}
do
    COMMON_DIR=${DATASET}/kl_path/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}_kl_${KL_PATH_W}/${TRAINER}/${CFG}/seed${SEED}
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
