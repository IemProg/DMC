#!/bin/bash
#
# Evaluate ZSDD-trained model on NEW classes (base-to-new generalization).
#
# Usage:
#   bash zsdd_test.sh <DATASET> <CLIP_W> <W_LMC> <ZSDD_W> <ZSDD_TAU> <SEED> <CFG>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
DATASET=$1
Clip_WEIGHT=$2
W_LMC=$3
ZSDD_W=$4
ZSDD_TAU=$5
SEED=$6
CFG=$7

CTP=end
NCTX=4
SHOTS=16
CSC=False

SUB=new

for SEED in ${SEED}
do
    COMMON_DIR=${DATASET}/zsdd/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}_zsdd_${ZSDD_W}_tau_${ZSDD_TAU}/${TRAINER}/${CFG}/seed${SEED}
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
