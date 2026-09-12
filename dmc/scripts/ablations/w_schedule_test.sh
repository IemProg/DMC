#!/bin/bash
#
# Evaluate W-schedule-trained model on NEW classes.
#
# Usage:
#   bash w_schedule_test.sh <DATASET> <W_MAX> <W_LMC> <W_SCHEDULE> <W_MIN> <SEED> <CFG>

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
DATASET=$1
W_MAX=$2
W_LMC=$3
W_SCHEDULE=$4
W_MIN=$5
SEED=$6
CFG=$7

CTP=end
NCTX=4
SHOTS=16
CSC=False

SUB=new

for SEED in ${SEED}
do
    COMMON_DIR=${DATASET}/w_schedule/shots_${SHOTS}_${W_MAX}_${W_LMC}_${W_SCHEDULE}_wmin_${W_MIN}/${TRAINER}/${CFG}/seed${SEED}
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
