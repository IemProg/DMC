#!/bin/bash

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

# custom config
TRAINER=KgCoOp_COOP_LMC
DATASET=$1
Clip_WEIGHT=$2
LOSS_TYPE=$3
COOP_LMC=$4
W_LMC=$5
SEED=$6
CFG=$7
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

SUB=base

for SEED in ${SEED}
do
    COMMON_DIR=${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base/${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
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
