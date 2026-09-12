#!/bin/bash

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

# custom config
TRAINER=KgCoOp_COOP_Fisher_LMC
DATASET=$1
Clip_WEIGHT=$2
LOSS_TYPE=$3
COOP_LMC=$4
W_LMC=$5
BETA1=$6
SEED=$7
CFG=$8
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
NUM_SAMPLES=5

for SEED in ${SEED}
do
    DIR=${OUTPUT}/KgCoOp_COOP_Fisher_LMC/CoOp/train_base/${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}_b${BETA1}/${TRAINER}/${CFG}/seed${SEED}
    RESUME_COOP=${OUTPUT}/coop/train_base/${DATASET}/shots_16/CoOp/vit_b16_ep100_ctxv1/seed${SEED}

    rm -rf "${DIR}"
    echo "Run this job and save the output to ${DIR}"
    python "${TRAIN_PY}" \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --resume-coop ${RESUME_COOP} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${Clip_WEIGHT} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.LOSS_TYPE ${LOSS_TYPE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.COOP.W_LMC ${W_LMC} \
        TRAINER.COOP.COOP_LMC ${COOP_LMC} \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        TRAINER.COOP.BETA1 ${BETA1} \
        DATASET.SUBSAMPLE_CLASSES base
done
