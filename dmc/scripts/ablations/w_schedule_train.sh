#!/bin/bash
#
# Train KgCoOp_COOP_LMC with adaptive W scheduling (Proposal G).
#
# Usage:
#   bash w_schedule_train.sh <DATASET> <W_MAX> <W_LMC> <W_SCHEDULE> <W_MIN> <SEED> <CFG>
#
# Example:
#   bash w_schedule_train.sh caltech101 10.0 1.0 cosine 0.0 1 vit_b16_ep100_ctxv1
#   bash w_schedule_train.sh oxford_flowers 10.0 1.0 linear 2.0 1 vit_b16_ep100_ctxv1
#
# W_MIN=0.0 means auto (W_MAX / 3)

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

LOSS_TYPE=cosine
COOP_LMC=True
CTP=end
NCTX=4
SHOTS=16
CSC=False
NUM_SAMPLES=5

for SEED in ${SEED}
do
    DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base/${DATASET}/w_schedule/shots_${SHOTS}_${W_MAX}_${W_LMC}_${W_SCHEDULE}_wmin_${W_MIN}/${TRAINER}/${CFG}/seed${SEED}
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
        TRAINER.COOP.W ${W_MAX} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.LOSS_TYPE ${LOSS_TYPE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.COOP.W_LMC ${W_LMC} \
        TRAINER.COOP.COOP_LMC ${COOP_LMC} \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        TRAINER.COOP.W_SCHEDULE ${W_SCHEDULE} \
        TRAINER.COOP.W_MIN ${W_MIN} \
        DATASET.SUBSAMPLE_CLASSES base
done
