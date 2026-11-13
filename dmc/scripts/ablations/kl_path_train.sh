#!/bin/bash
#
# Train KgCoOp_COOP_LMC with Proposal B: Task-1 KL path (w1 -> w).
# Adds a KL divergence path constraint toward zero-shot on top of MERGETUNE.
#
# Usage:
#   bash kl_path_train.sh <DATASET> <CLIP_W> <W_LMC> <KL_PATH_W> <SEED> <CFG>
#
# Example:
#   bash kl_path_train.sh caltech101 8.0 1.0 0.3 1 vit_b16_ep100_ctxv1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
DATASET=$1
Clip_WEIGHT=$2
W_LMC=$3
KL_PATH_W=$4
SEED=$5
CFG=$6

LOSS_TYPE=cosine
COOP_LMC=True
CTP=end
NCTX=4
SHOTS=16
CSC=False
NUM_SAMPLES=5

for SEED in ${SEED}
do
    DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base/${DATASET}/kl_path/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}_kl_${KL_PATH_W}/${TRAINER}/${CFG}/seed${SEED}
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
        TRAINER.COOP.KL_PATH_W ${KL_PATH_W} \
        DATASET.SUBSAMPLE_CLASSES base
done
