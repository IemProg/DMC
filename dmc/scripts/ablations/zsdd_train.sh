#!/bin/bash
#
# Train KgCoOp_COOP_LMC with ZSDD (Proposal F: Zero-Shot Distribution Distillation).
#
# Usage:
#   bash zsdd_train.sh <DATASET> <CLIP_W> <W_LMC> <ZSDD_W> <ZSDD_TAU> <SEED> <CFG>
#
# Example:
#   bash zsdd_train.sh caltech101 10.0 1.0 1.0 2.0 1 vit_b16_ep100_ctxv1
#   bash zsdd_train.sh oxford_flowers 10.0 1.0 2.0 4.0 1 vit_b16_ep100_ctxv1

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
DATASET=$1
Clip_WEIGHT=$2
W_LMC=$3
ZSDD_W=$4
ZSDD_TAU=$5
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
    DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base/${DATASET}/zsdd/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}_zsdd_${ZSDD_W}_tau_${ZSDD_TAU}/${TRAINER}/${CFG}/seed${SEED}
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
        TRAINER.COOP.ZSDD_W ${ZSDD_W} \
        TRAINER.COOP.ZSDD_TAU ${ZSDD_TAU} \
        DATASET.SUBSAMPLE_CLASSES base
done
