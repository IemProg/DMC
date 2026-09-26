#!/bin/bash
#
# Train KgCoOp_COOP_LMC with Proposal A: prompt-space Fisher EWC.
# Adds an EWC penalty on ctx parameters on top of the original MERGETUNE loss.
#
# Usage:
#   bash prompt_ewc_train.sh <DATASET> <CLIP_W> <W_LMC> <PROMPT_FISHER_W> <FISHER_NORM> <SEED> <CFG>
#
# Example:
#   bash prompt_ewc_train.sh caltech101 8.0 1.0 0.1 max 1 vit_b16_ep100_ctxv1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
DATASET=$1
Clip_WEIGHT=$2
W_LMC=$3
PROMPT_FISHER_W=$4
FISHER_NORM=$5
SEED=$6
CFG=$7

LOSS_TYPE=cosine
COOP_LMC=True
CTP=end
NCTX=4
SHOTS=16
CSC=False
NUM_SAMPLES=5
FISHER_SAMPLES=1024

for SEED in ${SEED}
do
    DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base/${DATASET}/prompt_fisher/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}_pf_${PROMPT_FISHER_W}_fn_${FISHER_NORM}/${TRAINER}/${CFG}/seed${SEED}
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
        TRAINER.COOP.PROMPT_FISHER_W ${PROMPT_FISHER_W} \
        TRAINER.COOP.FISHER_SAMPLES ${FISHER_SAMPLES} \
        TRAINER.COOP.FISHER_NORM ${FISHER_NORM} \
        DATASET.SUBSAMPLE_CLASSES base
done
