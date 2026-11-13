#!/bin/bash

# Visual Anchor training
# Usage: bash single_prompt_va_train.sh <DATASET> <Clip_WEIGHT> <LOSS_TYPE> <COOP_LMC> <W_LMC> <SEED> <CFG> <VA_W> [VA_TAU]
#
# VA_W: Visual Anchor weight (e.g., 0.1, 0.5, 1.0, 2.0)
# VA_TAU: Visual Anchor temperature (default: 2.0)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
DATASET=$1
Clip_WEIGHT=$2
LOSS_TYPE=$3
COOP_LMC=$4
W_LMC=$5
SEED=$6
CFG=$7
VA_W=$8
VA_TAU=${9:-2.0}
CTP=end
NCTX=4
SHOTS=16
CSC=False
NUM_SAMPLES=5

for SEED in ${SEED}
do
    DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base_va/${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}_va${VA_W}_tau${VA_TAU}/${TRAINER}/${CFG}/seed${SEED}
    RESUME_COOP=${OUTPUT}/coop/train_base/${DATASET}/shots_16/CoOp/vit_b16_ep100_ctxv1/seed${SEED}

    rm -rf "${DIR}"
    echo "Run Visual Anchor training (VA_W=${VA_W}, VA_TAU=${VA_TAU}) and save to ${DIR}"
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
        TRAINER.COOP.VA_W ${VA_W} \
        TRAINER.COOP.VA_TAU ${VA_TAU} \
        DATALOADER.TRAIN_X.BATCH_SIZE 32 \
        DATASET.SUBSAMPLE_CLASSES base
done
