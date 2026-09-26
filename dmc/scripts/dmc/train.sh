#!/bin/bash

# DPP + Visual Anchor training
# Usage: bash train.sh <DATASET> <W_GEN> <W_LMC> <VA_W> <SEED> <CFG> [VA_TAU]

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
DATASET=$1
W_GEN=$2
W_LMC=$3
VA_W=$4
SEED=$5
CFG=$6
VA_TAU=${7:-2.0}
CTP=end
NCTX=4
SHOTS=16
CSC=False
NUM_SAMPLES=5

for SEED in ${SEED}
do
    DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base_dpp_va/${DATASET}/dpp_wgen${W_GEN}_wlmc${W_LMC}_va${VA_W}/${TRAINER}/${CFG}/seed${SEED}
    RESUME_COOP=${OUTPUT}/coop/train_base/${DATASET}/shots_16/CoOp/vit_b16_ep100_ctxv1/seed${SEED}

    rm -rf "${DIR}"
    echo "Run DPP+VA training (W_GEN=${W_GEN}, W_LMC=${W_LMC}, VA_W=${VA_W}) and save to ${DIR}"
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
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.DPP True \
        TRAINER.COOP.DPP_W_GEN ${W_GEN} \
        TRAINER.COOP.W_LMC ${W_LMC} \
        TRAINER.COOP.COOP_LMC True \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        TRAINER.COOP.VA_W ${VA_W} \
        TRAINER.COOP.VA_TAU ${VA_TAU} \
        DATALOADER.TRAIN_X.BATCH_SIZE 32 \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
done
