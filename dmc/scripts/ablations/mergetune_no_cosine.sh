#!/bin/bash

# Ablation: MERGETUNE without R (W=0)
# Tests whether the cosine regularizer is necessary for the single-prompt method
# Loss: CE(c) + β·LMC(ŵ₂ → c), no cosine score
# Usage: bash mergetune_no_cosine.sh <DATASET>

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

DATASET=$1
if [ -z "$DATASET" ]; then
    echo "Usage: bash mergetune_no_cosine.sh <DATASET>"
    exit 1
fi

TRAINER=KgCoOp_COOP_LMC
CFG=vit_b16_ep100_ctxv1
CTP=end
NCTX=4
SHOTS=16
CSC=False
NUM_SAMPLES=5
W=0.0
W_LMC=1.0
LOSS_TYPE=cosine
SEED=1

RESUME_COOP=${OUTPUT}/coop/train_base/${DATASET}/shots_16/CoOp/vit_b16_ep100_ctxv1/seed${SEED}

# Train
TRAIN_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_ablation_mt_no_R/${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_w${W}_wlmc${W_LMC}/${TRAINER}/${CFG}/seed${SEED}

if [ -d "$TRAIN_DIR" ] && ls ${TRAIN_DIR}/prompt_mid_learner/model* > /dev/null 2>&1; then
    echo "[Train] Already done. Skip."
else
    rm -rf "${TRAIN_DIR}"
    echo "[Train] ${DATASET}: MERGETUNE with W=0 (no R)..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer ${TRAINER} \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${TRAIN_DIR} --resume-coop ${RESUME_COOP} \
        TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.W ${W} \
        TRAINER.COOP.LOSS_TYPE ${LOSS_TYPE} \
        TRAINER.COOP.W_LMC ${W_LMC} TRAINER.COOP.COOP_LMC True \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
fi

# Eval
for SUB in base new; do
    EVAL_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/evaluate/ablation_mt_no_R/test_${SUB}/${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_w${W}_wlmc${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    if [ ! -d "$EVAL_DIR" ]; then
        echo "[Eval] ${DATASET} ${SUB}..."
        python "${TRAIN_PY}" \
            --root ${DATA} --seed ${SEED} --trainer ${TRAINER} \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${EVAL_DIR} --model-dir ${TRAIN_DIR} \
            --eval-only --resume-coop None \
            TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
