#!/bin/bash

# Ablation: λ_gen=0 (remove R entirely from DMC)
# Usage: bash dmc_no_cosine.sh <DATASET>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

DATASET=$1
if [ -z "$DATASET" ]; then
    echo "Usage: bash dmc_no_cosine.sh <DATASET>"
    exit 1
fi

TRAINER=KgCoOp_COOP_LMC
CFG=vit_b16_ep100_ctxv1
CTP=end
NCTX=4
SHOTS=16
CSC=False
NUM_SAMPLES=5
W_GEN=0.0
W_LMC=1.0
SEED=1

RESUME_COOP=${OUTPUT}/coop/train_base/${DATASET}/shots_16/CoOp/vit_b16_ep100_ctxv1/seed${SEED}

TRAIN_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_ablation_no_R/${DATASET}/dpp_wgen${W_GEN}_wlmc${W_LMC}/${TRAINER}/${CFG}/seed${SEED}

if [ -d "$TRAIN_DIR" ] && ls ${TRAIN_DIR}/prompt_mid_learner/model* > /dev/null 2>&1; then
    echo "[Train] Already done. Skip."
else
    rm -rf "${TRAIN_DIR}"
    echo "[Train] ${DATASET}: DMC with λ_gen=0 (no R)..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer ${TRAINER} \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${TRAIN_DIR} --resume-coop ${RESUME_COOP} \
        TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.DPP True \
        TRAINER.COOP.DPP_W_GEN ${W_GEN} \
        TRAINER.COOP.W_LMC ${W_LMC} TRAINER.COOP.COOP_LMC True \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
fi

for SUB in base new; do
    EVAL_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/evaluate/ablation_no_R/test_${SUB}/${DATASET}/dpp_wgen${W_GEN}_wlmc${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    if [ ! -d "$EVAL_DIR" ]; then
        echo "[Eval] ${DATASET} ${SUB}..."
        python "${TRAIN_PY}" \
            --root ${DATA} --seed ${SEED} --trainer ${TRAINER} \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${EVAL_DIR} --model-dir ${TRAIN_DIR} \
            --eval-only-dpp --resume-coop None \
            TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            TRAINER.COOP.DPP True \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
