#!/bin/bash

# Full KgCoOp pipeline: KgCoOp Stage 1 → MERGETUNE → DPP → VA → DPP+VA
# Usage: bash run_kgcoop_pipeline.sh <DATASET>
#
# Runs all 3 seeds. Each stage skips if output already exists.
# KgCoOp Stage 1 uses W=8.0 (the KgCoOp regularization weight).

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

DATASET=$1
if [ -z "$DATASET" ]; then
    echo "Usage: bash run_kgcoop_pipeline.sh <DATASET>"
    echo "  e.g.: caltech101, oxford_flowers, oxford_pets, food101"
    exit 1
fi

CFG=vit_b16_ep100_ctxv1
CTP=end
NCTX=4
SHOTS=16
CSC=False
NUM_SAMPLES=5
KGCOOP_W=8.0  # KgCoOp Stage 1 regularization weight

for SEED in 1 2 3
do

echo "############################################################"
echo "  KgCoOp pipeline: ${DATASET} — seed ${SEED}"
echo "############################################################"

# ================================================================
# 1. KgCoOp Stage 1 (trainer=KgCoOp)
# ================================================================
KGCOOP_DIR=${OUTPUT}/KgCoOp/base2new/train_base/${DATASET}/shots_${SHOTS}_${KGCOOP_W}/KgCoOp/${CFG}/seed${SEED}
if [ -d "$KGCOOP_DIR" ] && ls ${KGCOOP_DIR}/prompt_learner/model* > /dev/null 2>&1; then
    echo "[Stage 1] KgCoOp already trained. Skip."
else
    # Clean stale dir from failed runs
    rm -rf "${KGCOOP_DIR}"
    echo "[Stage 1] Training KgCoOp (W=${KGCOOP_W})..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer KgCoOp \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/KgCoOp/${CFG}.yaml \
        --output-dir ${KGCOOP_DIR} \
        TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${KGCOOP_W} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATALOADER.TRAIN_X.BATCH_SIZE 32 \
        DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
fi
RESUME_COOP=${KGCOOP_DIR}

# ================================================================
# 2. MERGETUNE baseline on KgCoOp (W=8.0, cosine, LMC)
# ================================================================
MT_DIR=${OUTPUT}/KgCoOp_COOP_LMC/KgCoOp/train_base/${DATASET}/cosine/shots_${SHOTS}_8.0_1.0/KgCoOp_COOP_LMC/${CFG}/seed${SEED}
if [ -d "$MT_DIR" ] && ls ${MT_DIR}/prompt_mid_learner/model* > /dev/null 2>&1; then
    echo "[Stage 2] KgCoOp+MERGETUNE already trained. Skip."
else
    rm -rf "${MT_DIR}"
    echo "[Stage 2] Training KgCoOp+MERGETUNE..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer KgCoOp_COOP_LMC \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/KgCoOp_COOP_LMC/${CFG}.yaml \
        --output-dir ${MT_DIR} --resume-coop ${RESUME_COOP} \
        TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W 8.0 TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.LOSS_TYPE cosine DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.COOP.W_LMC 1.0 TRAINER.COOP.COOP_LMC True \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} DATASET.SUBSAMPLE_CLASSES base
fi

# MERGETUNE eval base + new
for SUB in base new; do
    MT_EVAL=${OUTPUT}/KgCoOp_COOP_LMC/KgCoOp/evaluate/test_${SUB}/${DATASET}/cosine/shots_${SHOTS}_8.0_1.0/KgCoOp_COOP_LMC/${CFG}/seed${SEED}
    if [ ! -d "$MT_EVAL" ]; then
        echo "[Eval] KgCoOp+MERGETUNE ${SUB}..."
        python "${TRAIN_PY}" \
            --root ${DATA} --seed ${SEED} --trainer KgCoOp_COOP_LMC \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/KgCoOp_COOP_LMC/${CFG}.yaml \
            --output-dir ${MT_EVAL} --model-dir ${MT_DIR} --eval-only \
            --resume-coop None \
            TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done

# ================================================================
# 3. DPP (W_GEN=8.0, W_LMC=4.0)
# ================================================================
DPP_DIR=${OUTPUT}/KgCoOp_COOP_LMC/KgCoOp/train_base_dpp/${DATASET}/dpp_wgen8.0_wlmc4.0/KgCoOp_COOP_LMC/${CFG}/seed${SEED}
if [ -d "$DPP_DIR" ] && ls ${DPP_DIR}/prompt_mid_learner/model* > /dev/null 2>&1; then
    echo "[DPP] Already trained. Skip."
else
    rm -rf "${DPP_DIR}"
    echo "[DPP] Training..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer KgCoOp_COOP_LMC \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/KgCoOp_COOP_LMC/${CFG}.yaml \
        --output-dir ${DPP_DIR} --resume-coop ${RESUME_COOP} \
        TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.DPP True TRAINER.COOP.DPP_W_GEN 8.0 \
        TRAINER.COOP.W_LMC 4.0 TRAINER.COOP.COOP_LMC True \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
fi

for SUB in base new; do
    DPP_EVAL=${OUTPUT}/KgCoOp_COOP_LMC/KgCoOp/evaluate/dpp_test_${SUB}/${DATASET}/dpp_wgen8.0_wlmc4.0/KgCoOp_COOP_LMC/${CFG}/seed${SEED}
    if [ ! -d "$DPP_EVAL" ]; then
        echo "[Eval] DPP ${SUB}..."
        python "${TRAIN_PY}" \
            --root ${DATA} --seed ${SEED} --trainer KgCoOp_COOP_LMC \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/KgCoOp_COOP_LMC/${CFG}.yaml \
            --output-dir ${DPP_EVAL} --model-dir ${DPP_DIR} --eval-only-dpp \
            --resume-coop None \
            TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} TRAINER.COOP.DPP True \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done

# ================================================================
# 4. VA standalone (W=8.0, VA_W=0.5)
# ================================================================
VA_DIR=${OUTPUT}/KgCoOp_COOP_LMC/KgCoOp/train_base_va/${DATASET}/cosine/shots_${SHOTS}_8.0_1.0_va0.5_tau2.0/KgCoOp_COOP_LMC/${CFG}/seed${SEED}
if [ -d "$VA_DIR" ] && ls ${VA_DIR}/prompt_mid_learner/model* > /dev/null 2>&1; then
    echo "[VA] Already trained. Skip."
else
    rm -rf "${VA_DIR}"
    echo "[VA] Training..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer KgCoOp_COOP_LMC \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/KgCoOp_COOP_LMC/${CFG}.yaml \
        --output-dir ${VA_DIR} --resume-coop ${RESUME_COOP} \
        TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W 8.0 TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.LOSS_TYPE cosine DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.COOP.W_LMC 1.0 TRAINER.COOP.COOP_LMC True \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        TRAINER.COOP.VA_W 0.5 TRAINER.COOP.VA_TAU 2.0 \
        DATALOADER.TRAIN_X.BATCH_SIZE 32 \
        DATASET.SUBSAMPLE_CLASSES base
fi

for SUB in base new; do
    VA_EVAL=${OUTPUT}/KgCoOp_COOP_LMC/KgCoOp/evaluate/va_test_${SUB}/${DATASET}/cosine/shots_${SHOTS}_8.0_1.0_va0.5_tau2.0/KgCoOp_COOP_LMC/${CFG}/seed${SEED}
    if [ ! -d "$VA_EVAL" ]; then
        echo "[Eval] VA ${SUB}..."
        python "${TRAIN_PY}" \
            --root ${DATA} --seed ${SEED} --trainer KgCoOp_COOP_LMC \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/KgCoOp_COOP_LMC/${CFG}.yaml \
            --output-dir ${VA_EVAL} --model-dir ${VA_DIR} --eval-only \
            --resume-coop None \
            TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done

# ================================================================
# 5. DPP+VA (W_GEN=8.0, W_LMC=4.0, VA_W=0.5)
# ================================================================
DPPVA_DIR=${OUTPUT}/KgCoOp_COOP_LMC/KgCoOp/train_base_dpp_va/${DATASET}/dpp_wgen8.0_wlmc4.0_va0.5/KgCoOp_COOP_LMC/${CFG}/seed${SEED}
if [ -d "$DPPVA_DIR" ] && ls ${DPPVA_DIR}/prompt_mid_learner/model* > /dev/null 2>&1; then
    echo "[DPP+VA] Already trained. Skip."
else
    rm -rf "${DPPVA_DIR}"
    echo "[DPP+VA] Training..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer KgCoOp_COOP_LMC \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/KgCoOp_COOP_LMC/${CFG}.yaml \
        --output-dir ${DPPVA_DIR} --resume-coop ${RESUME_COOP} \
        TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.DPP True TRAINER.COOP.DPP_W_GEN 8.0 \
        TRAINER.COOP.W_LMC 4.0 TRAINER.COOP.COOP_LMC True \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        TRAINER.COOP.VA_W 0.5 TRAINER.COOP.VA_TAU 2.0 \
        DATALOADER.TRAIN_X.BATCH_SIZE 32 \
        DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
fi

for SUB in base new; do
    DPPVA_EVAL=${OUTPUT}/KgCoOp_COOP_LMC/KgCoOp/evaluate/dpp_va_test_${SUB}/${DATASET}/dpp_wgen8.0_wlmc4.0_va0.5/KgCoOp_COOP_LMC/${CFG}/seed${SEED}
    if [ ! -d "$DPPVA_EVAL" ]; then
        echo "[Eval] DPP+VA ${SUB}..."
        python "${TRAIN_PY}" \
            --root ${DATA} --seed ${SEED} --trainer KgCoOp_COOP_LMC \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/KgCoOp_COOP_LMC/${CFG}.yaml \
            --output-dir ${DPPVA_EVAL} --model-dir ${DPPVA_DIR} --eval-only-dpp \
            --resume-coop None \
            TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} TRAINER.COOP.DPP True \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done

echo ""
done

echo "############################################################"
echo "  KgCoOp pipeline: ${DATASET} — ALL DONE"
echo "############################################################"
