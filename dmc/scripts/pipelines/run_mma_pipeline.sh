#!/bin/bash

# MMA pipeline v2: Properly trained (50 epochs)
# Stage 1: MMA (50ep) → MERGETUNE (50ep) → DPP → DPP+VA → evals + WiSE-FT
#
# Usage: bash run_mma_pipeline.sh <DATASET>
# Example: bash run_mma_pipeline.sh caltech101

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

DATASET=$1
if [ -z "$DATASET" ]; then
    echo "Usage: bash run_mma_pipeline.sh <DATASET>"
    exit 1
fi

CFG_MMA=vit_b16_ep50
CFG_LMC=vit_b16_ep50
SHOTS=16
NUM_SAMPLES=5

for SEED in 1 2 3
do

echo "############################################################"
echo "  MMA v2 pipeline: ${DATASET} — seed ${SEED}"
echo "############################################################"

# ================================================================
# 1. MMA Stage 1 (50 epochs, trainer=MultiModalAdapter)
# ================================================================
MMA_DIR=${OUTPUT}/MMA/base2new/train_base/${DATASET}/shots_${SHOTS}/MultiModalAdapter/seed${SEED}
if [ -d "$MMA_DIR" ] && ls ${MMA_DIR}/adapter_learner/model* > /dev/null 2>&1; then
    echo "[Stage 1] MMA (50ep) already trained. Skip."
else
    rm -rf "${MMA_DIR}"
    echo "[Stage 1] Training MMA (50 epochs)..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer MultiModalAdapter \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/MultiModalAdapter/${CFG_MMA}.yaml \
        --output-dir ${MMA_DIR} \
        DATALOADER.TRAIN_X.BATCH_SIZE 32 \
        DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
fi
RESUME_MMA=${MMA_DIR}

# ================================================================
# 2. MMA+MERGETUNE (50 epochs)
# ================================================================
MT_DIR=${OUTPUT}/MMA_LMC/train_base/${DATASET}/shots_${SHOTS}_wlmc1.0/MMA_LMC/${CFG_LMC}/seed${SEED}
if [ -d "$MT_DIR" ] && ls ${MT_DIR}/adapter_learner/model* > /dev/null 2>&1; then
    echo "[Stage 2] MMA+MERGETUNE (50ep) already trained. Skip."
else
    rm -rf "${MT_DIR}"
    echo "[Stage 2] Training MMA+MERGETUNE (50 epochs)..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer MMA_LMC \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/MMA_LMC/${CFG_LMC}.yaml \
        --output-dir ${MT_DIR} --resume-coop ${RESUME_MMA} \
        TRAINER.COOP.W_LMC 1.0 TRAINER.COOP.COOP_LMC True \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        DATALOADER.TRAIN_X.BATCH_SIZE 32 \
        DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
fi

# MMA+MERGETUNE eval
for SUB in base new; do
    MT_EVAL=${OUTPUT}/MMA_LMC/evaluate/test_${SUB}/${DATASET}/shots_${SHOTS}_wlmc1.0/MMA_LMC/${CFG_LMC}/seed${SEED}
    if [ ! -d "$MT_EVAL" ]; then
        echo "[Eval] MMA+MERGETUNE ${SUB}..."
        python "${TRAIN_PY}" \
            --root ${DATA} --seed ${SEED} --trainer MMA_LMC \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/MMA_LMC/${CFG_LMC}.yaml \
            --output-dir ${MT_EVAL} --model-dir ${MT_DIR} --eval-only \
            --resume-coop None \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done

# ================================================================
# 3. DPP (W_GEN=8.0, W_LMC=4.0)
# ================================================================
DPP_DIR=${OUTPUT}/MMA_LMC/train_base_dpp/${DATASET}/dpp_wgen8.0_wlmc4.0/MMA_LMC/${CFG_LMC}/seed${SEED}
if [ -d "$DPP_DIR" ] && ls ${DPP_DIR}/adapter_learner/model* > /dev/null 2>&1; then
    echo "[DPP] Already trained. Skip."
else
    rm -rf "${DPP_DIR}"
    echo "[DPP] Training..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer MMA_LMC \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/MMA_LMC/${CFG_LMC}.yaml \
        --output-dir ${DPP_DIR} --resume-coop ${RESUME_MMA} \
        TRAINER.COOP.DPP True TRAINER.COOP.DPP_W_GEN 8.0 \
        TRAINER.COOP.W_LMC 4.0 TRAINER.COOP.COOP_LMC True \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        DATALOADER.TRAIN_X.BATCH_SIZE 32 \
        DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
fi

for SUB in base new; do
    DPP_EVAL=${OUTPUT}/MMA_LMC/evaluate/dpp_test_${SUB}/${DATASET}/dpp_wgen8.0_wlmc4.0/MMA_LMC/${CFG_LMC}/seed${SEED}
    if [ ! -d "$DPP_EVAL" ]; then
        echo "[Eval] DPP ${SUB}..."
        python "${TRAIN_PY}" \
            --root ${DATA} --seed ${SEED} --trainer MMA_LMC \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/MMA_LMC/${CFG_LMC}.yaml \
            --output-dir ${DPP_EVAL} --model-dir ${DPP_DIR} --eval-only-dpp \
            --resume-coop None \
            TRAINER.COOP.DPP True \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done

# ================================================================
# 4. DPP+VA (W_GEN=8.0, W_LMC=4.0, VA_W=0.5)
# ================================================================
DPPVA_DIR=${OUTPUT}/MMA_LMC/train_base_dpp_va/${DATASET}/dpp_wgen8.0_wlmc4.0_va0.5/MMA_LMC/${CFG_LMC}/seed${SEED}
if [ -d "$DPPVA_DIR" ] && ls ${DPPVA_DIR}/adapter_learner/model* > /dev/null 2>&1; then
    echo "[DPP+VA] Already trained. Skip."
else
    rm -rf "${DPPVA_DIR}"
    echo "[DPP+VA] Training..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer MMA_LMC \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/MMA_LMC/${CFG_LMC}.yaml \
        --output-dir ${DPPVA_DIR} --resume-coop ${RESUME_MMA} \
        TRAINER.COOP.DPP True TRAINER.COOP.DPP_W_GEN 8.0 \
        TRAINER.COOP.W_LMC 4.0 TRAINER.COOP.COOP_LMC True \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        TRAINER.COOP.VA_W 0.5 TRAINER.COOP.VA_TAU 2.0 \
        DATALOADER.TRAIN_X.BATCH_SIZE 32 \
        DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
fi

for SUB in base new; do
    DPPVA_EVAL=${OUTPUT}/MMA_LMC/evaluate/dpp_va_test_${SUB}/${DATASET}/dpp_wgen8.0_wlmc4.0_va0.5/MMA_LMC/${CFG_LMC}/seed${SEED}
    if [ ! -d "$DPPVA_EVAL" ]; then
        echo "[Eval] DPP+VA ${SUB}..."
        python "${TRAIN_PY}" \
            --root ${DATA} --seed ${SEED} --trainer MMA_LMC \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/MMA_LMC/${CFG_LMC}.yaml \
            --output-dir ${DPPVA_EVAL} --model-dir ${DPPVA_DIR} --eval-only-dpp \
            --resume-coop None \
            TRAINER.COOP.DPP True \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done

# ================================================================
# 5. WiSE-FT diagnostic (adapter weight interpolation sweep)
# ================================================================
for SUB in base new; do
    WISEFT_EVAL=${OUTPUT}/MMA_LMC/evaluate_wiseft/test_${SUB}/${DATASET}/shots_${SHOTS}_wlmc1.0/MMA_LMC/${CFG_LMC}/seed${SEED}
    if [ ! -d "$WISEFT_EVAL" ]; then
        echo "[WiSE-FT] ${SUB} sweep..."
        python "${TRAIN_PY}" \
            --root ${DATA} --seed ${SEED} --trainer MMA_LMC \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/MMA_LMC/${CFG_LMC}.yaml \
            --output-dir ${WISEFT_EVAL} --model-dir ${MT_DIR} \
            --eval-only-wiseft --resume-coop None \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done

echo ""
done

echo "############################################################"
echo "  MMA v2 pipeline: ${DATASET} — ALL DONE"
echo "############################################################"
