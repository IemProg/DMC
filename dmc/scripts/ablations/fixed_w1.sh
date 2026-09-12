#!/bin/bash

# Ablation Level 2: Train DMC with fixed w1 as gen endpoint (no learnable c_gen)
# Compares: DMC (learned f_gen) vs DMC-fixed (w1 as gen, c_base only trained)
# Training: CE(f_base) + VA(f_base) + LMC(w1 -> f_base)
# Eval: alpha sweep from w1 to f_base

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
CFG=vit_b16_ep100_ctxv1
CTP=end
NCTX=4
SHOTS=16
CSC=False
NUM_SAMPLES=5
W_GEN=8.0
W_LMC=4.0
VA_W=0.5
VA_TAU=2.0

DATASET=$1
if [ -z "$DATASET" ]; then
    echo "Usage: bash fixed_w1.sh <DATASET>"
    exit 1
fi

DATASET_CFG=${CONFIGS}/datasets/${DATASET}.yaml
if [ ! -f "${DATASET_CFG}" ]; then
    echo "ERROR: dataset config not found: ${DATASET_CFG}"
    echo "Available datasets:"
    ls ${CONFIGS}/datasets/ | sed 's/\.yaml$//' | sed 's/^/  /'
    exit 1
fi

for SEED in 1 2 3
do

echo "############################################################"
echo "  Ablation fixed_w1: ${DATASET} seed ${SEED}"
echo "############################################################"

# Resume from CoOp Stage 1 checkpoint (same as DMC)
RESUME_COOP=${OUTPUT}/coop/train_base/${DATASET}/shots_16/CoOp/vit_b16_ep100_ctxv1/seed${SEED}

# ================================================================
# Train: DMC with fixed w1 gen endpoint
# ================================================================
TRAIN_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_ablation_fixed_w1/${DATASET}/dpp_wlmc${W_LMC}_va${VA_W}/${TRAINER}/${CFG}/seed${SEED}

if [ -d "$TRAIN_DIR" ] && ls ${TRAIN_DIR}/prompt_mid_learner/model* > /dev/null 2>&1; then
    echo "[Train] Already done. Skip."
else
    rm -rf "${TRAIN_DIR}"
    echo "[Train] Training DMC with fixed w1..."
    python "${TRAIN_PY}" \
        --root ${DATA} --seed ${SEED} --trainer ${TRAINER} \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${TRAIN_DIR} --resume-coop ${RESUME_COOP} \
        TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.DPP True \
        TRAINER.COOP.DPP_GEN_MODE fixed_w1 \
        TRAINER.COOP.W_LMC ${W_LMC} TRAINER.COOP.COOP_LMC True \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        TRAINER.COOP.VA_W ${VA_W} TRAINER.COOP.VA_TAU ${VA_TAU} \
        DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
fi

# ================================================================
# Eval: alpha sweep on base and new
# ================================================================
for SUB in base new; do
    EVAL_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/evaluate/ablation_fixed_w1/test_${SUB}/${DATASET}/dpp_wlmc${W_LMC}_va${VA_W}/${TRAINER}/${CFG}/seed${SEED}
    if [ ! -d "$EVAL_DIR" ]; then
        echo "[Eval] fixed_w1 ${SUB}..."
        python "${TRAIN_PY}" \
            --root ${DATA} --seed ${SEED} --trainer ${TRAINER} \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${EVAL_DIR} --model-dir ${TRAIN_DIR} \
            --eval-only-dpp --resume-coop None \
            TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            TRAINER.COOP.DPP True \
            TRAINER.COOP.DPP_GEN_MODE fixed_w1 \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done

echo ""
done

echo "############################################################"
echo "  Ablation fixed_w1: ${DATASET} — DONE"
echo "############################################################"
