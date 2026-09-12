#!/bin/bash

# Compute gradient conflict γ for MERGETUNE checkpoints
# Usage: bash gradient_conflict.sh
#
# Computes γ = cos(∇L_CE, ∇R) at the MERGETUNE convergence point
# for each dataset and seed. No retraining needed.

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
CFG=vit_b16_ep100_ctxv1
CTP=end
NCTX=4
SHOTS=16
CSC=False

for DATASET in caltech101 oxford_flowers oxford_pets food101
do
    for SEED in 1 2 3
    do
        MODEL_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base/${DATASET}/cosine/shots_${SHOTS}_8.0_1.0/${TRAINER}/${CFG}/seed${SEED}
        OUT_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/evaluate/gradient_conflict/${DATASET}/${TRAINER}/${CFG}/seed${SEED}

        if [ ! -d "$MODEL_DIR" ] || ! ls ${MODEL_DIR}/prompt_mid_learner/model* > /dev/null 2>&1; then
            echo "SKIP ${DATASET} seed${SEED}: no MERGETUNE checkpoint"
            continue
        fi

        if [ -d "$OUT_DIR" ]; then
            echo "SKIP ${DATASET} seed${SEED}: already computed"
            continue
        fi

        echo "======================================================"
        echo "  Gradient conflict: ${DATASET} seed ${SEED}"
        echo "======================================================"
        python "${TRAIN_PY}" \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${OUT_DIR} \
            --model-dir ${MODEL_DIR} \
            --eval-gradient-conflict \
            --resume-coop None \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base
        echo ""
    done
done

echo "======================================================"
echo "  DONE — check logs for γ values"
echo "======================================================"
