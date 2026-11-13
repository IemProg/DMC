#!/bin/bash

# Ablation: Is the learnable generalization prompt necessary?
# Compares f_gen (learned) vs w1 (zero-shot) vs w2 (CoOp) as interpolation endpoints
# Uses existing DMC+VA checkpoints — no retraining needed
# Evaluates on base and new splits for each dataset/seed

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
CFG=vit_b16_ep100_ctxv1
W_GEN=8.0
W_LMC=4.0
VA_W=0.5
CTP=end
NCTX=4
SHOTS=16
CSC=False

for DATASET in oxford_flowers caltech101 stanford_cars eurosat
do
    for SEED in 1 2 3
    do
        COMMON_DIR=${DATASET}/dpp_wgen${W_GEN}_wlmc${W_LMC}_va${VA_W}/${TRAINER}/${CFG}/seed${SEED}
        MODEL_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base_dpp_va/${COMMON_DIR}

        for SUB in base new
        do
            DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/evaluate/dpp_ablation_gen/test_${SUB}/${COMMON_DIR}

            if [ -d "$DIR" ]; then
                echo "Results available: ${DIR}. Skip."
            else
                echo "======================================================"
                echo "  Ablation: ${DATASET} seed${SEED} ${SUB}"
                echo "======================================================"
                python "${TRAIN_PY}" \
                    --root ${DATA} \
                    --seed ${SEED} \
                    --trainer ${TRAINER} \
                    --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
                    --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
                    --output-dir ${DIR} \
                    --model-dir ${MODEL_DIR} \
                    --eval-only-dpp-ablation \
                    --resume-coop None \
                    TRAINER.COOP.N_CTX ${NCTX} \
                    TRAINER.COOP.CSC ${CSC} \
                    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                    TRAINER.COOP.DPP True \
                    DATASET.NUM_SHOTS ${SHOTS} \
                    DATASET.SUBSAMPLE_CLASSES ${SUB}
            fi
        done
        echo ""
    done
done
