#!/bin/bash

# TTAI: Test-Time Adaptive Interpolation evaluation
# Usage: bash ttai_test.sh <DATASET> <Clip_WEIGHT> <LOSS_TYPE> <COOP_LMC> <W_LMC> <SEED> <CFG> [SUB]
#        bash scripts/coop_LMC/ttai_test.sh ...  (from mergetune/)
#
# SUB defaults to "new". Pass "base" to evaluate on base classes.
#
# Workflow:
#   1. Run with SUB=base first — this trains the MLP and saves it to the output dir.
#   2. Run with SUB=new — this loads the saved MLP and evaluates on new classes.

# Navigate to user home where "${TRAIN_PY}" is accessible
# scripts/coop_LMC -> scripts -> mergetune -> MERGETUNE -> home
HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

TRAINER=KgCoOp_COOP_LMC
DATASET=$1
Clip_WEIGHT=$2
LOSS_TYPE=$3
COOP_LMC=$4
W_LMC=$5
SEED=$6
CFG=$7
SUB=${8:-new}
CTP=end
NCTX=4
SHOTS=16
CSC=False

for SEED in ${SEED}
do
    COMMON_DIR=${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/train_base/${DATASET}/${LOSS_TYPE}/shots_${SHOTS}_${Clip_WEIGHT}_${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/evaluate/ttai_test_${SUB}/${COMMON_DIR}
    RESUME_COOP=${OUTPUT}/coop/train_base/${DATASET}/shots_16/CoOp/vit_b16_ep100_ctxv1/seed${SEED}

    # MLP path: always points to the base evaluation output (where MLP was trained and saved)
    MLP_DIR=${OUTPUT}/KgCoOp_COOP_LMC/CoOp/evaluate/ttai_test_base/${COMMON_DIR}
    MLP_PATH=${MLP_DIR}/ttai_mlp.pth

    # Build TTAI MLP args
    TTAI_ARGS=""
    if [ "${SUB}" != "base" ] && [ -f "${MLP_PATH}" ]; then
        echo "Loading pre-trained TTAI MLP from ${MLP_PATH}"
        TTAI_ARGS="--ttai-mlp-path ${MLP_PATH}"
    elif [ "${SUB}" != "base" ] && [ ! -f "${MLP_PATH}" ]; then
        echo "WARNING: MLP not found at ${MLP_PATH}. Run with SUB=base first to train the MLP."
        echo "Proceeding without pre-trained MLP (will train on ${SUB} data instead)."
    fi

    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run TTAI evaluation (sub=${SUB}) and save to ${DIR}"
        python "${TRAIN_PY}" \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
            --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --eval-only-ttai \
            --resume-coop ${RESUME_COOP} \
            ${TTAI_ARGS} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
