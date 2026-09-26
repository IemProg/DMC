#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

# custom config
TRAINER=MMA_LMC
DATASET=$1
LOSS_TYPE=cosine
COOP_LMC=True
W_LMC=$2
CFG=$3
SEED=$4
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
NUM_SAMPLES=5

for SEED in ${SEED}
do
    DIR=${OUTPUT}/MMA_LMC/base2new/train_base/${DATASET}/shots_${SHOTS}_${W_LMC}/${TRAINER}/${CFG}/seed${SEED}
    RESUME=${OUTPUT}/MMA/base2new/train_base/${DATASET}/shots_16/MultiModalAdapter/seed${SEED}
    
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"        
        PYTHONPATH=Dassl.ProGrad.pytorch:$PYTHONPATH \
        python "${TRAIN_PY}" \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
        --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --resume-coop ${RESUME} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.COOP.LOSS_TYPE ${LOSS_TYPE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.COOP.W_LMC ${W_LMC} \
        TRAINER.COOP.COOP_LMC ${COOP_LMC} \
        TRAINER.COOP.NUM_SAMPLES ${NUM_SAMPLES} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done
