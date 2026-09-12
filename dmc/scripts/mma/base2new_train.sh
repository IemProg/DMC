#!/bin/bash

HERE="$(cd "${HERE}" && pwd)"
source "${HERE}/../env.sh"

# custom config
TRAINER=MultiModalAdapter

DATASET=$1
SEED=$2
CFG=$3
SHOTS=16


DIR=${OUTPUT}/MMA/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python "${TRAIN_PY}" \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file ${CONFIGS}/datasets/${DATASET}.yaml \
    --config-file ${CONFIGS}/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi