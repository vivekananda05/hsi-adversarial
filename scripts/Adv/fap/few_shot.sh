#!/bin/bash

# Run from FAP (or use the absolute script path):
# CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/few_shot.sh indian_pines 16 0
# Optional environment overrides: DATA_ROOT, OUTPUT_DIR, PYTHON.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <hsi_dataset> <shots> <seed>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FAP_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${FAP_ROOT}"

# Config names match configs/datasets/*.yaml and the HSI_DATA folders.
# Supported: houstonu indian_pines ksc paviac paviau salinas
DATASET="${1,,}"
case "${DATASET}" in
    houstonu|indian_pines|ksc|paviac|paviau|salinas) ;;
    *) echo "Unsupported HSI dataset: ${DATASET}" >&2; exit 2 ;;
esac

DATA="${DATA_ROOT:-/home/vivek/HSI_DATA}"
OUTPUT_DIR="${OUTPUT_DIR:-${FAP_ROOT}/output}"
PYTHON="${PYTHON:-python}"
CFG=hsi_vit_b32
TRAINER=FAP

# Values are divided by 255 once inside attack_cube_pgd.
EPS=1
ALPHA=1
TRAIN_ITER=2
TEST_ITER=50
ADV_TERM=cos
LAMBDA_1=1.5

ATTACK_OPTIONS=(
    ATTACK.HSI_PGD.EPS "${EPS}"
    ATTACK.HSI_PGD.ALPHA "${ALPHA}"
    ATTACK.HSI_PGD.TRAIN_ITER "${TRAIN_ITER}"
    ATTACK.HSI_PGD.TEST_ITER "${TEST_ITER}"
    ATTACK.HSI_PGD.ADV_TERM "${ADV_TERM}"
    ATTACK.HSI_PGD.LAMBDA_1 "${LAMBDA_1}"
)
ATTACK_RELATED="adv_term-${ADV_TERM}/eps-${EPS}_alpha-${ALPHA}_train_iter-${TRAIN_ITER}_test_iter-${TEST_ITER}_lambda1-${LAMBDA_1}"
MODEL_RELATED="${TRAINER}_${CFG}"

SHOTS="$2"
SEED="$3"
TRAIN_DIR="${OUTPUT_DIR}/few_shot/${ATTACK_RELATED}/${MODEL_RELATED}/${DATASET}_shots_${SHOTS}/seed${SEED}"
CHECKPOINT="${TRAIN_DIR}/MultiModalPromptLearner/model.pth.tar-20"

if [[ -f "${CHECKPOINT}" ]]; then
    echo "Training checkpoint exists: ${CHECKPOINT}"
else
    "${PYTHON}" train.py \
        --root "${DATA}" \
        --seed "${SEED}" \
        --trainer "${TRAINER}" \
        --dataset-config-file "configs/datasets/${DATASET}.yaml" \
        --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
        --output-dir "${TRAIN_DIR}" \
        DATASET.NUM_SHOTS "${SHOTS}" \
        DATASET.SUBSAMPLE_CLASSES all \
        "${ATTACK_OPTIONS[@]}"
fi
