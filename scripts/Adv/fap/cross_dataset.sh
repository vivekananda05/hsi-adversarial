#!/bin/bash

# running command:
# CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/cross_dataset.sh 0

ADV_TERM=cos
EPS=1
ALPHA=1
TRAIN_ITER=2
TEST_ITER=50
LAMBDA_1=1.5


ATTACK_RELATED=adv_term-${ADV_TERM}/eps-${EPS}_alpha-${ALPHA}_train_iter-${TRAIN_ITER}_test_iter-${TEST_ITER}_lambda1-${LAMBDA_1}
#2. dataset_related
DATASET=imagenet
SHOTS=16

DATASET_RELATED=${DATASET}_shots_${SHOTS}

# 3. model_related 
CFG=vit_b32_ep5_batch4_2ctx_notransform

TRAINER=FAP

MODEL_RELATED=${TRAINER}_${CFG}

#4. seed
SEED=${1}

SEED_RELATED=seed${SEED}

# 4. path_related
#modify this
OUTPUT_DIR=/output_dir

TRAIN_DIR=${OUTPUT_DIR}/cross_dataset/imagenet_shot${SHOTS}/${ATTACK_RELATED}/${MODEL_RELATED}/${SEED_RELATED}

DATA=/data



        if [ -d "$TRAIN_DIR" ]; then
            echo "Oops! The training results exist at ${TRAIN_DIR} (so skip this job)"
        else
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/imagenet.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${TRAIN_DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            ATTACK.HSI_PGD.ADV_TERM ${ADV_TERM}\
            ATTACK.HSI_PGD.TRAIN_ITER ${TRAIN_ITER}\
            ATTACK.HSI_PGD.TEST_ITER ${TEST_ITER}\
            ATTACK.HSI_PGD.EPS ${EPS}\
            ATTACK.HSI_PGD.LAMBDA_1 ${LAMBDA_1}\
            ATTACK.HSI_PGD.ALPHA ${ALPHA}
        fi

# Testing section


# Test datasets array
TEST_DATASETS=(caltech101 dtd eurosat oxford_pets oxford_flowers stanford_cars food101 fgvc_aircraft sun397 ucf101)

for TEST_DATASET in "${TEST_DATASETS[@]}"
do
    TEST_DIR=${OUTPUT_DIR}/cross_dataset/evaluation/${ATTACK_RELATED}/test_dataset_${TEST_DATASET}/${MODEL_RELATED}/${SEED_RELATED}
    echo ${TEST_DIR}

    if [ -d "$TEST_DIR" ]; then
        echo "Oops! The testing results exist at ${TEST_DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${TEST_DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${TEST_DIR} \
        --model-dir ${TRAIN_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        ATTACK.HSI_PGD.TRAIN_ITER ${TRAIN_ITER}\
        ATTACK.HSI_PGD.TEST_ITER ${TEST_ITER}\
        ATTACK.HSI_PGD.EPS ${EPS}\
        ATTACK.HSI_PGD.ALPHA ${ALPHA}

    fi
done
