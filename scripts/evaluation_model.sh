#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src 

DataName=sichuan
BEST_A=2.6
BEST_MODEL=model_60000.pt
EVAL="./Dialect_Dataset/MagicData_use/ref_sichuan_eval.csv"
BASE="./Dialect_Dataset/MagicDialect"

SEED=0
NFE=32
RESULT_DIR="ref_changsha_eval/seed${SEED}_euler_nfe${NFE}_vocos_ss-1_cfg2.0_speed1.0"

python Expressive-Vectors/expressive_vector/model_action.py \
            --model_dir "ckpts/${DataName}" \
            --model2 $BEST_MODEL \
            --alpha $BEST_A


accelerate launch Expressive-Vectors/expressive_vector/my_eval_infer.py \
    -s $SEED \
    -c "ckpts/${DataName}/interpolated_${BEST_MODEL}_a${BEST_A}_nFalse.pt" \
    -t $EVAL \
    -b $BASE \
    -nfe $NFE

accelerate launch Expressive-Vectors/expressive_vector/my_eval_infer.py \
    -s $SEED \
    -c "ckpts/${DataName}/${BEST_MODEL}" \
    -t $EVAL \
    -b $BASE \
    -nfe $NFE

accelerate launch Expressive-Vectors/expressive_vector/my_eval_infer.py \
    -s $SEED \
    -c "ckpts/${DataName}/model_last.pt" \
    -t  $EVAL \
    -b  $BASE \
    -nfe $NFE

### choose
# MODEL_DIRS=(
#     "./results/F5TTS_v1_Base_ckpts/${DataName}/interpolated_${BEST_MODEL}_a${BEST_A}_nFalse.pt"
#     "./results/F5TTS_v1_Base_ckpts/${DataName}/${BEST_MODEL}"
#     "./results/F5TTS_v1_Base_ckpts/${DataName}/model_last.pt"
# )

# python dialect/seed_asr.py \
#   --csv_file $EVAL \
#   --model_dirs "${MODEL_DIRS[@]}" \
#   --out_dir "./dialect/seed_asr_result/${DataName}"  \
#   --base_wav_dir $BASE \
#   --result_wav_dir $RESULT_DIR