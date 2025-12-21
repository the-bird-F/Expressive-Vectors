#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src 

CKPT_LIST=(
    "model_60000.pt"
    "model_80000.pt"
)

for CKPT in "${CKPT_LIST[@]}"; do

    python ./expressive_vector/mining_model.py \
        --ckpt_dir ckpts/Tianjing \
        --dataset_name Tianjing_dev \
        --model2 "$CKPT" \
        --min_alpha 1.5 \
        --max_alpha 3.5 \
        --save_fig_path "./mining/Tianjing_${CKPT%.*}.png"

done
