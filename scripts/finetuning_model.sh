#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src 

python ./expressive_vector/data_preprocess.py \
    --dataset sichuan

python ./F5-TTS/src/f5_tts/train/finetune_cli.py \
    --finetune \
    --pretrain ./ckpts/F5TTS_v1_Base/model_1250000.pt \
    --dataset_name sichuan

