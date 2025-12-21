#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src 

# define dialect and emotion
declare -A emotion=(
    ["Happy"]="开心地说"
    ["Angry"]="伤心地说"
    ["Sad"]="生气地说"
    ["Surprise"]="惊讶地说"
)

declare -A styles=(
    ["Sichuan"]="用四川话"
    ["Guangzhou"]="用广东话"
    ["Shanghai"]="用上海话"
    ["Tianjing"]="用天津话"
    ["Zhengzhou"]="用河南话"
    ["Changsha"]="用湖南话"
    ["Jinan"]="用山东话"
    ["Xian"]="用陕西话"
)


# path
BASE_INPUT_PATH="./dataset/MagicData_use"
BASE_REF_CSV="./dataset/Prompt_use/prompt10.csv"
BASE="./dataset/Prompt_use/"
BASE_OUTPUT_PATH="./results/F5TTS_multi_full"
BEST_A=3.0
SEED=0
NFE=32


for style in "${!styles[@]}"; do
    for emo in "${!emotion[@]}"; do  

        echo "=============================================="
        echo "Processing style: $style, emotion: $emo"
        echo "=============================================="
        
        INPUT_CSV="$BASE_INPUT_PATH/$style/eval.csv"
        OUTPUT_DIR="$BASE_OUTPUT_PATH/$style/$emo"

        python ./expressive_vector/model_action_multi.py \
                    --model_dir1 "ckpts/${style}" \
                    --model_dir2 "ckpts/ESD_${emo}" \
                    --model1 "model_60000.pt" \
                    --model2 "model_80000.pt"  \
                    --alpha1 $BEST_A \
                    --alpha2 $BEST_A \
                    --output_model "ckpts/test/test.pt"

        accelerate launch ./expressive_vector/my_eval_infer2.py \
            -s $SEED \
            -c "ckpts/test/test.pt" \
            -t $INPUT_CSV \
            -b $BASE \
            -r $BASE_REF_CSV \
            -nfe $NFE \
            -o $OUTPUT_DIR

        echo "Finished processing style: $style, emotion: $emo"
        echo ""
    done 
done

echo "All styles processed!"