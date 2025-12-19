python -m f5_tts.infer.infer_cli \
    --config-path exps/F5TTS_v1_Base_vocos_pinyin_Speech_ESD/2025-12-19_22-21-51/finetune_config.yaml \
    --ref_text "妈妈从我手中拿走我歌曲的歌看，边看边赞不绝口。" \
    --ref_audio "examples/basic/mandarin_prompt.wav" \
    --gen_text "成都嘛巴适得板有火锅茶馆还有萌萌哒的大熊猫来了就不想走咯" \
    --lora_label Sichuan Happy \
    --lora_alpha 1.6 \
    --device "cuda:0" \