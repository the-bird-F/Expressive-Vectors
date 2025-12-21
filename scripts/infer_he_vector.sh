#!/bin/bash

python -m f5_tts_lora/infer/infer_cli.py \
  --config-path f5_tts_lora/exps/F5TTS_v1_Base_vocos_pinyin_Speech_ESD/2025-12-19_22-21-51 \
  --config-name finetune_config \
  ++model.arch.lora_alpha=5.0 \
  "++gen.ref_text='妈妈从我手中拿走我歌曲的歌看，边看边赞不绝口。'" \
  ++gen.ref_audio="examples/basic/mandarin_prompt.wav" \
  "++gen.gen_text='成都嘛巴适得板有火锅茶馆还有萌萌哒的大熊猫来了就不想走咯'" \
  ++gen.lora_label="['Sichuan','Happy']" \
  ++gen.device="cuda:0"