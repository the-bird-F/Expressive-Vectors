#!/bin/bash

cd f5_tts_lora

python -m f5_tts.infer.infer_cli \
  --config-path /inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/yaoxiao/F5-HE-Vector/f5_tts_lora/src/f5_tts/configs/ \
  --config-name F5TTS_v1_Base \
  hydra.run.dir="exps/F5TTS_v1_Base" \
  ++gen.ckpt_file="/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/yaoxiao/F5-HE-Vector/f5_tts_lora/exps/F5TTS_v1_Base/model_1250000.safetensors" \
  "++gen.ref_text='妈妈从我手中拿走我歌曲的歌看，边看边赞不绝口。'" \
  ++gen.ref_audio="examples/basic/mandarin_prompt.wav" \
  "++gen.gen_text='成都嘛巴适得板有火锅茶馆还有萌萌哒的大熊猫来了就不想走咯'" \
  ++gen.device="cuda:0"