accelerate launch -m f5_tts.train.finetune_cli \
    datasets.name=Speech_ESD \
    datasets.batch_size_per_gpu=16000 \
    optim.epochs=200 \
    model.tokenized=True\
    model.tokenizer=pinyin \
    model.vocoder.is_local=True \
    model.vocoder.local_path=./src/third_party/vocos \
    ++model.arch.use_lora=True \
    ++model.arch.lora_rank=64 \
    ++model.arch.lora_feature_dim="[11,5]" \
    ++ckpts.pretrain=exps/F5TTS_v1_Base/model_1250000.safetensors \
    ckpts.logger=wandb \
    ckpts.wandb_project=F5TTS_v1_LoRA \
    ckpts.save_per_updates=10000 \
    ckpts.last_per_updates=2000 \