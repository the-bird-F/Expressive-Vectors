import os
from httpx import get
import hydra
from hydra.utils import to_absolute_path, get_class
import shutil
import logging
from importlib.resources import files
from omegaconf import OmegaConf

from cached_path import cached_path

from f5_tts.model import CFM, DiT, Trainer, UNetT, MMDiT
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer

logging.basicConfig(
    level=logging.INFO,
)


def clean_config(d):
    clean_cfg = OmegaConf.to_container(d, resolve=True)
    return clean_cfg

# -------------------------- Training Settings -------------------------- #


@hydra.main(config_path=str(files("f5_tts").joinpath("configs")), config_name="F5TTS_v1_Base.yaml", version_base="1.3")
def main(config):
    if config.ckpts.pretrain is None:
        ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
    else:
        ckpt_path = to_absolute_path(config.ckpts.pretrain)

    save_dir = config.ckpts.get("save_dir", "exps/F5TTS_finetune")
    ckpt_save_dir = os.path.join(save_dir, "ckpts")
    os.makedirs(ckpt_save_dir, exist_ok=True)

    file_checkpoint = os.path.basename(ckpt_path)
    if not file_checkpoint.startswith("pretrained_"):  # Change: Add 'pretrained_' prefix to copied model
        file_checkpoint = "pretrained_" + file_checkpoint
    file_checkpoint = os.path.join(ckpt_save_dir, file_checkpoint)
    if not os.path.isfile(file_checkpoint):
        shutil.copy2(ckpt_path, file_checkpoint)
        print("copy checkpoint for finetune")

    tokenizer = config.model.tokenizer
    vocab_char_map, vocab_size = get_tokenizer(config.datasets.name, tokenizer)
    if tokenizer in ["pinyin", "char"]:
        vocab_file = os.path.join(
            files("f5_tts").joinpath("../../data"),
            f"{config.datasets.name}_{tokenizer}/vocab.txt"
        )
        shutil.copy2(vocab_file, os.path.join(save_dir, "vocab.txt"))

    print("\nvocab : ", vocab_size)
    print("\nvocoder : ", config.model.mel_spec.mel_spec_type)

    mel_spec_kwargs = dict(
        n_fft=config.model.mel_spec.n_fft,
        hop_length=config.model.mel_spec.hop_length,
        win_length=config.model.mel_spec.win_length,
        n_mel_channels=config.model.mel_spec.n_mel_channels,
        target_sample_rate=config.model.mel_spec.target_sample_rate,
        mel_spec_type=config.model.mel_spec.mel_spec_type,
    )

    model_cls = get_class(f"f5_tts.model.{config.model.backbone}")
    model_cfg = config.model.arch
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=config.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
        tokenized=config.model.tokenized,
        tokenizer=tokenizer,
    )

    if model_cfg.get("use_lora", False):
        model_mapping_file = os.path.join(
            files("f5_tts").joinpath("../../data"),
            f"{config.datasets.name}_{tokenizer}/lora_mapping.json"
        )
        shutil.copy2(model_mapping_file, os.path.join(save_dir, "lora_mapping.json"))

    optim_cfg = {
        "epochs": config.optim.get("epochs", 10),
        "learning_rate": config.optim.get("learning_rate", 5e-5),
        "num_warmup_updates": config.optim.get("num_warmup_updates", 10000),
        "grad_accumulation_steps": config.optim.get("grad_accumulation_steps", 1),
        "max_grad_norm": config.optim.get("max_grad_norm", 1.0),
        "bnb_optimizer": config.optim.get("bnb_optimizer", False),
    }
    ckpts_cfg = {
        "logger": config.ckpts.get("logger", "wandb"),
        "wandb_project": config.ckpts.get("wandb_project", "F5TTS"),
        "wandb_run_name": config.ckpts.get(
            "wandb_run_name",
            f"{config.model.name}_{config.model.mel_spec.mel_spec_type}_{tokenizer}_{config.datasets.name}",
        ),
        "log_samples": config.ckpts.get("log_samples", True),
        "save_per_updates": config.ckpts.get("save_per_updates", 10000),
        "keep_last_n_checkpoints": config.ckpts.get("keep_last_n_checkpoints", -1),
        "last_per_updates": config.ckpts.get("last_per_updates", 5000),
        "checkpoint_path": ckpt_save_dir,
    }
    trainer = Trainer(
        model,
        batch_size_per_gpu=config.datasets.batch_size_per_gpu,
        batch_size_type=config.datasets.batch_size_type,
        max_samples=config.datasets.max_samples,
        is_local_vocoder=config.model.vocoder.is_local,
        local_vocoder_path=to_absolute_path(config.model.vocoder.local_path),
        use_lora=model_cfg.get("use_lora", False),
        **optim_cfg,
        **ckpts_cfg,
    )

    train_dataset = load_dataset(
        config.datasets.name, 
        tokenizer, 
        audio_type="train",
        mel_spec_kwargs=mel_spec_kwargs,
        use_lora=model_cfg.get("use_lora", False),
    )

    valid_dataset = load_dataset(
        config.datasets.name,
        tokenizer, 
        audio_type="valid",
        mel_spec_kwargs=mel_spec_kwargs,
        use_lora=model_cfg.get("use_lora", False),
    )

    OmegaConf.save(clean_config(config), os.path.join(save_dir, "finetune_config.yaml"))

    trainer.train(
        train_dataset,
        valid_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
