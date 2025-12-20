import argparse
import codecs
from html import parser
import os
import re
import json
from weakref import ref
import hydra
import logging
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    nfe_step,
    preprocess_ref_audio_text,
    speed,
    sway_sampling_coef,
    target_rms,
)


logging.basicConfig(
    level=logging.INFO,
)


# inference process

@hydra.main(config_path=None, config_name=None, version_base="1.3")
def main(config):
    save_dir = config.ckpts.save_dir
    model = config.model.backbone

    ckpt_file = config.gen.get("ckpt_file", os.path.join(save_dir, "ckpts/model_last.pt"))
    vocab_file = config.model.get("vocab_file", os.path.join(save_dir, "vocab.txt"))

    ref_audio = config.gen.ref_audio
    ref_text = config.gen.ref_text
    gen_text = config.gen.gen_text

    use_lora = config.model.arch.get("use_lora", False)
    lora_label = config.gen.get("lora_label", None)
    if use_lora and lora_label is not None:
        lora_alpha = config.model.arch.get("lora_alpha", None)
        if lora_alpha is None:
            output_file = f"{'_'.join(lora_label)}.wav"
        else:
            output_file = f"{'_'.join(lora_label)}_alpha_{lora_alpha}.wav"
        lora_mapping_file = config.gen.get("lora_mapping_file", os.path.join(save_dir, "lora_mapping.json"))
        lora_map = json.load(open(lora_mapping_file, "r", encoding="utf-8"))
        lora_idx = []
        for i, (label_name, label_map) in enumerate(lora_map.items()):
            lora_idx.append(label_map[lora_label[i]])
            logging.info(f"{label_name}: {lora_label[i]} -> {label_map[lora_label[i]]}")
    else:
        output_file = "output.wav"
        lora_idx = None

    cfg_target_rms = config.gen.get("target_rms", target_rms)
    cfg_cross_fade_duration = config.gen.get("cross_fade_duration", cross_fade_duration)
    cfg_nfe_step = config.gen.get("nfe_step", nfe_step)
    cfg_cfg_strength = config.gen.get("cfg_strength", cfg_strength)
    cfg_sway_sampling_coef = config.gen.get("sway_sampling_coef", sway_sampling_coef)
    cfg_speed = config.gen.get("speed", speed)
    cfg_fix_duration = config.gen.get("fix_duration", fix_duration)
    cfg_device = config.gen.get("device", device)

    # output path

    output_dir = config.gen.get("output_dir", os.path.join(save_dir, "examples"))
    wave_path = Path(output_dir) / output_file

    vocoder = load_vocoder(
        vocoder_name=config.model.mel_spec.mel_spec_type, 
        is_local=config.model.vocoder.is_local, 
        local_path=config.model.vocoder.local_path, 
        device=cfg_device
    )

    # load TTS model

    model_cls = get_class(f"f5_tts.model.{model}")
    model_arc = config.model.arch

    print(f"Using {model}...")
    ema_model = load_model(
        model_cls, 
        model_arc, 
        ckpt_file, 
        mel_spec_type=config.model.mel_spec.mel_spec_type, 
        vocab_file=vocab_file, 
        device=cfg_device
    )

    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice]["ref_audio"]
        ref_text_ = voices[voice]["ref_text"]
        gen_text_ = text.strip()
        print(f"Voice: {voice}")
        audio_segment, final_sample_rate, spectrogram = infer_process(
            ref_audio_,
            ref_text_,
            gen_text_,
            ema_model,
            vocoder,
            mel_spec_type=config.model.mel_spec.mel_spec_type,
            target_rms=cfg_target_rms,
            cross_fade_duration=cfg_cross_fade_duration,
            nfe_step=cfg_nfe_step,
            cfg_strength=cfg_cfg_strength,
            sway_sampling_coef=cfg_sway_sampling_coef,
            speed=cfg_speed,
            fix_duration=cfg_fix_duration,
            device=cfg_device,
            lora_idx=lora_idx,
        )
        generated_audio_segments.append(audio_segment)

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            logging.info(f"Saved generated audio to {f.name}")


if __name__ == "__main__":
    main()
