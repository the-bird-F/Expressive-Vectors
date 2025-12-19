import argparse
import codecs
from html import parser
import os
import re
import json
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

parser = argparse.ArgumentParser(description="F5-TTS Inference CLI")

parser.add_argument(
    "--config-path",
    type=str,
    required=True,
    help="path to the config file",
)
parser.add_argument(
    "--ckpt_file",
    type=str,
    default=None,
    help="checkpoint file path for the TTS model",
)
parser.add_argument(
    "--vocab_file",
    type=str,
    default=None,
    help="vocab file path for the tokenizer",
)
parser.add_argument(
    "--ref_audio",
    type=str,
    required=True,
    help="path to reference audio file",
)
parser.add_argument(
    "--ref_text",
    type=str,
    required=True,
    help="reference text corresponding to the reference audio",
)
parser.add_argument(
    "--gen_text",
    type=str,
    required=True,
    help="text to be synthesized",
)
parser.add_argument(
    "--lora_label",
    type=str,
    nargs="+",
    default=None,
    help="list of lora labels corresponding to each lora dimension",
)
parser.add_argument(
    "--lora_mapping_file",
    type=str,
    default=None,
    help="path to lora mapping json file",
)
parser.add_argument(
    "--lora_alpha",
    type=float,
    default=1.0,
    help="scaling factor for LoRA modules",
)
parser.add_argument(
    "--target_rms",
    type=float,
    default=target_rms,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    default=cross_fade_duration,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    default=nfe_step,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    default=cfg_strength,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    default=sway_sampling_coef,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    default=speed,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    default=fix_duration,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--device",
    type=str,
    default=device,
    help="Specify the device to run on",
)
args = parser.parse_args()

# inference process

def main():
    config = OmegaConf.load(args.config_path)
    model = config.model.backbone
    save_dir = config.ckpts.save_dir

    if args.ckpt_file is None:
        ckpt_file = os.path.join(save_dir, "ckpts/model_last.pt")
    else:
        ckpt_file = args.ckpt_file

    if args.vocab_file is None:
        vocab_file = os.path.join(save_dir, "vocab.txt")
    else:
        vocab_file = args.vocab_file


    ref_audio = args.ref_audio
    ref_text = args.ref_text
    gen_text = args.gen_text

    use_lora = config.model.arch.get("use_lora", False)
    output_dir = os.path.join(save_dir, "examples")
    if use_lora:
        lora_label = args.lora_label
        output_file = f"{'_'.join(lora_label)}_{args.lora_alpha}.wav"
        if args.lora_mapping_file is None:
            lora_mapping_file = os.path.join(save_dir, "lora_mapping.json")
        else:
            lora_mapping_file = args.lora_mapping_file
        lora_map = json.load(open(lora_mapping_file, "r", encoding="utf-8"))
        lora_idx = []
        for i, (label_name, label_map) in enumerate(lora_map.items()):
            lora_idx.append(label_map[lora_label[i]])
            logging.info(f"{label_name}: {lora_label[i]} -> {label_map[lora_label[i]]}")
    else:
        output_file = "output.wav"
        lora_idx = None

    target_rms = args.target_rms
    cross_fade_duration = args.cross_fade_duration
    nfe_step = args.nfe_step
    cfg_strength = args.cfg_strength
    sway_sampling_coef = args.sway_sampling_coef
    speed = args.speed
    fix_duration = args.fix_duration
    device = args.device


    # output path

    wave_path = Path(output_dir) / output_file

    vocoder = load_vocoder(
        vocoder_name=config.model.mel_spec.mel_spec_type, 
        is_local=config.model.vocoder.is_local, 
        local_path=config.model.vocoder.local_path, 
        device=device
    )

    # load TTS model

    model_cls = get_class(f"f5_tts.model.{model}")
    model_arc = config.model.arch
    if use_lora:
        model_arc["lora_alpha"] = args.lora_alpha

    print(f"Using {model}...")
    ema_model = load_model(
        model_cls, 
        model_arc, 
        ckpt_file, 
        mel_spec_type=config.model.mel_spec.mel_spec_type, 
        vocab_file=vocab_file, 
        device=device
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
        local_speed = voices[voice].get("speed", speed)
        gen_text_ = text.strip()
        print(f"Voice: {voice}")
        audio_segment, final_sample_rate, spectrogram = infer_process(
            ref_audio_,
            ref_text_,
            gen_text_,
            ema_model,
            vocoder,
            mel_spec_type=config.model.mel_spec.mel_spec_type,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=local_speed,
            fix_duration=fix_duration,
            device=device,
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
