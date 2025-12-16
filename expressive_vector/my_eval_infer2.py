import os
import sys
import csv
sys.path.append(os.getcwd())

import argparse
import time
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from f5_tts.eval.utils_eval import (
    get_inference_prompt,
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
)
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer


accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"


use_ema = True
target_rms = 0.1

rel_path = str(files("f5_tts").joinpath("../../"))

def load_ref_data(ref_csv, base_wav_dir):
    ref_data = []
    with open(ref_csv, 'r', encoding='utf-8') as csvfile:
        next(csvfile)   
        reader = csv.reader(csvfile, delimiter='|')
        for row in reader:
            try:
                ref_audio, ref_text = row
                ref_audio_path = os.path.join(base_wav_dir, ref_audio)
                ref_data.append((ref_audio_path, ref_text))
            except Exception as e:
                print(f"Error loading ref data for row {row}: {str(e)}")
                continue
    
    return ref_data

def get_dialect_metainfo(input_csv, base_wav_dir, ref_csv):
    ref_data = load_ref_data(ref_csv, base_wav_dir)
    metainfo = []
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile, delimiter='|')
        idx = 0
        for row in reader:
            if idx >= len(ref_data):
                break
        
            gt_wav, gt_text = row
            if len(gt_text) < 5:
                continue
            ref_audio, ref_text = ref_data[idx]
            ref_audio = os.path.join(base_wav_dir, ref_audio)
            idx += 1

            utt = gt_wav.split('/')[-1].split('.')[0]
            metainfo.append((utt, ref_text, ref_audio, gt_text, gt_wav))
            
    return metainfo


def main():
    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("-s", "--seed", default=None, type=int)
    parser.add_argument("-n", "--expname", default="F5TTS_v1_Base")
    parser.add_argument("-c", "--ckptpath", required=True, type=str)
    parser.add_argument("-o", "--output_dir", default=None)

    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-ode", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)

    parser.add_argument("-t", "--testset", required=True)
    parser.add_argument("-r", "--ref_csv", required=True)
    parser.add_argument("-b", "--base_dir", required=True)

    args = parser.parse_args()

    seed = args.seed
    exp_name = args.expname
    ckpt_path = args.ckptpath

    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling

    testset = args.testset
    base_wav_dir = args.base_dir
    ref_csv = args.ref_csv
    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    cfg_strength = 2.0
    speed = 1.0
    use_truth_duration = False
    no_ref_audio = False

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    dataset_name = model_cfg.datasets.name
    tokenizer = model_cfg.model.tokenizer

    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft

    if os.path.exists(testset) :
        metainfo = get_dialect_metainfo(testset, base_wav_dir, ref_csv)
    else:
        raise ValueError(f"testset error: get line {testset}")

    # path to save genereted wavs
    output_dir = args.output_dir or (
        f"{rel_path}/"
        f"results/{exp_name}_new_{ckpt_path}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"_cfg{cfg_strength}_speed{speed}"
        f"{'_gt-dur' if use_truth_duration else ''}"
        f"{'_no-ref-audio' if no_ref_audio else ''}"
    )

    # -------------------------------------------------#
    prompts_all = get_inference_prompt(
        metainfo,
        speed=speed,
        tokenizer=tokenizer,
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
        mel_spec_type=mel_spec_type,
        target_rms=target_rms,
        use_truth_duration=use_truth_duration,
        infer_batch_size=infer_batch_size,
    )

    # Vocoder model
    local = False
    if mel_spec_type == "vocos":
        vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

    # Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # Model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)

    # start batch inference
    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = prompt
            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
            total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)

            # Inference
            with torch.inference_mode():
                generated, _ = model.sample(
                    cond=ref_mels,
                    text=final_text_list,
                    duration=total_mel_lens,
                    lens=ref_mel_lens,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    no_ref_audio=no_ref_audio,
                    seed=seed,
                )
                # Final result
                for i, gen in enumerate(generated):
                    gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
                    gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
                    if mel_spec_type == "vocos":
                        generated_wave = vocoder.decode(gen_mel_spec).cpu()
                    elif mel_spec_type == "bigvgan":
                        generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

                    if ref_rms_list[i] < target_rms:
                        generated_wave = generated_wave * ref_rms_list[i] / target_rms
                    torchaudio.save(f"{output_dir}/{utts[i]}.wav", generated_wave, target_sample_rate)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
