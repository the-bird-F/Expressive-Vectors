import argparse
import codecs
import os
import re
import sys
from tqdm import tqdm
from datetime import datetime
from importlib.resources import files
from pathlib import Path
sys.path.append(os.getcwd() + "/../src/")

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from f5_tts.model.dataset import load_dataset, DynamicBatchSampler, collate_fn

from f5_tts.infer.utils_infer import load_model

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

import torch
import os
from collections import OrderedDict

def flatten_state_dict(d, parent_key='', sep='@'):
    """递归展开嵌套 OrderedDict"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, (OrderedDict, dict)):
            flattened = flatten_state_dict(v, new_key, sep)
            items.extend(flattened.items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_state_dict(flat_dict, sep='@'):
    """将扁平化的 state_dict 恢复为嵌套 dict"""
    unflat = dict()
    for flat_key, value in flat_dict.items():
        parts = flat_key.split(sep)
        d = unflat
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return unflat

def load_state_dict(path):
    """读取模型"""
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, (dict, OrderedDict)):
        return flatten_state_dict(checkpoint)
        
def compare_models(model_dir, model1, model2):
    """比较微调前后模型的数值差距"""
    state_dict1 = load_state_dict(os.path.join(model_dir,model1))
    state_dict2 = load_state_dict(os.path.join(model_dir,model2))

    diff_dict = {}
    diff_norms = {}

    for key in state_dict1:
        if key not in state_dict2:
            print(f"[info] model2 中缺少参数: {key}")
            continue

        param1, param2 = state_dict1[key], state_dict2[key]

        if not isinstance(param1, torch.Tensor) or not isinstance(param2, torch.Tensor):
            print(f"[info] 参数 {key} 不是 Tensor 类型，跳过")
            continue

        if param1.shape != param2.shape:
            print(f"[info] 参数维度不一致: {key}: {param1.shape} vs {param2.shape}")
            continue

        if not torch.is_floating_point(param1):
            print(f"[info] 非浮点类型参数: {key}: {param1} (dtype = {param1.dtype})")
            continue

        diff = param2 - param1
        diff_dict[key] = diff

        diff_norms[key] = diff.pow(2).mean().sqrt().item() # torch.norm(diff).item()
    
    return state_dict1, state_dict2, diff_dict, diff_norms

def normalize_dict_values(diff_norms):
    """
    对字典的 values 进行归一化，使其范围在 [0, 1] 之间
    """
    if not diff_norms:
        return {}
    
    values = list(diff_norms.values())
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return {k: 1.0 for k in diff_norms.keys()}
    
    normalized = {}
    for k, v in diff_norms.items():
        normalized[k] = (v - min_val) / (max_val - min_val)
    
    return normalized

def get_eval_loss(model, train_dataloader):    
    total_loss = 0.0
    total_batches = 0 
    model = model.float() 
    model.eval()
    for batch in tqdm(train_dataloader, desc=f"enhance"):
        text_inputs = batch["text"]
        mel_spec = batch["mel"].permute(0, 2, 1).to(model.device)
        mel_lengths = batch["mel_lengths"].to(model.device)
        loss, cond, pred = model(
            mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=None, 
        )
        total_loss += loss.item()
        total_batches += 1 
    return total_loss / total_batches

def draw_result(loss_list, alphas, args):
    # Create figure with constrained layout
    plt.figure(figsize=(8, 5), dpi=300, facecolor='white')
    ax = plt.axes()

    # Set modern style parameters
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

    # Plot with technical styling
    ax.plot(alphas, loss_list,
            color='#1f77b4',  # Professional blue
            marker='o',
            markersize=5,
            markerfacecolor='#ff7f0e',  # Complementary orange
            markeredgecolor='#1f77b4',
            markeredgewidth=0.8,
            linewidth=1.5,
            linestyle='-',
            alpha=0.9)

    # Enhanced scientific formatting
    ax.set_title('Enhancing Loss Convergence', pad=15, fontweight='semibold')
    ax.set_xlabel('Enhance degree', labelpad=8)
    ax.set_ylabel('Mel Loss', labelpad=8)

    # Advanced grid and tick formatting
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='major', linestyle='-', linewidth='0.5', alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth='0.5', alpha=0.4)

    # Professional annotation with LaTeX-style math
    min_loss = min(loss_list)
    mark_loss = (max(loss_list) + 2 * min_loss) / 3
    min_idx = alphas[loss_list.index(min_loss)]
    ax.annotate(rf'$\min\mathcal{{L}} = {min_loss:.4f} @ {min_idx:.1f}$',
                xy=(min_idx, min_loss),
                xytext=(min_idx, mark_loss),
                arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3',
                            linewidth=1,
                            color='#2ca02c'),
                bbox=dict(boxstyle='round',
                        facecolor='white',
                        alpha=0.8,
                        edgecolor='#2ca02c'),
                fontsize=10,
                color='#2ca02c')

    # Add technical details
    plt.text(0.02, 0.98,
            f'Dataset: {args.dataset_name}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            fontsize=9)

    # Final layout adjustments
    plt.tight_layout(pad=2.0)
    plt.savefig(args.save_fig_path,
                bbox_inches='tight',
                pad_inches=0.05,
                format='png',
                transparent=False)
    plt.close()  

def model_interpolate(ckpt_dir, model1, model2, alpha, beta=1.0, norm_flag=False, save_diff=False, save_name=None):

    state_dict1, state_dict2, diff_dict, diff_norms = compare_models(ckpt_dir, model1, model2)
    diff_norms_n = normalize_dict_values(diff_norms) 

    if save_diff:
        diff_nested = unflatten_state_dict(diff_dict)
        diff_path = os.path.join(ckpt_dir, f"diff_{model2}_{model1}")
        torch.save(diff_nested, diff_path)

    interpolated = {}
    for k, v in state_dict1.items():
        if k in diff_dict:
            if norm_flag:
                interpolated[k] = v + alpha * diff_dict[k] ** beta * diff_norms_n[k]
            else:
                interpolated[k] = v + alpha * diff_dict[k] ** beta
        else:
            interpolated[k] = v
            print("[warning] 缺少",{k})

    interpolated_nested = unflatten_state_dict(interpolated)
    interpolated_name = save_name or f'interpolated_{model2}_a{alpha:.1f}_b{beta:.1f}_n{str(norm_flag)}_{model1}.pt'
    interpolated_path = os.path.join(ckpt_dir, interpolated_name)
    torch.save(interpolated_nested, interpolated_path)

    return interpolated_path

def main(args, model_action_fn):
    tokenizer = "pinyin"
    batch_size_per_gpu = 8000
    vocoder_name = "vocos"
    model_name = "F5TTS_v1_Base"
    model_cfg_path = "src/f5_tts/configs/F5TTS_v1_Base.yaml"
    vocab_file = "data/vocab.txt" 
    
    model_cfg = OmegaConf.load(model_cfg_path).model
    model_cls =  get_class(f"f5_tts.model.{model_cfg.backbone}")
    mel_spec_kwargs = model_cfg.mel_spec
    
    train_dataset = load_dataset(args.dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    sampler = SequentialSampler(train_dataset)
    batch_sampler = DynamicBatchSampler(
                    sampler,
                    batch_size_per_gpu,
                    max_samples=64,
                    random_seed=666,
                    drop_residual=False,
                )
    train_dataloader = DataLoader(
                    train_dataset,
                    collate_fn=collate_fn,
                    num_workers=16,
                    pin_memory=True,
                    persistent_workers=True,
                    batch_sampler=batch_sampler,
                )
    
    loss_list = list()
    num_points = 21
    alphas = np.linspace(args.min_alpha, args.max_alpha, num_points)
    for alpha in alphas:
        print("[info] alpha:", alpha)
        ckpt_file = model_action_fn(ckpt_dir=args.ckpt_dir, model1=args.model1, model2=args.model2, alpha=alpha, save_name="tmp1.pt")
        model = load_model(model_cls, model_cfg.arch, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file)
        
        loss_list.append(get_eval_loss(model, train_dataloader))
        
        del model  
        torch.cuda.empty_cache()
        
    draw_result(loss_list, alphas, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="参数")
    parser.add_argument('--ckpt_dir', type=str, default="ckpts/ESD_Angry")
    parser.add_argument('--dataset_name', type=str, default="ESD_Angry_dev")
    parser.add_argument('--model1', type=str, default='pretrained_model_1250000.pt')
    parser.add_argument('--model2', type=str, default='model_80000.pt')
    parser.add_argument('--min_alpha', type=float, default=0)
    parser.add_argument('--max_alpha', type=float, default=4)   
    parser.add_argument('--save_fig_path', type=str, default="./dialect/Angry_8w.png")
    
    args = parser.parse_args()
    
    model_action_fn = model_interpolate
    
    main(args, model_action_fn)
    