"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    precompute_freqs_cis,
    LoRALinear,
    LoRAEmbedding,
    LoRAConv1d,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(
        self, 
        text_num_embeds, 
        text_dim, 
        mask_padding=True, 
        average_upsampling=False, 
        conv_layers=0, 
        conv_mult=2,
        use_lora: bool = False,
        lora_rank: int | None = None,
        lora_feature_dim: int | None = None,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        self.average_upsampling = average_upsampling  # zipvoice-style text late average upsampling (after text encoder)
        if average_upsampling:
            assert mask_padding, "text_embedding_average_upsampling requires text_mask_padding to be True"

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192  # 8192 is ~87.38s of 24khz audio; 4096 is ~43.69s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult, use_lora=use_lora, lora_rank=lora_rank, lora_feature_dim=lora_feature_dim) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False
        

    def average_upsample_text_by_mask(self, text, text_mask):
        batch, text_len, text_dim = text.shape

        audio_len = text_len  # cuz text already padded to same length as audio sequence
        text_lens = text_mask.sum(dim=1)  # [batch]

        upsampled_text = torch.zeros_like(text)

        for i in range(batch):
            text_len = text_lens[i].item()

            if text_len == 0:
                continue

            valid_ind = torch.where(text_mask[i])[0]
            valid_data = text[i, valid_ind, :]  # [text_len, text_dim]

            base_repeat = audio_len // text_len
            remainder = audio_len % text_len

            indices = []
            for j in range(text_len):
                repeat_count = base_repeat + (1 if j >= text_len - remainder else 0)
                indices.extend([j] * repeat_count)

            indices = torch.tensor(indices[:audio_len], device=text.device, dtype=torch.long)
            upsampled = valid_data[indices]  # [audio_len, text_dim]

            upsampled_text[i, :audio_len, :] = upsampled

        return upsampled_text

    def forward(self, text: int["b nt"], seq_len, drop_text=False, lora_idx: torch.Tensor | int | None = None):
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0)  # (opt.) if not self.average_upsampling:
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            text = text + self.freqs_cis[:seq_len, :]

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text, lora_idx)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text, lora_idx)

        if self.average_upsampling:
            text = self.average_upsample_text_by_mask(text, ~text_mask)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(
        self, 
        mel_dim, 
        text_dim, 
        out_dim,
        use_lora: bool = False,
        lora_rank: int | None = None,
        lora_feature_dim: int | None = None,
    ):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim, use_lora=use_lora, lora_rank=lora_rank, lora_feature_dim=lora_feature_dim)
        self.use_lora = use_lora

    def forward(
        self,
        x: float["b n d"],
        cond: float["b n d"],
        text_embed: float["b n d"],
        drop_audio_cond=False,
        audio_mask: bool["b n"] | None = None,
        lora_idx: torch.Tensor | int | None = None,
    ):
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)
        x = torch.cat((x, cond, text_embed), dim=-1)

        x = self.proj(x)
        x = self.conv_pos_embed(x, mask=audio_mask, lora_idx=lora_idx) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        use_lora: bool = False,
        lora_rank: int | None = None,
        lora_feature_dim: list | None = None,
    ):
        super().__init__()

        self.use_lora = use_lora
        if use_lora:
            self.lora_proj_out = LoRALinear(dim, mel_dim, lora_rank=lora_rank, lora_feature_dim=lora_feature_dim[0] if lora_feature_dim else None)
        self.time_embed = TimestepEmbedding(
            dim, 
            use_lora=use_lora, 
            lora_rank=lora_rank, 
            lora_feature_dim=lora_feature_dim[0] if lora_feature_dim else None
        )
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_feature_dim=lora_feature_dim[0] if lora_feature_dim else None,
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(
            mel_dim, 
            text_dim, 
            dim,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_feature_dim=lora_feature_dim[0] if lora_feature_dim else None,
        )

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.lora_map = {i: i * len(lora_feature_dim) // depth for i in range(depth)} if lora_feature_dim else {}
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                    use_lora=use_lora,
                    lora_rank=lora_rank,
                    lora_feature_dim=lora_feature_dim[self.lora_map[i]] if lora_feature_dim else None,
                )
                for i in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(
            dim, 
            use_lora=use_lora, 
            lora_rank=lora_rank,
            lora_feature_dim=lora_feature_dim[0] if lora_feature_dim else None,
        )  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

        if self.use_lora:
            for name, param in self.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"total_params: {total_params}, trainable_params: {trainable_params}")

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        text,  # b nt
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: bool["b n"] | None = None,
        lora_idx: torch.Tensor | int | None = None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                text_embed = self.text_embed(text, x.shape[1], drop_text=drop_text, lora_idx=lora_idx)
            else:
                batch = x.shape[0]
                seq_lens = audio_mask.sum(dim=1)  # Calculate the actual sequence length for each sample
                text_embed_list = []
                for i in range(batch):
                    text_embed_i = self.text_embed(
                        text[i].unsqueeze(0),
                        seq_len=seq_lens[i].item(),
                        drop_text=drop_text,
                        lora_idx=lora_idx,
                    )
                    text_embed_list.append(text_embed_i[0])
                text_embed = pad_sequence(text_embed_list, batch_first=True, padding_value=0)
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            if drop_text:
                text_embed = self.text_uncond
            else:
                text_embed = self.text_cond

        x = self.input_embed(
            x, 
            cond, 
            text_embed, 
            drop_audio_cond=drop_audio_cond, 
            audio_mask=audio_mask, 
            lora_idx=lora_idx
        )

        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b n d"],  # nosied input audio
        cond: float["b n d"],  # masked cond audio
        text: int["b nt"],  # text
        time: float["b"] | float[""],  # time step
        mask: bool["b n"] | None = None,
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
        lora_idx: torch.Tensor | None = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time, lora_idx[0] if lora_idx is not None else None)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(
                x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache, audio_mask=mask, lora_idx=lora_idx[0] if lora_idx is not None else None
            )
            x_uncond = self.get_input_embed(
                x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache, audio_mask=mask, lora_idx=lora_idx[0] if lora_idx is not None else None
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache, audio_mask=mask, lora_idx= lora_idx[0] if lora_idx is not None else None
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for i, block in enumerate(self.transformer_blocks):
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope, lora_idx=lora_idx[self.lora_map[i]] if lora_idx is not None else None)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t, lora_idx=lora_idx[0] if lora_idx is not None else None)
        output = self.proj_out(x)

        return output
