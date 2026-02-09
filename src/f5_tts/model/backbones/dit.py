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
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(
        self, text_num_embeds, text_dim, mask_padding=True, average_upsampling=False, conv_layers=0, conv_mult=2,
        emphasis_enhanced="mlp", emphasis_dim=None
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token
        # self.emphasis_embed = nn.Embedding(2, text_dim)  # 0: no emphasis, 1: emphasis
        # Initialize emphasis_embed with zeros
        # nn.init.zeros_(self.emphasis_embed.weight)
        
        # 增强的 emphasis 机制
        self.emphasis_enhanced = emphasis_enhanced
        if emphasis_enhanced == "mlp":
            emphasis_dim = emphasis_dim or text_dim
            # 可学习的缩放因子
            self.emphasis_scale = nn.Parameter(torch.ones(1) * 1.5)  # 初始化为1.5，可学习
            # MLP 处理 emphasis embedding（基于当前 text embedding 生成）
            self.emphasis_mlp = nn.Sequential(
                nn.Linear(text_dim, emphasis_dim),
                nn.SiLU(),
                nn.Linear(emphasis_dim, text_dim)
            )
            # 门控机制控制 emphasis 强度
            self.emphasis_gate = nn.Sequential(
                nn.Linear(text_dim, text_dim),
                nn.Sigmoid()
            )

        elif emphasis_enhanced == "transfomer":
            # 为 transformer 方案创建位置编码（如果不存在）
            # 因为 DiTBlock 需要位置信息，即使没有 ConvNeXtV2Block
            if conv_layers == 0:
                self.precompute_max_pos = 8192
                self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            
            # 创建 DiTBlock 用于 emphasis 处理
            self.emphasis_block = DiTBlock(
                    dim=text_dim,
                    heads=8,
                    dim_head=64,
                    ff_mult=4,
                    dropout=0.1,
                    qk_norm=None,
                    pe_attn_head=None,
                    attn_backend="torch",
                    attn_mask_enabled=False,
                )
            # DiTBlock 需要 time embedding，创建一个固定的 time embedding
            # 使用 TimestepEmbedding 但固定输入为 0（表示不需要 diffusion 时间步）
            self.emphasis_time_embed_module = TimestepEmbedding(text_dim)
            # 创建 RotaryEmbedding 用于 rope
            self.emphasis_rotary = RotaryEmbedding(64)  # dim_head=64
        else:
            # 原始简单版本（向后兼容）
            self.emphasis_embed = nn.Parameter(torch.zeros(text_dim))  # 全0初始化的可学习参数


        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        self.average_upsampling = average_upsampling  # zipvoice-style text late average upsampling (after text encoder)
        if average_upsampling:
            assert mask_padding, "text_embedding_average_upsampling requires text_mask_padding to be True"

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192  # 8192 is ~87.38s of 24khz audio; 4096 is ~43.69s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
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

    def forward(self, text: int["b nt"], seq_len, drop_text=False, emphasis_ids: int["b nt"] | None = None):
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0)  # (opt.) if not self.average_upsampling:
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d
        
        # Add emphasis embedding if provided
        if emphasis_ids is not None:
            # Pad emphasis_ids to match text length
            if emphasis_ids.shape[1] < seq_len:
                emphasis_ids = F.pad(emphasis_ids, (0, seq_len - emphasis_ids.shape[1]), value=0)
            elif emphasis_ids.shape[1] > seq_len:
                emphasis_ids = emphasis_ids[:, :seq_len]
            
            if self.emphasis_enhanced == "mlp":
                # 增强的 emphasis 处理
                # 使用与 text 相同的数据类型，避免类型不匹配
                emphasis_mask = emphasis_ids.unsqueeze(-1).to(dtype=text.dtype)  # (b, n, 1)
                
                # 方法1: 基于当前 text embedding 生成 emphasis embedding（更智能）
                emphasis_embed = self.emphasis_mlp(text)  # [b, n, d]
                emphasis_embed = emphasis_embed * self.emphasis_scale  # 可学习缩放
                
                # 方法2: 门控机制，根据 text 内容自适应调整 emphasis 强度
                gate = self.emphasis_gate(text)  # [b, n, d]
                emphasis_embed = emphasis_embed * gate
                
                # 应用到 emphasis 位置
                text = text + emphasis_mask * emphasis_embed
            elif self.emphasis_enhanced == "transfomer":
                # 添加位置编码（如果存在）
                text_input = text.clone()
                if hasattr(self, 'freqs_cis'):
                    text_input = text_input + self.freqs_cis[:seq_len, :]
                
                # DiTBlock 需要 time embedding 和 rope
                batch_size = text.shape[0]
                device = text.device
                dtype = text.dtype  # 确保使用与 text 相同的 dtype
                
                # 创建固定的 time embedding（输入为 0，表示不需要 diffusion 时间步）
                t_zero = torch.zeros(batch_size, device=device, dtype=dtype)
                t_embed = self.emphasis_time_embed_module(t_zero)  # [b, text_dim]
                
                # 创建 rope（Rotary Position Embedding）
                rope = self.emphasis_rotary.forward_from_seq_len(seq_len)
                
                # 先让整个 text 通过 DiTBlock 处理（捕获上下文信息）
                text_processed = self.emphasis_block(text_input, t=t_embed, mask=None, rope=rope)
                
                # 计算 DiTBlock 产生的增量（delta），避免重复叠加原始 text
                # text_processed = text_input + delta，所以 delta = text_processed - text_input
                delta = text_processed - text_input
                
                # 使用 emphasis mask：在重音位置将增量加到 text 上，其他位置保持原样
                emphasis_mask = emphasis_ids.unsqueeze(-1).to(dtype=text.dtype)  # (b, n, 1) - 使用原始 text 的 dtype
                
                # 方案：重音位置加上 DiTBlock 的增量，非重音位置保持原样
                text = text + delta * emphasis_mask
            else:
                # 原始简单版本（向后兼容）
                emphasis_mask = emphasis_ids.unsqueeze(-1)  # (b, n, 1)
                text = text + emphasis_mask * self.emphasis_embed.unsqueeze(0).unsqueeze(0)  # (b, n, d)


        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            text = text + self.freqs_cis[:seq_len, :]

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        if self.average_upsampling:
            text = self.average_upsample_text_by_mask(text, ~text_mask)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: float["b n d"],
        cond: float["b n d"],
        text_embed: float["b n d"],
        drop_audio_cond=False,
        audio_mask: bool["b n"] | None = None,
    ):
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x, mask=audio_mask) + x
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
        emphasis_enhanced="mlp",  # 是否使用增强的 emphasis 机制
        emphasis_dim=None,  # emphasis MLP 的中间维度，None 则使用 text_dim
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
            emphasis_enhanced=emphasis_enhanced,
            emphasis_dim=emphasis_dim,
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

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
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
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
        emphasis_ids: int["b nt"] | None = None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                text_embed = self.text_embed(text, x.shape[1], drop_text=drop_text, emphasis_ids=emphasis_ids)
            else:
                batch = x.shape[0]
                seq_lens = audio_mask.sum(dim=1)  # Calculate the actual sequence length for each sample
                text_embed_list = []
                for i in range(batch):
                    emphasis_ids_i = emphasis_ids[i].unsqueeze(0) if emphasis_ids is not None else None
                    text_embed_i = self.text_embed(
                        text[i].unsqueeze(0),
                        seq_len=seq_lens[i].item(),
                        drop_text=drop_text,
                        emphasis_ids=emphasis_ids_i,
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

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)

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
        emphasis_ids: int["b nt"] | None = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(
                x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache, audio_mask=mask, emphasis_ids=emphasis_ids
            )
            x_uncond = self.get_input_embed(
                x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache, audio_mask=mask, emphasis_ids=emphasis_ids
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache, audio_mask=mask, emphasis_ids=emphasis_ids
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
