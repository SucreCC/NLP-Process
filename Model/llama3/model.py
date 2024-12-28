import os
import glob
import fire
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, TypedDict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tokenizer import Tokenizer
from config import ModelArgs


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self.norm(x.float()).type_as(x)
        return output * self.weight


'''
在代码中使用 smooth 参数对低频和高频之间进行平滑过渡，避免频率直接从高频过渡到低频时产生突变。
	•高频区域（波长较短）：保留频率不变，以准确处理短距离依赖。
	•低频区域（波长较长）：对频率进行缩放，以增强长距离编码能力。
	•中间频率区域：通过平滑插值，逐渐调整频率，避免边界上的不连续性。
'''


def apply_scaling(freqs: torch.Tensor):
    scale_factory = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192
    low_freq_wave_len = old_context_len / low_freq_factor
    high_freq_wave_len = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wave_len:
            new_freqs.append(freq)
        elif wavelen > low_freq_wave_len:
            new_freqs.append(freq / scale_factory)
        else:
            assert low_freq_wave_len != high_freq_wave_len
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append(1 - smooth) * freq / scale_factory + smooth * freq
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


'''
用于预计算旋转位置嵌入（RoPE）所需的频率张量，生成的是每个位置和嵌入维度对的正弦和余弦值。
这些值被存储为复数形式，并用于后续的旋转操作，以便模型能够编码位置信息。
'''


def precompute_freqs_cis(n_dim: int, seq_len: int, theta: float = 10000.0, use_scaled: bool = False):
    # Step 1: 计算频率基准
    # freqs: 形状 (n_dim // 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, n_dim, 2)[: (n_dim // 2)].float() / n_dim))

    # Step 2: 生成位置索引
    # t: 形状 (seq_len,)
    t = torch.arange(seq_len, device=freqs.device, dtype=freqs.dtype)

    # Step 3: 可选频率缩放
    # 如果使用频率缩放，freqs 的形状保持不变 -> (n_dim // 2,)
    if use_scaled:
        freqs = apply_scaling(freqs)

    # Step 4: 计算位置索引 t 和频率张量 freqs 的外积
    # freqs: 形状 (seq_len, n_dim // 2)
    freqs = torch.outer(t, freqs)

    # Step 5: 转换为复数表示 (极坐标形式)
    # freqs_cis: 形状保持为 (seq_len, n_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    # Step 6: 提取正弦和余弦值，并沿最后一维堆叠
    # freqs_cis_real: 形状 (seq_len, n_dim // 2, 2)
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)

    # (batch_size, n_heads, seq_len, n_dim, 2)
    return freqs_cis_real


def apply_rotary_emb(x, freqs_cis):
    # Step 1: 将输入张量 x 重新调整形状
    # 原始 x 的形状: (batch_size, seq_len, n_heads, n_dim)
    # 重塑后 xshaped 的形状: (batch_size, seq_len, n_heads, n_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

    # Step 2: 调整 freqs_cis 的形状
    # freqs_cis 的原始形状: (seq_len, n_dim // 2, 2)
    # 调整后 freqs_cis 的形状: (1, seq_len, 1, n_dim // 2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

    # Step 3: 对每对嵌入维度进行旋转操作 (应用旋转位置嵌入)
    # 使用复数的公式进行旋转计算:
    # 实部: xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1]
    # 虚部: xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1]
    # 旋转后的结果 x_out2 的形状: (batch_size, seq_len, n_heads, n_dim // 2, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    # Step 4: 将最后两维重新平铺，恢复原始嵌入维度
    # x_out2 的形状: (batch_size, seq_len, n_heads, n_dim)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    对输入张量的键值头数 (n_kv_heads) 进行重复，使其与查询头数对齐。

    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, n_kv_heads, head_dim)。
        n_rep (int): 每个键值头的重复次数。

    Returns:
        torch.Tensor: 输出张量，形状为 (batch_size, seq_len, n_kv_heads * n_rep, head_dim)。
    """
    # Step 1: 获取输入张量的形状信息
    # bs: batch_size (批量大小)
    # slen: seq_len (序列长度)
    # n_kv_heads: 键值头的数量
    # head_dim: 每个头的嵌入维度
    bs, slen, n_kv_heads, head_dim = x.shape

    # Step 2: 如果不需要重复 (n_rep=1)，直接返回原始张量
    if n_rep == 1:
        return x

    # Step 3: 对键值头维度进行重复
    return (
        x[:, :, :, None, :]  # 插入一个新维度，使形状变为 (batch_size, seq_len, n_kv_heads, 1, head_dim)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 扩展新插入的维度，变为 (batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        # 合并 n_kv_heads 和 n_rep，形状变为 (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


import torch
import torch.nn as nn


class KVCache(nn.Module):
    """
    键值缓存 (Key-Value Cache) 的实现，用于 Transformer 等模型的增量计算。
    该类管理存储键 (Key) 和值 (Value) 的缓存，并在序列生成过程中动态更新。

    Args:
        batch_size (int): 批量大小。
        seq_length (int): 缓存的最大序列长度。
        n_kv_heads (int): 键值头的数量。
        head_dim (int): 每个头的嵌入维度。
        dtype (torch.dtype): 缓存数据的类型 (如 float32)。
        device (torch.device): 缓存所在的设备 (如 GPU 或 CPU)。
    """

    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
        super().__init__()
        # Step 1: 定义缓存的形状
        # cache_shape 表示缓存张量的形状: (batch_size, seq_length, n_kv_heads, head_dim)
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)

        # Step 2: 注册缓存为模型的缓冲区
        # self.cache_k 和 self.cache_v 分别存储键 (Key) 和值 (Value)
        # register_buffer 会将张量注册为模块的持久缓冲区（不会被优化器更新）
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, start_pos, xk, xv):
        """
        更新缓存中的键和值，同时返回更新后的完整键值。

        Args:
            start_pos (int): 当前更新的起始位置 (新数据在序列中的位置)。
            xk (torch.Tensor): 新的键 (Key)，形状为 (batch_size, seq_len, n_kv_heads, head_dim)。
            xv (torch.Tensor): 新的值 (Value)，形状为 (batch_size, seq_len, n_kv_heads, head_dim)。

        Returns:
            xk (torch.Tensor): 更新后的完整键缓存，形状为 (batch_size, start_pos + seq_len, n_kv_heads, head_dim)。
            xv (torch.Tensor): 更新后的完整值缓存，形状为 (batch_size, start_pos + seq_len, n_kv_heads, head_dim)。
        """
        # Step 1: 获取当前更新段的序列长度
        seq_len = xk.size(1)  # 新增序列的长度 seq_len

        # Step 2: 更新缓存的键 (Key)
        # 在指定的起始位置范围内 (start_pos:start_pos+seq_len) 填入新的键值
        self.cache_k[:, start_pos: start_pos + seq_len, :, :] = xk

        # Step 3: 更新缓存的值 (Value)
        self.cache_v[:, start_pos: start_pos + seq_len, :, :] = xv

        # Step 4: 获取更新后的完整键缓存
        # 从缓存中截取到当前已更新的序列范围 (从 0 到 start_pos + seq_len)
        xk = self.cache_k[:, :start_pos + seq_len]

        # Step 5: 获取更新后的完整值缓存
        xv = self.cache_v[:, :start_pos + seq_len]
        return xk, xv


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.flash = args.flash
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_kv_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache = None

    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
                ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        sv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        if self.cache is not None:
            xk, xv = self.cache.update(start_pos, xk, xv)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
        if self.flash:
            output = F.scaled_dot_product_attention(xq, xk, xv, mask)
        else:
            scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        proj = self.wo(output)
        return proj


class FeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 multiple_of: int,
                 ffn_dim_multiplier: Optional[int],
                 ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.eps)

    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(x))
        return out
