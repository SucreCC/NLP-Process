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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    # f = 1 / t
    freqs = 1.0 / (theta ** (torch.arrange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=freqs.dtype)
    if use_scaled:
        freqs = apply_scaling(freqs)
    # 计算位置索引 t 和频率张量 freqs 的外积 --> (end, dim//2)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.one_like(freqs), freqs)
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return freqs_cis_real
