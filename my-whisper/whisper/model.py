from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor

from model_dimensions import ModelDimensions


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length: int, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class AudioEncoder(nn.Module):
    def __init__(self,
                 n_mels: int, n_audio_ctx: int, n_audio_state: int, n_audio_head: int, n_audio_layer: int
                 ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_audio_state, n_audio_head, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_audio_ctx, n_audio_state))


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
