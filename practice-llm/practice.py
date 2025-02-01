import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from transformer.transformer import clones, PositionWiseFeedForward


class Attention(nn.Module):
    def __init__(self, q, k, v, mask=None, dropout=None):
        super(Attention, self).__init__()
        d_k = q.size(-1)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = F.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        result = torch.matmul(attention_scores, v)
        return result, attention_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn

    def forward(self, q, k, v, mask=None):
        batch_size, = q.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, k, v = [
            l(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears(), (q, k, v))
        ]
        x, self.attn = Attention(q, k, v, mask=mask, dropout=self.dropout)
        x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)



