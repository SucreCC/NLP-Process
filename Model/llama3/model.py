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
