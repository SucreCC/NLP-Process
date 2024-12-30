import math
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from config import ModelArgs


class RMSNorm(torch.nn.Module):
    '''
    与layerNorm 相比， RMSNorm 只进行缩放（归一化），不进行中心化操作，减少计算成本。
    '''

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor):
    """
    逻辑:
        - 通过指定的上下界 (low_freq_wavelen 和 high_freq_wavelen) 对不同波长的频率应用不同的缩放规则。
        - 缩放分为三种情况：
            1. 高频（波长小于 high_freq_wavelen）：保留原始频率。
            2. 低频（波长大于 low_freq_wavelen）：频率被按 scale_factor 缩放。
            3. 中间频率：通过线性插值的方式进行平滑过渡。
    """
    # RoPE 缩放的缩放因子和上下界参数
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # 原始 Llama3 模型的上下文长度

    # 根据上下界因子计算对应的波长范围
    low_freq_wavelen = old_context_len / low_freq_factor  # 低频波长
    high_freq_wavelen = old_context_len / high_freq_factor  # 高频波长

    # 存储缩放后的频率
    new_freqs = []

    # 对输入的每个频率依次进行处理
    for freq in freqs:
        wavelen = 2 * math.pi / freq  # 计算频率对应的波长

        # 如果波长小于高频波长，属于高频部分，保留原始频率
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        # 如果波长大于低频波长，属于低频部分，按缩放因子缩放
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            # 中间频率部分，通过线性插值进行平滑过渡
            assert low_freq_wavelen != high_freq_wavelen  # 确保上下界波长不同
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    """
    预计算频率张量 (freqs) 和复数旋转向量 (freqs_cis)。

    参数:
        dim (int): 输入张量的维度，即特征维度。
        end (int): 时间步的终点，用于生成时间序列范围 [0, end)。
        theta (float): 控制频率范围的比例因子，默认为 10000.0。
        use_scaled (bool): 是否对频率张量进行缩放，默认为 False。

    逻辑:
        1. 生成频率张量 `freqs`，形状为 [dim // 2]。
        2. 根据 `end` 生成时间步张量 `t`，形状为 [end]。
        3. 计算时间步与频率的外积 `freqs` 和 `t`，结果形状为 [end, dim // 2]。
        4. 使用极坐标形式生成复数旋转张量 `freqs_cis`，形状为 [end, dim // 2]。
        5. 提取复数旋转张量的实部和虚部，堆叠为形状 [end, dim, 2]。
    """
    # Step 1: 生成频率张量 freqs，形状为 [dim // 2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # Step 2: 生成时间步张量 t，形状为 [end]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    # 如果启用了频率缩放，调用 apply_scaling 对频率张量 freqs 进行调整
    if use_scaled:
        freqs = apply_scaling(freqs)  # 调用频率缩放函数，freqs 形状保持 [dim // 2]

    # Step 3: 计算外积 freqs 和 t，结果形状为 [end, dim // 2]
    freqs = torch.outer(t, freqs)

    # Step 4: 使用极坐标形式生成复数旋转向量 freqs_cis，形状为 [end, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 生成 complex64 类型张量

    # Step 5: 提取实部和虚部，并沿最后一维堆叠，结果形状为 [end, dim, 2]
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return freqs_cis_real


def apply_rotary_emb(x, freqs_cis):
    """
    应用旋转位置嵌入 (Rotary Position Embedding, RoPE) 到输入张量 x。
    逻辑:
        1. 将输入张量 x 转换为实部和虚部分离的形式，形状为 [batch_size, seq_len, n_heads, head_dim/2, 2]。
        2. 将频率复数旋转张量 freqs_cis 进行广播扩展，适配 x 的形状。
        3. 计算旋转位置嵌入的加权结果，通过复数乘法公式完成。
        4. 将结果转换回原始形状 [batch_size, seq_len, n_heads, head_dim]。
    """
    # Step 1: 将 x 转换为实部和虚部分离的形式
    # xshaped 的形状为 [batch_size, seq_len, n_heads, head_dim/2, 2]
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

    # Step 2: 调整 freqs_cis 的形状以适配 xshaped
    # freqs_cis 的形状从 [seq_len, head_dim/2, 2] -> [1, seq_len, 1, head_dim/2, 2]
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

    # Step 3: 计算旋转嵌入，通过复数乘法公式：
    # real_part = real(x) * real(freqs_cis) - imag(x) * imag(freqs_cis)
    # imag_part = imag(x) * real(freqs_cis) + real(x) * imag(freqs_cis)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],  # 实部计算
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],  # 虚部计算
        ],
        -1,
    )

    # Step 4: 将结果还原为原始形状
    # x_out2 的形状从 [batch_size, seq_len, n_heads, head_dim/2, 2] -> [batch_size, seq_len, n_heads, head_dim]
    x_out2 = x_out2.flatten(3)

    return x_out2.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    在多头注意力中重复 Key 和 Value 张量的头数，用于适配多头注意力机制。

    逻辑:
        1. 如果 `n_rep == 1`，直接返回输入张量，不做任何处理。
        2. 否则，将输入张量的 `n_kv_heads` 维度扩展为 `[n_kv_heads, n_rep]`，通过 `expand` 增加维度。
        3. 再将扩展后的张量重塑为目标形状 `[batch_size, seq_len, n_kv_heads * n_rep, head_dim]`。
    """
    # 获取输入张量的形状信息
    bs, slen, n_kv_heads, head_dim = x.shape  # [batch_size, seq_len, n_kv_heads, head_dim]

    # 如果重复次数为 1，直接返回原始张量
    if n_rep == 1:
        return x

    # 重复并调整张量的形状
    return (
        x[:, :, :, None, :]  # 插入一个新维度，形状变为 [batch_size, seq_len, n_kv_heads, 1, head_dim]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 广播新维度，形状变为 [batch_size, seq_len, n_kv_heads, n_rep, head_dim]
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        # 合并重复的头，形状变为 [batch_size, seq_len, n_kv_heads * n_rep, head_dim]
    )


class KVCache(nn.Module):
    """
    用于存储和更新注意力机制中的 Key 和 Value 缓存，优化 Transformer 等模型的推理效率。
    """

    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
        super().__init__()
        # 定义 Key 和 Value 的缓存张量，初始化为零
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, start_pos, xk, xv):
        """
        更新缓存中的 Key 和 Value，并返回当前有效的 Key 和 Value 张量。
        """
        # 获取本次更新的序列长度
        seqlen = xk.size(1)

        # 更新 Key 的缓存，从 start_pos 开始写入新 Key
        self.cache_k[:, start_pos: start_pos + seqlen] = xk

        # 更新 Value 的缓存，从 start_pos 开始写入新 Value
        self.cache_v[:, start_pos: start_pos + seqlen] = xv

        # 提取缓存中所有有效的 Key 和 Value
        xk = self.cache_k[:, : start_pos + seqlen]
        xv = self.cache_v[:, : start_pos + seqlen]

        return xk, xv


class Attention(nn.Module):
    """
    多头注意力模块，支持闪存注意力和键值缓存（KVCache）。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.flash = args.flash
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # 线性层定义
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache = None  # KV 缓存

    def forward(self, x, start_pos, freqs_cis, mask):
        """
        执行多头注意力的前向传播。
        """
        # 计算 Query、Key 和 Value
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入 (RoPE)
        xq, xk = apply_rotary_emb(xq, freqs_cis), apply_rotary_emb(xk, freqs_cis)

        # 更新并获取缓存中的 Key 和 Value
        if self.cache is not None:
            xk, xv = self.cache.update(start_pos, xk, xv)

        # 如果头数量不匹配，重复 Key 和 Value
        xk, xv = repeat_kv(xk, self.n_rep), repeat_kv(xv, self.n_rep)

        # 转置以适配注意力计算
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))

        # 注意力计算（闪存注意力或普通注意力）
        if self.flash:
            output = F.scaled_dot_product_attention(xq, xk, xv, mask)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores += mask
            scores = F.softmax(scores, dim=-1)
            output = torch.matmul(scores, xv)

        # 合并多头输出并投影到输出维度
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    前馈神经网络模块，用于 Transformer 模型中的非线性特征变换。
    """

    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        # 调整隐藏层维度
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 定义线性变换层
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
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """
    方法:
        forward_inference: 推理模式下的前向传播，支持序列生成。
        forward_loss: 训练模式下的前向传播，计算交叉熵损失。
    """

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # Transformer 层
        self.layers = nn.ModuleList(
            TransformerBlock(params) for _ in range(params.n_layers)
        )

        # 归一化和输出层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # 预计算的旋转嵌入频率 (RoPE)
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        """
        推理模式下的前向传播，用于生成序列。
        逻辑:
            1. 计算输入 tokens 的嵌入表示。
            2. 应用旋转嵌入频率。
            3. 生成注意力掩码，确保因果性。
            4. 逐层通过 TransformerBlock 处理。
            5. 应用归一化和输出层，返回 logits。
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        # 构造因果注意力掩码
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        # 逐层 TransformerBlock
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        # 应用归一化和输出投影
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def forward_loss(self, inputs: torch.Tensor, targets: torch.Tensor, ignore_index=-100):
        """
        训练模式下的前向传播，计算交叉熵损失。

        逻辑:
            1. 计算输入 tokens 的嵌入表示。
            2. 应用旋转嵌入频率。
            3. 生成因果注意力掩码。
            4. 逐层通过 TransformerBlock 处理。
            5. 应用归一化和输出层，计算 logits。
            6. 计算 logits 和目标序列之间的交叉熵损失。
        """
        _bsz, seqlen = inputs.shape
        h = self.tok_embeddings(inputs)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        # 构造因果注意力掩码
        mask = torch.full((seqlen, seqlen), float("-inf"), device=inputs.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        # 禁用 KV 缓存逻辑
        start_pos = -1
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        # 应用归一化和输出投影
        h = self.norm(h)
        logits = self.output(h).float()

        # 计算交叉熵损失
        loss = F.cross_entropy(
            input=logits.transpose(1, 2),
            target=targets,
            reduction="mean",
            ignore_index=ignore_index,
        )
        return loss

    def configure_optimizers(self, learning_rate, weight_decay=0.0, betas=(0.9, 0.97), device_type='cuda'):
        """
        配置优化器，用于模型训练。

        参数:
            learning_rate (float): 学习率。
            weight_decay (float): 权重衰减系数，默认值为 0.0。
            betas (Tuple[float, float]): AdamW 优化器的动量参数，默认值为 (0.9, 0.97)。
            device_type (str): 设备类型，默认为 'cuda'。

        返回:
            torch.optim.AdamW: 配置好的 AdamW 优化器。

        逻辑:
            1. 根据 `finetune_type` 决定哪些参数参与训练。
               - `"rmsnorm"`: 只训练 RMSNorm 层的参数。
               - `"all"`: 训练所有模型参数。
               - `"all_no_pos"`: 训练除位置嵌入和输出层以外的所有参数。
            2. 统计模型总参数和可训练参数数量。
            3. 使用 AdamW 优化器（如果设备是 CUDA，优先使用 fused 版本以提高性能）。
        """
        train_params = []

        # 定义微调类型，控制可训练参数的选择
        finetune_type = "all"  # 可选值: "rmsnorm", "all", "all_no_pos"

        if finetune_type == "rmsnorm":
            # 仅训练 RMSNorm 层的参数
            for name, param in self.named_parameters():
                if "norm" in name:
                    train_params.append(param)
        elif finetune_type == "all":
            # 训练所有参数
            for param in self.parameters():
                train_params.append(param)
        elif finetune_type == "all_no_pos":
            # 训练除位置嵌入和输出层以外的参数
            n, m = 0, 0
            for name, param in self.named_parameters():
                if name == "output.weight":
                    n += 1  # 跳过输出层参数
                    continue
                elif name == "tok_embeddings.weight":
                    m += 1  # 跳过位置嵌入参数，并禁用其梯度计算
                    param.requires_grad = False
                else:
                    train_params.append(param)
            assert n == 1, "did not find output.weight"
            assert m == 1, "did not find tok_embeddings.weight"

        # 打印模型参数数量和可训练参数数量
        print("number of parameters: ", sum(p.numel() for p in self.parameters()))
        print("number of trainable parameters: ", sum(p.numel() for p in train_params))

        # 判断是否可以使用 fused AdamW 优化器
        fused_available = True  # fused AdamW 的检查代码可根据具体版本修改
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        # 创建 AdamW 优化器
        optimizer = torch.optim.AdamW(
            train_params,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )
        return optimizer