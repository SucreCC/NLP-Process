import inspect
import math
from audioop import bias

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        # 用于一次性生成q,k,v三个的权重参数
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 一个全连接层用于输出投影，输入是 n_embd 输出是 n_embd.   (batch_size, seq_len, n_embd) -> ((batch_size, seq_len, n_embd))
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention,
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer('bias', torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1, 1,
                                                                                                     config.seq_len,
                                                                                                     config.seq_len))

    # 这里在获得 attention 分数后没有做layer norm， 只是做了残差连接, 加快程序的运行速度
    def forward(self, x):
        batch_size, seq_len, n_embd = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.n_embd // self.n_heads).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.n_embd // self.n_heads).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.n_embd // self.n_heads).permute(0, 2, 1, 3)

        if self.flash:
            attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                         dropout_p=self.dropout if self.training else 0,
                                                                         is_causal=True)
        else:
            attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attention = attention.masked_fill_(self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf'))
            attention = F.softmax(attention, dim=-1)
            attention = self.attn_dropout(attention)
            attention = attention @ v
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        return self.residual_dropout(self.c_proj(attention))


# feed forward 层
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None  # 确保词汇表大小存在
        assert config.seq_len is not None  # 确保序列长度存在
        self.config = config  # 保存配置

        # 定义transformer模块
        self.transformer = nn.ModuleDict(dict(
            # 词嵌入矩阵，大小为 vocab_size x n_embd
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # 位置嵌入矩阵，大小为 seq_len x n_embd
            wpe=nn.Embedding(config.seq_len, config.n_embd),
            # dropout层
            drop=nn.Dropout(config.dropout),
            # 多层 Transformer 块
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            # 最后的 LayerNorm 层
            ln_f=LayerNorm(config.n_embd, bias=config.bias)
        ))

        # 语言模型头，输入为n_embd, 输出大小为 vocab_size
        # 去掉 bias 是为了简化模型、提高训练效率，并且通常不会对模型性能产生负面影响。
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 将词嵌入矩阵的权重与语言模型头共享
        self.transformer.wte.weight = self.lm_head.weight
        # 初始化权重
        self.apply(self._init_weights)

        # 对所有参数进行初始化，如果参数名称以 'c_proj.weight' 结尾，使用特定的初始化方式
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(config.n_layers))

        # 打印模型参数数量
        print("number of parameters: %.2fM" % (self.get_num_parameters() / 1e6,))

    def get_num_parameters(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None):
        device = ids.device  # 获取输入 `ids` 张量所在的设备（如 GPU 或 CPU）
        batch_size, seq_len = ids.size()  # 获取输入数据的批量大小和序列长度

        # 确保输入序列的长度不超过配置中定义的最大序列长度
        assert seq_len <= self.config.seq_len, f"Cannot forward sequence of length {seq_len}, sequence length is only {self.config.seq_len}"

        # 生成位置索引张量：[0, 1, 2, ..., seq_len-1]，并且它的类型为 long 类型
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # 获取词嵌入（词向量），通过词嵌入层 `wte` 将 ids 转换为对应的词向量
        tok_emb = self.transformer.wte(ids)

        # 获取位置嵌入（位置编码），通过位置嵌入层 `wpe` 将位置索引 `pos` 转换为对应的位置信息
        pos_emb = self.transformer.wpe(pos)

        # 将词嵌入和位置嵌入加和，并通过 dropout 层进行正则化
        x = self.transformer.drop(tok_emb + pos_emb)

        # 依次通过每个 Transformer Block 进行处理
        for block in self.transformer.h:
            x = block(x)  # 每一层 Transformer block 处理后的结果作为输入传递到下一个 block

        # 最后通过 LayerNorm 层进行归一化，通常是为了减少训练中的偏差
        x = self.transformer.ln_f(x)

        if targets is not None:  # 如果提供了目标（用于训练）
            # 使用语言模型头（线性层）将 Transformer 输出映射到词汇表大小
            # (batch_size * seq_len, n_embd) --> (batch_size * seq_len, vocab_size)
            logits = self.lm_head(x)

            # 计算交叉熵损失，logits 的形状为 (batch_size * seq_len, vocab_size)，目标 `targets` 为 (batch_size * seq_len)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:  # 如果没有提供目标（用于推理/生成模式）
            # 只取序列的最后一个 token 的输出，生成下一个词的概率分布
            logits = self.lm_head(x[:, [-1], :])
            loss = None  # 没有计算损失

        return logits, loss  # 返回 logits（预测结果）和损失（训练时）或 None（推理时）

    def corp_block_size(self, seq_len):
        assert seq_len <= self.config.seq_len
        self.config.seq_len = seq_len
        self.transformer.wpe.weight = self.nn.Parameter(self.transformer.wpe.weight[:seq_len])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :seq_len, :seq_len]

    # 加载一个预训练模型然后把里面的权重参数转移到，自己的模型中。
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2_medium', 'gpt2_large', 'gpt2_xl'}
        overrides = override_args if override_args is not None else {}
        assert all(k == 'dropout' for k in overrides)
        from transformers import GPT2LMHeadModel
        from config import GPTConfig
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2': dict(n_layers=12, n_heads=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layers=24, n_heads=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layers=36, n_heads=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layers=48, n_heads=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True

        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            a = sd_hf[k]
            b = sd[k]
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.item() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW:{use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_heads, cfg.n_embd // cfg.n_heads, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token + T
        flops_per_iter = flops_per_fwdbwd + fwdbwd_per_iter
        flops_achived = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achived / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, - self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
