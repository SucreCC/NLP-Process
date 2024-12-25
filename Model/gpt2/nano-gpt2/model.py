import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(std + self.eps)
        return self.weight * x + self.bias


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        # 用于一次性生成q,k,v三个的权重参数
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 一个全连接层用于输出投影，输入是 3*n_embd 输出是 n_embd.   (batch_size, seq_len, 3*n_embd) -> ((batch_size, seq_len, n_embd))
        self.c_proj = nn.Linear(3 * config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention,
        self.flash = hasattr(torch.nn.functional, 'scaled_do_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer('mask', torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1, 1,
                                                                                                     config.seq_len,
                                                                                                     config.seq_len))

    # 这里在获得 attention 分数后没有做layer norm， 只是做了残差连接, 加快程序的运行速度
    def forward(self, x, mask):
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
            attention = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(k.size(-1)))
            attention = attention.masked_fill_(self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf'))
            attention = F.softmax(attention, dim=-1)
            attention = self.attn_dropout(attention)
            attention = attention @ v
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        return self.residual_dropout(attention)


# feed forward 层
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.AlphaDropout(config.dropout)

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
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ln2(self.mlp(x))
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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        # 将词嵌入矩阵的权重与语言模型头共享
        self.transformer.wte.weight = self.lm_head.weight
        # 初始化权重
        self.apply(self.init_weights)

        # 对所有参数进行初始化，如果参数名称以 'c_proj.weight' 结尾，使用特定的初始化方式
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / torch.sqrt(config.n_layers))

        # 打印模型参数数量
        print("number of parameters: %.2fM" % (self.get_num_parameters() / 1e6, ))