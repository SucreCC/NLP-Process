import torch
import torch.nn as nn
import math
import copy
from torch.nn.parameter import Parameter


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()

        # 增加这些参数是为了让模型学习到更合适的均值和方差
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(std + self.variance_epsilon)
        return self.weight * x + self.bias


# 全连接层
class Conv1d(nn.Module):
    def __init__(self, nf, nx):
        """
        初始化 Conv1d 模块。

        参数:
        - nf (int): 输出特征数量（输出维度）。
        - nx (int): 输入特征数量（输入维度）。
        """
        super(Conv1d, self).__init__()
        self.nf = nf  # 保存输出特征数量

        # 初始化权重矩阵，形状为 (nx, nf)
        # 使用正态分布（均值为 0，标准差为 0.02）初始化权重
        w = torch.empty(nx, nf)  # 创建空张量
        nn.init.normal_(w, std=0.02)  # 正态分布初始化
        self.weight = nn.Parameter(w)  # 将权重定义为可训练参数

        # 初始化偏置向量，形状为 (nf,)
        self.bias = nn.Parameter(torch.zeros(nf))  # 偏置初始化为 0

    def forward(self, x):
        """
        前向传播过程。

        参数:
        - x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, nx)。

        返回:
        - torch.Tensor: 输出张量，形状为 (batch_size, seq_length, nf)。
        """
        # 计算输出的目标形状，保持前面维度不变，最后一维替换为 nf
        size_out = x.size()[:-1] + (self.nf,)
        # 示例: 如果 x.shape == (batch_size, seq_length, nx)，
        # 则 size_out == (batch_size, seq_length, nf)

        # 将输入张量展平为二维形状 (total_elements, nx)
        # `total_elements` 是 batch_size 和 seq_length 的乘积
        x = x.view(-1, x.size(-1))  # 形状变为 (total_elements, nx)

        # 使用 torch.addmm 进行矩阵乘法和加偏置操作
        # 计算公式: x @ weight + bias
        # - x: (total_elements, nx)
        # - weight: (nx, nf)
        # - bias: (nf,)
        x = torch.addmm(self.bias, x, self.weight)  # 输出形状 (total_elements, nf)

        # 恢复张量形状为原始输入的 batch 和序列维度
        x = x.view(*size_out)  # 恢复形状为 (batch_size, seq_length, nf)

        return x  # 返回输出张量


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()

        n_state = nx
        assert n_ctx % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_state = n_state
        self.scale = scale
        self.c_attn = Conv1d(n_state * 3, nx)
        self.c_proj = Conv1d(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(self.n_head)
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (x.size(-2) * x.size(-1),)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = self.split_heads(x, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-1)
        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1d(n_state, nx)
        self.proj = Conv1d(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.n_vocab

        self.wte = nn.Embedding(config.n_vocab, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_embedding_weight(self, model_embedding_weights):
        embed_shape = model_embedding_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embedding_weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past=layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents


class GPT2LMHead(nn.Module):
    def __init__(self, model_embedding_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embedding_weights)

    def set_embeddings_weights(self, model_embedding_weights):
        embed_shape = model_embedding_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embedding_weights

    def forward(self, hidden_states):
        lm_logits = self.decoder(hidden_states)
        return lm_logits


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)

    def set_tied(self):
        self.lm_head.set_embeddins_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past=past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.sieze(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents
