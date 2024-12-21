from tkinter import Scale

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt



def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


def get_sinusoid_encoding_table(n_position, d_model):
    """
    生成位置编码表（Sinusoidal Positional Encoding）。

    参数:
        n_position (int): 序列的最大长度（位置数量）。
        d_model (int): 每个位置的编码维度（即模型的隐藏层维度）。

    返回:
        torch.FloatTensor: 一个形状为 [n_position, d_model] 的张量，
                           每行表示对应位置的编码。
    """

    def cal_angle(position, hid_idx):
        # 使用公式: angle = position / 10000^(2 * (i // 2) / d_model)
        # 这里 `hid_idx // 2` 表示偶数维度（sin）或奇数维度（cos）的分段处理
        return position / np.power(10000, 2 * (hid_idx // 2) / np.float32(d_model))

    def get_position_angle_(position):
        # 对每个维度（hid_idx）计算角度值
        return [cal_angle(position, hid_idx) for hid_idx in range(d_model)]

    # 构建正弦位置编码表
    # sinusoid_table 的形状为 [n_position, d_model]
    sinusoid_table = np.array([get_position_angle_(pos_i) for pos_i in range(n_position)])

    # 对于偶数索引的维度（0, 2, 4...），应用 sin 函数
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])

    # 对于奇数索引的维度（1, 3, 5...），应用 cos 函数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    # 转换为 PyTorch 的浮点张量
    return torch.FloatTensor(sinusoid_table)


def get_attention_pad_mask(seq_q, seq_k):
    """
    生成填充掩码，用于在注意力机制中屏蔽填充位置。

    参数:
        seq_q (torch.Tensor): Query 序列，形状为 [batch_size, len_q]。
        seq_k (torch.Tensor): Key 序列，形状为 [batch_size, len_k]。

    返回:
        torch.Tensor: 注意力掩码张量，形状为 [batch_size, len_q, len_k]。
    """
    # 获取批量大小和 Query 序列长度
    batch_size, len_q = seq_q.size()

    # 获取批量大小和 Key 序列长度
    batch_size, len_k = seq_k.size()

    # 创建填充掩码
    # seq_k.data.eq(0): 检查 Key 序列中的每个位置是否为填充值 0，返回布尔张量
    # unsqueeze(1): 在第 1 维插入一个新的维度，形状变为 [batch_size, 1, len_k]
    pad_attention_mask = seq_k.data.eq(0).unsqueeze(1)

    # 扩展掩码张量，使其适配注意力机制中需要的形状 [batch_size, len_q, len_k]
    # expand: 将形状扩展到 [batch_size, len_q, len_k]
    return pad_attention_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequent_mask(seq):
    """
    生成一个未来位置的掩码，用于屏蔽注意力机制中当前时间步之后的 token。

    参数:
        seq (torch.Tensor): 输入序列张量，形状为 [batch_size, seq_len]。

    返回:
        torch.ByteTensor: 掩码张量，形状为 [batch_size, seq_len, seq_len]。
    """
    # 定义掩码的形状: [batch_size, seq_len, seq_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    # 使用 NumPy 创建上三角矩阵，k=1 表示对角线以上的元素为 1，其他为 0
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)

    # 将 NumPy 数组转换为 PyTorch 张量，并指定为字节类型 (byte)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()

    # 返回生成的掩码张量
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attention_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2))
        # if attention_mask is not None:
        scores.masked_fill_(attention_mask, -np.inf)
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self):
        """
        初始化多头注意力模块，包括投影层和层归一化。
        """
        super(MultiHeadAttention, self).__init__()

        # 定义线性变换，用于生成 Q (Query), K (Key), V (Value)
        # d_k 是指的每一个多头的维度，  d_model = d_k * n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # 输入: d_model, 输出: n_heads * d_k
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)  # 输入: d_model, 输出: n_heads * d_k
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)  # 输入: d_model, 输出: n_heads * d_v

        # 定义线性变换，用于将多头注意力的输出映射回 d_model
        self.linear = nn.Linear(n_heads * d_v, d_model, bias=False)  # 输入: n_heads * d_v, 输出: d_model

        # 定义层归一化层，用于残差连接后的归一化
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 归一化维度为 d_model

    def forward(self, Q, K, V, attention_mask):
        """
        前向传播逻辑，计算多头注意力的输出。

        参数:
            Q (Tensor): Query 张量，形状为 [batch_size, seq_len, d_model]
            K (Tensor): Key 张量，形状为 [batch_size, seq_len, d_model]
            V (Tensor): Value 张量，形状为 [batch_size, seq_len, d_model]
            attention_mask (Tensor): 注意力掩码，形状为 [batch_size, seq_len, seq_len]

        返回:
            output (Tensor): 多头注意力输出，形状为 [batch_size, seq_len, d_model]
            attention (Tensor): 注意力权重，形状为 [batch_size, n_heads, seq_len, seq_len]
        """
        # 保存残差连接
        residual = Q  # 残差连接，形状为 [batch_size, seq_len, d_model]

        # 获取 batch 的大小
        residual, batch_size = Q, Q.size(0)

        # 生成 Query 矩阵，形状从 [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # 生成 Key 矩阵，形状从 [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # 生成 Value 矩阵，形状从 [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_v] -> [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 调整 attention_mask 的形状，使其匹配多头的形状 [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attention_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 调用缩放点积注意力 (Scaled Dot-Product Attention)
        # context: 注意力输出，形状为 [batch_size, n_heads, seq_len, d_v]
        # attention: 注意力权重，形状为 [batch_size, n_heads, seq_len, seq_len]
        context, attention = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # 将 context 转换回原来的形状 [batch_size, seq_len, n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        # 将多头注意力的输出通过线性变换映射回 d_model 维度
        output = self.linear(context)

        # 残差连接并层归一化
        output = self.layer_norm(output + residual)

        # 返回输出和注意力权重
        return output, attention


class PositionwiseFeedForward(nn.Module):
    def __init__(self):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        fc1 = F.relu(self.fc1(inputs))
        fc2 = self.fc2(fc1)
        return self.layer_norm(fc2 + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention()
        self.feed_forward = PositionwiseFeedForward()

    def forward(self, inputs, enc_self_attn_mask):
        outputs, attention = self.multihead_attention(inputs, inputs, inputs, enc_self_attn_mask)
        outputs = self.feed_forward(outputs)
        return outputs, attention


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.masked_multihead_attention = MultiHeadAttention()
        self.multihead_attention = MultiHeadAttention()
        self.feed_forward = PositionwiseFeedForward()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.masked_multihead_attention(dec_inputs, dec_inputs, dec_inputs,
                                                                     dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.multihead_attention(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.feed_forward(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 预训练embedding模型
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        positions = torch.arange(enc_inputs.size(1)).unsqueeze(0)
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(positions)
        enc_self_attn_mask = get_attention_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        positions = torch.arange(dec_inputs.size(1)).unsqueeze(0).repeat(dec_inputs.size(0), 1)
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(positions)
        dec_self_attn_pad_mask = get_attention_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask).float(), 0)

        dec_enc_attn_mask = get_attention_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # 作为 sequence to sequence 的结构， 原序列长度可以和目标序列长度不一样的。如英文的 “diner”， 在中文可以翻译文 “晚餐”。
        self.projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attn = self.encoder(enc_inputs)
        dec_outputs, dec_self_attn, dec_enc_attn = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        # 展平张量，将 [batch_size, tgt_seq_len, tgt_vocab_size] 转换为 [batch_size * tgt_seq_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attn, dec_self_attn, dec_enc_attn


def show_graph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()


if __name__ == '__main__':
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer Parameters
    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5  # length of source
    tgt_len = 5  # length of target

    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V  这里的指的是多头中每个头的维度
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    model = Transformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(3):
        optimizer.zero_grad()
        outputs, enc_self_attn, dec_self_attn, dec_enc_attn = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    print('first head of last state enc_self_attns')
    show_graph(enc_self_attn)

    print('first head of last state dec_self_attns')
    show_graph(dec_self_attn)

    print('first head of last state dec_enc_attns')
    show_graph(dec_enc_attn)