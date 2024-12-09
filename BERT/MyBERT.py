import math
import re
from random import randrange, shuffle, random, randint
import numpy as np
import torch
from torch import nn, optim

# 4. 定义 make_batch 函数，用于生成批次数据，包括 input_ids, segment_ids, masked_tokens, masked_pos 和 isNext 标签。
def make_batch():
    batch = []  # 初始化批次列表，用于存储生成的样本
    positive_samples = negative_samples = 0  # 初始化正样本和负样本计数

    # 循环直到正样本和负样本分别达到 batch_size / 2
    while positive_samples != batch_size / 2 or negative_samples != batch_size / 2:
        # 随机选择两个句子索引
        sentence_index_a, sentence_index_b = randrange(len(sentences)), randrange(len(sentences))

        # 根据索引获取对应的句子（已转为 token 序列）
        sentence_a = token_list[sentence_index_a]
        sentence_b = token_list[sentence_index_b]

        # 构造输入序列，包含特殊标记 [CLS], [SEP]
        input_ids = [word_dict['[CLS]']] + sentence_a + [word_dict['[SEP]']] + sentence_b + [word_dict['[SEP]']]

        # 构造分段 ID（0 表示句子 A，1 表示句子 B）
        segements_ids = [0] * (1 + len(sentence_a) + 1) + [1] * (len(sentence_b) + 1)

        # 确定需要被掩码的 token 数量，取 15% 的输入序列长度，至少 1 个，最多 `max_pred`
        n_mask = min(max_pred, int(max(1, round(len(input_ids) * 0.15))))

        # 找出可以被掩码的位置（排除 [CLS] 和 [SEP]）
        can_mask_pos = [i for i, id in enumerate(input_ids) if
                        id != word_dict['[CLS]'] and id != word_dict['[SEP]']]

        # 随机打乱可掩码位置
        shuffle(can_mask_pos)

        # 初始化存储被掩码的 token ID 和位置的列表
        masked_ids, masked_pos = [], []

        # 遍历需要掩码的位置
        for pos in can_mask_pos[:n_mask]:
            masked_pos.append(pos)  # 记录被掩码的位置
            masked_ids.append(input_ids[pos])  # 记录被掩码的原始 token ID

            if random() < 0.8:  # 80% 的概率用 [MASK] 替换
                input_ids[pos] = word_dict['[MASK]']
            elif random() > 0.9:  # 10% 的概率用随机 token 替换
                input_ids[pos] = randint(0, vocab_size - 1)
            # 其余 10% 的概率保持原样（不替换）

        # 计算需要填充的长度，确保 `input_ids` 的长度与 `maxlen` 一致
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)  # 填充 0（[PAD]）
        segements_ids.extend([0] * n_pad)  # 分段 ID 也填充 0

        # 如果被掩码的 token 数少于 `max_pred`，则填充到 `max_pred` 长度
        if max_pred > n_mask:
            n_pad = max_pred - n_mask
            masked_pos.extend([0] * n_pad)  # 填充 0
            masked_ids.extend([0] * n_pad)  # 填充 0

        # 判断样本是否为正样本（句子 B 是句子 A 的后续）
        if sentence_index_a + 1 == sentence_index_b and positive_samples < batch_size / 2:
            batch.append([input_ids, segements_ids, masked_ids, masked_pos, True])  # 添加正样本
            positive_samples += 1  # 更新正样本计数

        # 否则为负样本（句子 B 不是句子 A 的后续）
        elif sentence_index_a + 1 != sentence_index_b and negative_samples < batch_size / 2:
            batch.append([input_ids, segements_ids, masked_ids, masked_pos, False])  # 添加负样本
            negative_samples += 1  # 更新负样本计数

    return batch  # 返回生成的批次

# 5. 构建 BERT 模型的结构，包括 Embedding 层、Encoder 层、多头注意力机制等。

def get_attn_pad_mask(seq_q, seq_k):
    # 获取序列长度和批量大小
    batch_size, len_q = seq_q.size()  # len_q 是 Query 的长度
    batch_size, len_k = seq_k.size()  # len_k 是 Key 的长度

    # 生成填充位置的布尔掩码
    # seq_k.data.eq(0): 检查 seq_k 中的元素是否等于 0，返回布尔矩阵 [batch_size, len_k]
    # True 表示填充位置，False 表示有效位置
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # unsqueeze(1): 在第 1 维新增一个维度，使形状变为 [batch_size, 1, len_k]
    # 新增的维度方便后续与 Query 匹配

    # 扩展掩码矩阵的形状以匹配注意力得分矩阵
    # expand: 将 [batch_size, 1, len_k] 扩展为 [batch_size, len_q, len_k]
    # len_q 是 Query 的长度，len_k 是 Key 的长度
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Embedding
# 注意嵌入矩阵和嵌入向量的概念，不可混淆
# 定义 BERT 嵌入层类，继承自 nn.Module
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()  # 调用父类的构造方法

        # 词汇嵌入矩阵
        # 这里的 vocab_size 是词汇表大小，d_model 是嵌入向量的维度
        # 嵌入矩阵维护一个词汇表，每个词通过索引映射到对应的 d_model 维度嵌入向量
        self.tok_embed = nn.Embedding(vocab_size, d_model)

        # 位置嵌入矩阵
        # 这里的 maxlen 是句子的最大长度，d_model 是嵌入向量的维度
        # 嵌入矩阵维护每个位置（如第 0 位、第 1 位等）的位置信息
        self.pos_embed = nn.Embedding(maxlen, d_model)

        # 分段嵌入矩阵
        # 这里的 n_segements 是分段类别数，通常为 2，分别表示句子 A 和句子 B
        # 嵌入矩阵用来区分每个 token 属于哪个句子段
        self.seg_embed = nn.Embedding(n_segements, d_model)

        # 层归一化
        # 对嵌入向量的每一维进行归一化，减去均值并除以标准差，提升数值稳定性
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, segment_ids):
        # input_ids: 输入 token 的索引序列，形状为 [batch_size, seq_len]
        # segment_ids: 输入段 ID 序列，形状为 [batch_size, seq_len]

        # 获取输入序列的长度 seq_len
        seq_len = input_ids.size(1)

        # 生成位置索引
        # torch.arange(seq_len) 会生成 [0, 1, ..., seq_len-1]
        pos = torch.arange(seq_len, dtype=torch.long)

        # 将位置索引扩展为与 input_ids 相同的形状
        # expand_as 会将 pos 扩展到 [batch_size, seq_len]
        pos = pos.unsqueeze(0).expand_as(input_ids)

        # 嵌入计算：逐元素相加以下三种嵌入
        # 1. tok_embed：通过 input_ids 查询词汇嵌入矩阵来获取词向量
        # 2. pos_embed：通过 pos 查询位置嵌入矩阵
        # 3. seg_embed：通过 segment_ids 查询分段嵌入矩阵
        embedding = self.tok_embed(input_ids) + self.pos_embed(pos) + self.seg_embed(segment_ids)

        # 对嵌入向量进行归一化
        # 输出形状为 [batch_size, seq_len, d_model]
        return self.norm(embedding)


# Attention score
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_pad):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_pad, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


# 多头注意力机制的实现，继承自 nn.Module
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()  # 调用父类的构造方法

        # 定义线性变换，用于生成 Query、Key 和 Value 的投影
        # 输入维度为 d_model（模型维度），输出维度为 d_k * n_heads（每个头的维度 * 头数）
        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 用于生成 Q
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 用于生成 K
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 用于生成 V

    def forward(self, Q, K, V, attn_pad):
        """
        前向传播，计算多头注意力输出。

        参数：
        Q: torch.Tensor, Query，形状为 [batch_size, seq_len, d_model]
        K: torch.Tensor, Key，形状为 [batch_size, seq_len, d_model]
        V: torch.Tensor, Value，形状为 [batch_size, seq_len, d_model]
        attn_pad: torch.Tensor, Attention Mask，用于屏蔽无效位置

        返回：
        - output: torch.Tensor, 多头注意力的输出，形状为 [batch_size, seq_len, d_model]
        - attn: torch.Tensor, 注意力权重，形状为 [batch_size, n_heads, seq_len, seq_len]
        """
        # 残差连接，保留输入的 Q（Residual Connection）
        residual, batch_size = Q, Q.size(0)

        # 使用线性变换生成 Q, K, V，并将维度调整为 [batch_size, n_heads, seq_len, d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # 扩展 Attention Mask，形状调整为 [batch_size, n_heads, seq_len, seq_len]
        attn_pad = attn_pad.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 计算 Scaled Dot-Product Attention
        # context: 注意力输出，形状为 [batch_size, n_heads, seq_len, d_k]
        # attn: 注意力权重，形状为 [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_pad)

        # 调整 context 的形状为 [batch_size, seq_len, n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        # 使用线性变换，将多头注意力输出映射回 d_model 的维度
        output = nn.Linear(n_heads * d_v, d_model)(context)

        # 残差连接并层归一化
        # output 的形状为 [batch_size, seq_len, d_model]
        return nn.LayerNorm(d_model)(output + residual), attn  # 返回输出和注意力权重


# 前馈神经网络（Position-wise Feed Forward Network），继承自 nn.Module
class PosWiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PosWiseFeedForwardNet, self).__init__()  # 调用父类的构造方法

        # 第一层线性变换
        # 输入维度为 d_model（模型维度），输出维度为 d_ff（隐藏层维度，通常比 d_model 大）
        self.fc1 = nn.Linear(d_model, d_ff)

        # 第二层线性变换
        # 输入维度为 d_ff，输出维度为 d_model，将维度还原回模型维度
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # 第一层：线性变换后，应用 GELU 激活函数
        # x 的形状从 [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_ff]
        hidden = gelu(self.fc1(x))

        # 第二层：线性变换，将维度还原为 d_model
        # hidden 的形状从 [batch_size, seq_len, d_ff] -> [batch_size, seq_len, d_model]
        output = self.fc2(hidden)

        return output  # 返回最终输出


# Transformer 编码器层，继承自 nn.Module
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()  # 调用父类的构造方法

        # 定义多头自注意力机制
        # 用于计算输入序列之间的注意力关系
        self.enc_self_attn = MultiHeadAttention()

        # 定义逐位置前馈网络
        # 用于对每个位置的特征进行独立的非线性变换
        self.pos_ffn = PosWiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_pad):
        """
        前向传播，计算编码器层的输出。

        参数:
        enc_inputs: torch.Tensor, 编码器的输入，形状为 [batch_size, seq_len, d_model]
        enc_self_attn_pad: torch.Tensor, 自注意力的掩码，形状为 [batch_size, seq_len, seq_len]

        返回:
        - enc_outputs: torch.Tensor, 编码器的输出，形状为 [batch_size, seq_len, d_model]
        - attn: torch.Tensor, 注意力权重，形状为 [batch_size, n_heads, seq_len, seq_len]
        """
        # 1. 通过多头自注意力机制计算输出和注意力权重
        # enc_inputs 作为 Query, Key, Value 输入多头注意力
        # enc_outputs: [batch_size, seq_len, d_model]
        # attn: [batch_size, n_heads, seq_len, seq_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_pad)

        # 2. 通过逐位置前馈网络对输出进行非线性变换
        # enc_outputs: [batch_size, seq_len, d_model]
        enc_outputs = self.pos_ffn(enc_outputs)

        # 3. 返回编码器的最终输出和注意力权重
        return enc_outputs, attn



# 假设以下变量与函数在其他地方已定义或导入
# n_layers: Transformer Encoder层数
# d_model: 隐藏层维度（如768）
# Embedding: 定义了BERT嵌入层，包括token embedding, segment embedding, position embedding
# EncoderLayer: 定义了BERT的单层Transformer Encoder（包括多头自注意力和前馈网络）
# get_attn_pad_mask: 根据input_ids构造用于padding位置mask的张量，mask出padding token不让其参与注意力计算
# gelu: GELU激活函数

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        # 嵌入层模块，负责将input_ids和segment_ids映射到连续向量空间，包含词嵌入、位置嵌入、类型嵌入
        self.embeeding = Embedding()

        # 使用ModuleList将多个EncoderLayer堆叠起来，构成Transformer Encoder的主体部分
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        # 以下为BERT池化和预测部分所需的线性层和激活函数
        # fc + Tanh 用于对[CLS]位置向量进行简单转换（相当于BERT中的pooler）
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()

        # 用于处理在Masked LM任务中选出的masked位置的隐藏状态的线性变换和激活
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)

        # 分类器，用于NSP（Next Sentence Prediction）任务，对[CLS]向量进行二分类预测
        self.classfier = nn.Linear(d_model, 2)

        # 将decoder的权重与词嵌入的权重共享，以适配Masked LM的预测头。
        embed_weight = self.embeeding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()

        # 解码器（用于预测被mask掉的单词），权重与词嵌入共享
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        # 1. 得到输入序列的嵌入表示 x: [batch_size, seq_len, d_model]
        x = self.embeeding(input_ids, segment_ids)

        # 2. 构造自注意力用的padding mask，确保注意力时不会关注到padding位置
        enc_self_attn_pad = get_attn_pad_mask(input_ids, input_ids)

        # 3. 将输入x依次通过多个EncoderLayer，完成Transformer Encoder的堆叠
        # 每一层会执行多头自注意力和前馈网络的变换，并返回更新后的隐藏状态x和注意力图 enc_self_attn
        for layer in self.layers:
            x, enc_self_attn = layer(x, enc_self_attn_pad)

        # 4. 池化操作：取序列中的[CLS]位置向量 x[:, 0] 并通过全连接层和Tanh激活
        # h_pooled: [batch_size, d_model], 用于下游分类任务（如NSP）
        h_pooled = self.activ1(self.fc(x[:, 0]))

        # 5. NSP（Next Sentence Prediction）的分类器，对[CLS]位置向量进行二分类
        # logits_clsf: [batch_size, 2]
        logits_clsf = self.classfier(h_pooled)

        # 6. 针对Masked LM任务，从隐藏状态中取出masked位置对应的向量进行预测
        # masked_pos: [batch_size, masked_len] -> 扩展后: [batch_size, masked_len, d_model]
        masked_pos = masked_pos[:, :, None].expand(-1, -1, x.size(-1))

        # 从x中获取对应masked位置的隐藏状态向量 h_masked: [batch_size, masked_len, d_model]
        h_masked = torch.gather(x, 1, masked_pos)

        # 对这部分hidden states再通过一层linear和gelu激活，然后LayerNorm归一化
        h_masked = self.norm(self.activ2(self.linear(h_masked)))

        # 7. 通过共享权重的decoder层和decoder_bias预测原token id
        # logits_lm: [batch_size, masked_len, vocab_size]
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        # 返回两个输出：logits_lm用于Masked LM任务，logits_clsf用于NSP任务
        return logits_lm, logits_clsf


# 1. 定义 BERT 模型的参数，包括最大句子长度、批次大小、最大预测 token 数等。
if __name__ == '__main__':
    maxlen = 30  # 句子最大长度
    batch_size = 6  # 每一个batch有多少个句子进行训练
    max_pred = 5  # 输入的一个句子中最多可以有多少个被mask的词
    n_layers = 6  # 有多少个编码器层
    n_heads = 12  # transformer 的多头数目
    d_model = 768  # Embedding size
    d_ff = 3072  # FeedForward 层的维度
    d_k = d_v = 64  # K V 的维度
    n_segements = 2  # Next sentence predict 任务

# 2. 预处理文本，将文本去除标点后分割成句子列表，创建词表和词典映射。

# 原始文本
text = (
    "Hi, what are you doing? I'm planning my weekend.\n"
    "Oh, sounds fun! I might join you if that’s okay.\n"
    "Of course! We could also invite Sarah. She loves hiking.\n"
    "Perfect. I’ll bring snacks and drinks for everyone.\n"
    "Great! We’ll start early morning to avoid the heat.\n"
    "Sure, I'll text Sarah and let her know.\n"
    "Thanks for organizing this. It’s going to be awesome.\n"
    "No problem at all. Let’s finalize the time tonight.\n"
    "Cool! Should we also plan something for the evening?\n"
    "That’s a great idea! Maybe a movie or dinner?\n"
    "Let’s do both. I know a good Italian place.\n"
    "Sounds amazing. I’ll leave the restaurant booking to you.\n"
    "Deal! I’ll make sure everything is ready for the weekend.\n"
)

# 使用正则表达式去除标点符号（如.,!?-），并将文本转为小写，然后按行分割成句子列表
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')

# 使用空格分割所有句子中的单词，并创建一个唯一单词的集合
word_list = list(set(" ".join(sentences).split()))  # `set` 去重，`list` 转为列表

# 初始化词典，加入特殊标记
word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

# 遍历单词列表，依次为每个单词分配一个唯一的索引，并更新到词典中
for i, word in enumerate(word_list):
    word_dict[word] = i + 4  # 特殊标记占据前4个索引，单词索引从4开始

# 创建一个反向映射字典，将索引映射回单词
number_dict = {i: word for i, word in enumerate(word_list)}

# 计算词汇表的大小
vocab_size = len(word_dict)


# 3. 将文本转化为数字化 token 列表，为模型输入做准备。
token_list = []
for i, sentence in enumerate(sentences):
    ids = [word_dict[word] for word in sentence.split()]
    token_list.append(ids)

batch = make_batch()

# 将batch中的各个元素解包出来，然后转换成张量
input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))

# 6. 定义损失函数 (CrossEntropyLoss) 和优化器 (Adam)。
model = BERT()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 7. 通过循环训练模型：
#    - 清零优化器梯度。
#    - 将批次数据送入 BERT 模型，获取输出 logits。
#    - 计算语言模型任务 (MLM) 和句子分类任务 (NSP) 的损失。
#    - 合并损失并反向传播，更新模型参数。
#    - 每隔固定 epoch 打印损失值。

for epoch in range(100):
    optimizer.zero_grad()
    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
    loss_lm = (loss_lm.float()).mean()
    loss_clsf = criterion(logits_clsf, isNext)
    loss = loss_lm + loss_clsf
    if (epoch + 1) % 10 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()
