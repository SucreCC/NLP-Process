import numpy as np
import tensorflow as tf
from dataclasses import dataclass


@dataclass
class HParams:
    n_vocab: int = 0
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12


"""
    shape_list 函数的作用是 兼容静态形状和动态形状，通过结合静态信息（x.shape）和动态信息（tf.shape(x)），返回张量的每个维度大小。
        •	为什么需要：静态形状中的 None（未知值）在某些操作中会失败，而动态形状可以解决这个问题。
        •	用途：确保在形状操作（如 reshape 或动态批量处理）中代码的健壮性和兼容性，适配动态输入大小。
        •	优势：既能利用静态优化，又能正确处理动态形状，避免硬编码形状的错误。
"""


def shape_list(x):
    # 获取张量 x 的静态形状（如果可能），返回一个包含每个维度大小的列表，
    # 如果某个维度是动态的或未知，则返回 None。
    static = x.shape.as_list()

    # 获取张量 x 的动态形状，返回一个 tf.Tensor 类型的值，
    # 包含张量每个维度的大小（即使是动态的维度也会提供值）。
    dynamic = tf.shape(x)

    # 遍历静态形状中的每个维度：
    # - 如果静态形状中某个维度为 None（即未知或动态），使用动态形状中的值。
    # - 如果静态形状中有值，直接使用该静态值。
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def soft_max(x, axis=-1):
    # 通过减去 x 中的最大值，确保  e^{x - \max(x)}  的指数值不会过大, 提高数值稳定性，防止指数运算时出现溢出问题。
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


# 相比于 ReLU 和 Sigmoid，GELU 更适合处理复杂关系的任务（如 NLP 中的 Transformer 模型）。
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi * x ** 2)))


class LayerNorm(tf.Module):
    def __init__(self, scope, axis=-1, epsilon=1e-5):
        """
        Normalization Layer to normalize input to mean=0 and std=1,
        then apply diagonal affine transformation.
        """
        super().__init__(name=scope)
        self.axis = axis
        self.epsilon = epsilon
        self.g = None  # Scale parameter
        self.b = None  # Bias parameter

    def build(self, input_shape):
        n_state = input_shape[-1]
        self.g = tf.Variable(
            tf.ones([n_state]), name="g", trainable=True, dtype=tf.float32
        )
        self.b = tf.Variable(
            tf.zeros([n_state]), name="b", trainable=True, dtype=tf.float32
        )

    def __call__(self, x):
        # Build variables if not already built
        if self.g is None or self.b is None:
            self.build(x.shape)

        # Compute mean and variance along the specified axis
        u = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=self.axis, keepdims=True)

        # Normalize and apply affine transformation
        x = (x - u) * tf.math.rsqrt(s + self.epsilon)
        x = x * self.g + self.b
        return x


def split_heads(x, num_heads):
    """
    将输入张量 `x` 的最后一维拆分为多个注意力头（heads）。

    参数:
        x (tf.Tensor): 输入张量，形状为 [..., embedding_size]。
        num_heads (int): 注意力头的数量。

    返回:
        tf.Tensor: 返回拆分后的张量，形状为 [..., num_heads, head_dim]，
                   其中 head_dim = embedding_size // num_heads。
    """
    # 获取输入张量的动态形状
    input_shape = tf.shape(x)  # 动态形状，返回张量
    embedding_size = input_shape[-1]  # 获取最后一维（嵌入维度）的大小

    # 检查嵌入维度是否能被 num_heads 整除（防止非法操作）
    if isinstance(num_heads, int):
        assert embedding_size % num_heads == 0, "嵌入维度必须可以被 `num_heads` 整除。"

    # 构造新的形状，将嵌入维度拆分为 num_heads 和 head_dim 两个部分
    head_dim = embedding_size // num_heads
    new_shape = tf.concat([input_shape[:-1], [num_heads, head_dim]], axis=0)

    # 使用 tf.reshape 将张量重新调整为新的形状
    return tf.reshape(x, new_shape)


def merge_heads(x):
    """
    将最后两维合并成一个维度，用于将多头注意力的输出合并回单一维度。

    参数:
        x (tf.Tensor): 输入张量，形状为 [..., num_heads, head_dim]。

    返回:
        tf.Tensor: 合并后的张量，形状为 [..., num_heads * head_dim]。
    """
    # 获取输入张量的动态形状
    input_shape = tf.shape(x)  # 动态形状，返回张量
    num_heads, head_dim = input_shape[-2], input_shape[-1]  # 获取倒数第2维和倒数第1维的大小

    # 构造新的形状，将 num_heads 和 head_dim 合并
    new_shape = tf.concat([input_shape[:-2], [num_heads * head_dim]], axis=0)

    # 使用 tf.reshape 将张量重新调整为新的形状
    return tf.reshape(x, new_shape)


def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    """
    通过全连接模拟 1D 卷积操作。

    参数:
        x (tf.Tensor): 输入张量，形状为 [..., nx]。
        scope (str): 作用域名称。
        nf (int): 输出的特征数量。
        w_init_stdev (float): 权重初始化的标准差。

    返回:
        tf.Tensor: 输出张量，形状为 [..., nf]。
    """
    with tf.name_scope(scope):
        # 获取输入形状
        *start, nx = tf.shape(x)

        # 初始化权重和偏置
        w = tf.Variable(
            tf.random.normal([nx, nf], stddev=w_init_stdev), name="w", trainable=True
        )
        b = tf.Variable(
            tf.zeros([nf]), name="b", trainable=True
        )

        # 计算全连接的输出
        x_reshaped = tf.reshape(x, [-1, nx])  # 将输入张量展平为 2D
        c = tf.matmul(x_reshaped, w) + b  # 全连接操作
        c = tf.reshape(c, start + [nf])  # 恢复输出形状

        return c


def attention_mask(target_len, source_len, dtype=tf.float32):
    """
    创建解码器中的注意力掩码矩阵，用于防止看到未来的 Token。

    参数:
        target_len (int): 解码序列的长度（目标序列长度）。
        source_len (int): 编码序列的长度或解码器的最大序列长度（源序列长度）。
        dtype (tf.DType): 掩码矩阵的类型（默认 tf.float32）。

    返回:
        tf.Tensor: 注意力掩码矩阵，形状为 [target_len, source_len]。
    """
    # 生成 [target_len, source_len] 的二维网格矩阵
    target_indices = tf.range(target_len)[:, None]  # 目标序列的索引，形状 [target_len, 1]
    source_indices = tf.range(source_len)  # 源序列的索引，形状 [source_len]

    # 直接比较生成掩码矩阵
    mask = tf.cast(target_indices >= source_indices, dtype)
    return mask


class MultiHeadAttention(tf.Module):
    def __init__(self, d_model, num_heads, scope="multihead_attention"):
        # 初始化部分保持不变
        super().__init__(name=scope)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model 必须是 num_heads 的倍数"

        self.qkv_proj = tf.keras.layers.Dense(3 * d_model, use_bias=True, name="qkv_proj")
        self.out_proj = tf.keras.layers.Dense(d_model, use_bias=True, name="out_proj")

    def call(self, x, attention_mask=None, past_key_value=None):
        # 与之前一致的核心逻辑
        qkv = self.qkv_proj(x)
        query, key, value = tf.split(qkv, 3, axis=-1)

        # 调用外部的 split_heads 和 merge_heads
        query, key, value = map(lambda t: split_heads(t, self.num_heads), [query, key, value])

        # 剩余部分同原代码
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        present_key_value = (key, value)
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores /= tf.sqrt(tf.cast(self.head_dim, dtype=attention_scores.dtype))
        attention_scores = attention_mask(attention_scores, attention_mask)

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        context = tf.matmul(attention_weights, value)
        output = merge_heads(context)
        output = self.out_proj(output)

        return output, present_key_value


def mlp(x, scope, hidden_dim, *, hparams):
    """
    多层感知机模块，包含一个激活函数和两次线性变换。

    参数:
        x (tf.Tensor): 输入张量，形状为 [..., input_dim]。
        scope (str): 模块的作用域名称。
        hidden_dim (int): 隐藏层维度。
        hparams (HParams): 超参数对象。

    返回:
        tf.Tensor: 输出张量，形状为 [..., input_dim]。
    """
    with tf.name_scope(scope):
        input_dim = x.shape[-1]  # 输入维度
        hidden_output = gelu(conv1d(x, 'dense_hidden', hidden_dim))  # 第一次线性变换 + GELU 激活
        output = conv1d(hidden_output, 'dense_output', input_dim)  # 第二次线性变换
        return output


def transformer_block(x, scope, *, past_key_value, hparams):
    """
    Transformer 的基本构造块，包括多头注意力和 MLP。

    参数:
        x (tf.Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]。
        scope (str): 模块的作用域名称。
        past_key_value (tuple): 历史键值对缓存。
        hparams (HParams): 超参数对象。

    返回:
        tf.Tensor: 输出张量，形状为 [batch_size, seq_len, d_model]。
        present_key_value (tuple): 当前的键值对缓存。
    """
    with tf.name_scope(scope):
        d_model = hparams.n_embd  # 模型维度

        # 自注意力子层
        attention_layer = MultiHeadAttention(d_model, hparams.n_head, scope="multihead_attention")
        normed_x = LayerNorm("ln_attention")(x)
        attention_output, present_key_value = attention_layer(
            normed_x,
            attention_mask=None,  # 可选传入掩码
            past_key_value=past_key_value,
        )
        x = x + attention_output  # 残差连接

        # MLP 子层
        mlp_output = mlp(
            LayerNorm("ln_mlp")(x),
            scope="mlp",
            hidden_dim=d_model * 4,  # 隐藏层通常为输入维度的 4 倍
            hparams=hparams,
        )
        x = x + mlp_output  # 残差连接

        return x, present_key_value

def compute_past_shape(*, hparams, batch_size=None, sequence_length=None):
    """
    计算历史缓存的张量形状。

    参数:
        hparams (HParams): 超参数对象。
        batch_size (int): 批次大小。
        sequence_length (int): 序列长度。

    返回:
        list: 历史缓存的形状，形状为 [batch_size, n_layers, 2, n_heads, seq_len, head_dim]。
    """
    head_dim = hparams.n_embd // hparams.n_head
    return [
        batch_size,
        hparams.n_layer,
        2,  # 代表 [key, value]
        hparams.n_head,
        sequence_length,
        head_dim,
    ]


def expand_and_tile(value, size):
    """
    添加一个新维度并扩展指定大小。

    参数:
        value (tf.Tensor): 输入张量。
        size (int): 新维度的扩展大小。

    返回:
        tf.Tensor: 扩展后的张量。
    """
    value = tf.convert_to_tensor(value, name="value")
    expanded_value = tf.expand_dims(value, axis=0)  # 在第 0 维扩展
    tiled_value = tf.tile(expanded_value, [size] + [1] * value.shape.ndims)  # 按指定大小扩展
    return tiled_value


def generate_positions(tokens, past_length):
    """
    生成位置编码。

    参数:
        tokens (tf.Tensor): 输入序列的 token，形状为 [batch_size, seq_len]。
        past_length (int): 历史长度。

    返回:
        tf.Tensor: 位置索引，形状为 [batch_size, seq_len]。
    """
    batch_size = tf.shape(tokens)[0]  # 批次大小
    seq_len = tf.shape(tokens)[1]  # 当前序列长度
    positions = past_length + tf.range(seq_len)  # 生成当前序列的相对位置
    return expand_and_tile(positions, batch_size)  # 扩展到每个 batch


import tensorflow as tf

class GPTModel(tf.Module):
    def __init__(self, hparams, name="transformer"):
        """
        Transformer 模型类。

        参数:
            hparams (HParams): 超参数对象。
            name (str): 模型的作用域名称。
        """
        super().__init__(name=name)
        self.hparams = hparams

        # 初始化嵌入矩阵
        self.position_embeddings = tf.Variable(
            tf.random.normal([hparams.n_ctx, hparams.n_embd], stddev=0.01),
            trainable=True,
            name="position_embeddings"
        )
        self.token_embeddings = tf.Variable(
            tf.random.normal([hparams.n_vocab, hparams.n_embd], stddev=0.02),
            trainable=True,
            name="token_embeddings"
        )

        # 初始化层归一化
        self.final_layer_norm = LayerNorm("final_layer_norm")

        # 初始化 Transformer 块
        self.blocks = [
            lambda: transformer_block(scope=f"layer_{i}")
            for i in range(hparams.n_layer)
        ]

    def call(self, X, past=None):
        """
        执行前向传播。

        参数:
            X (tf.Tensor): 输入张量，形状为 [batch_size, seq_len]。
            past (tf.Tensor): 历史缓存，形状为 [batch_size, n_layer, 2, n_head, seq_len, head_dim]。

        返回:
            dict: 包含 logits 和 present 键的字典。
        """
        results = {}
        batch_size, seq_len = shape_list(X)

        # 计算位置索引
        past_length = 0 if past is None else tf.shape(past)[-2]
        positions = generate_positions(X, past_length)

        # 嵌入层
        h = tf.gather(self.token_embeddings, X) + tf.gather(self.position_embeddings, positions)

        # Transformer 层
        present_key_values = []
        past_key_values = (
            tf.unstack(past, axis=1) if past is not None else [None] * self.hparams.n_layer
        )
        assert len(past_key_values) == self.hparams.n_layer

        for layer_idx, (block_fn, past_key_value) in enumerate(zip(self.blocks, past_key_values)):
            h, present_key_value = block_fn()(h, past_key_value=past_key_value, hparams=self.hparams)
            present_key_values.append(present_key_value)

        results["present"] = tf.stack(present_key_values, axis=1)

        # 最后一层归一化
        h = self.final_layer_norm(h)

        # 语言模型 logits 计算
        h_flat = tf.reshape(h, [batch_size * seq_len, self.hparams.n_embd])
        logits = tf.matmul(h_flat, self.token_embeddings, transpose_b=True)
        logits = tf.reshape(logits, [batch_size, seq_len, self.hparams.n_vocab])
        results["logits"] = logits

        return results