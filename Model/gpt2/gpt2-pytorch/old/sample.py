import tensorflow as tf
from model import GPTModel


def top_k_logits(logits, k):
    """
    对 logits 应用 Top-K 筛选，保留最高的 K 个值。

    参数:
        logits (tf.Tensor): 输入的 logits，形状为 [batch_size, vocab_size]。
        k (int): 保留的最高概率的 K 个值。

    返回:
        tf.Tensor: 经过 Top-K 筛选后的 logits。
    """
    if k == 0:
        return logits  # 不进行筛选，直接返回原始 logits

    values, _ = tf.math.top_k(logits, k=k)
    min_values = values[:, -1, tf.newaxis]  # 第 K 大的值
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,  # 将低于阈值的 logits 设置为一个很小的值
        logits,
    )


def top_p_logits(logits, p):
    """
    对 logits 应用核采样（Nucleus Sampling），保留累计概率小于等于 P 的值。

    参数:
        logits (tf.Tensor): 输入的 logits，形状为 [batch_size, vocab_size]。
        p (float): 累计概率的阈值，0 < p <= 1。

    返回:
        tf.Tensor: 经过核采样筛选后的 logits。
    """
    sorted_logits = tf.sort(logits, direction="DESCENDING", axis=-1)
    cumulative_probs = tf.math.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

    # 找到满足累计概率小于等于 p 的位置
    mask = cumulative_probs > p
    min_values = tf.reduce_min(
        tf.where(mask, tf.fill(tf.shape(sorted_logits), tf.float32.max), sorted_logits),
        axis=-1,
        keepdims=True,
    )

    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,  # 将低于阈值的 logits 设置为一个很小的值
        logits,
    )


def sample_sequence(
    *,
    model: GPTModel,
    hparams,
    length,
    start_token=None,
    batch_size=None,
    context=None,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
):
    """
    基于 GPTModel 模型的序列采样函数。

    参数:
        model (GPTModel): GPT 模型实例。
        hparams (HParams): 超参数对象。
        length (int): 生成序列的长度。
        start_token (int): 起始 token 的索引。
        batch_size (int): 批次大小。
        context (tf.Tensor): 初始上下文序列，形状为 [batch_size, seq_len]。
        temperature (float): 控制分布的平滑程度，1.0 表示不调整。
        top_k (int): Top-K 筛选参数，0 表示不使用。
        top_p (float): 核采样参数，1.0 表示不使用。

    返回:
        tf.Tensor: 生成的 token 序列，形状为 [batch_size, length]。
    """
    if start_token is None:
        assert context is not None, "必须指定 context 或 start_token！"
    else:
        assert context is None, "不能同时指定 context 和 start_token！"
        context = tf.fill([batch_size, 1], start_token)

    def step(tokens, past_key_value=None):
        outputs = model(tokens, past=past_key_value)
        logits = outputs["logits"][:, :, : hparams.n_vocab]
        presents = outputs["present"]
        return {"logits": logits, "presents": presents}

    with tf.name_scope("sample_sequence"):
        def body(past_key_value, prev_tokens, generated_tokens):
            next_outputs = step(prev_tokens, past_key_value=past_key_value)
            logits = next_outputs["logits"][:, -1, :] / tf.cast(temperature, tf.float32)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
            return (
                tf.concat([past_key_value, next_outputs["presents"]], axis=-2)
                if past_key_value is not None
                else next_outputs["presents"],
                samples,
                tf.concat([generated_tokens, samples], axis=1),
            )

        # 初始化变量
        past_key_value, prev_tokens, output_tokens = None, context, context

        # 循环采样
        for _ in range(length - 1):
            past_key_value, prev_tokens, output_tokens = body(
                past_key_value, prev_tokens, output_tokens
            )

        return output_tokens