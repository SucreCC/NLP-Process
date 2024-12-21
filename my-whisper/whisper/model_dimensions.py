import dataclasses

@dataclasses.dataclass
class ModelDimensions:
    # 音频相关参数
    n_mels: int  # Mel 频谱的数量，音频特征表示的维度，通常取值为 80 或 128。
    n_audio_ctx: int  # 音频上下文窗口大小，即音频输入序列的最大长度（帧数）。
    n_audio_state: int  # 音频隐藏状态的维度，对应 Transformer 模型中的隐藏层维度。
    n_audio_head: int  # 音频注意力头的数量，多头注意力用于捕获不同的音频特征。
    n_audio_layer: int  # 音频 Transformer 的层数，表示编码器或解码器的深度。

    # 文本相关参数
    n_vocab: int  # 文本词汇表的大小，表示唯一词或子词的数量。
    n_text_ctx: int  # 文本上下文窗口大小，即文本输入序列的最大长度（Token 数量）。
    n_text_state: int  # 文本隐藏状态的维度，对应 Transformer 模型中的隐藏层维度。
    n_text_head: int  # 文本注意力头的数量，多头注意力用于捕获文本的上下文依赖关系。
    n_text_layer: int  # 文本 Transformer 的层数，表示编码器或解码器的深度。



	# 1.	音频参数
	# •	n_mels：Mel 频谱通常用于音频任务（如语音识别），是音频特征的维度。
	# •	n_audio_ctx：类似于文本中的上下文长度，决定音频输入的时间帧数。
	# •	n_audio_state：对应于 Transformer 中的隐藏层维度，影响模型的表示能力。
	# •	n_audio_head 和 n_audio_layer：用于多头注意力和深度层次结构，增强音频特征的学习能力。
	# 2.	文本参数
	# •	n_vocab：词汇表大小取决于分词方式（如 BPE、WordPiece）。
	# •	n_text_ctx：表示模型一次可以处理的最大输入 Token 数量，影响上下文捕获范围。
	# •	n_text_state：对应 Transformer 的隐藏层维度。
	# •	n_text_head 和 n_text_layer：用于多头注意力和层次结构，捕获文本的复杂语义关系。
    #
