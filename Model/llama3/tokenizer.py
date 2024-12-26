import os
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000


class Tokenizer:
    # 这是一个字典类型的类属性，保存特殊标记和其对应的编号
    special_tokens: Dict[str, int]

    # 定义保留的特殊标记数量，这里是 256
    num_reserved_special_tokens = 256

    # 这是用于标记化的正则表达式，旨在匹配不同的单词和符号模式
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path

        # 加载 tiktoken 的 BPE 合并规则，这些规则用于标记化处理
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)

        # 定义一些特殊的标记（例如文本的开始、结束等）
        special_tokens = [
            "<|begin_of_text|>",  # 文本开始标记
            "<|end_of_text|>",  # 文本结束标记
            "<|reserved_special_token_0|>",  # 保留的特殊标记（为后续保留空间）
            "<|reserved_special_token_1|>",  # 另一个保留的特殊标记
            "<|finetune_right_pad_id|>",  # 用于fine-tuning的填充标记
            "<|step_id|>",  # 步骤 ID
            "<|start_header_id|>",  # header 的开始标记
            "<|end_header_id|>",  # header 的结束标记
            "<|eom_id|>",  # 消息结束标记（end of message）
            "<|eot_id|>",  # 轮次结束标记（end of turn）
            "<|python_tag|>",  # Python 标签标记
        ]

        # 生成额外的保留特殊标记，总数为 num_reserved_special_tokens - special_tokens 中已有的数量
        reserved_tokens = [
            f"<|reserved_special_token_{2 + i}|>"
            for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]

        # 将保留的特殊标记添加到特殊标记列表中
        special_tokens = special_tokens + reserved_tokens

        # 创建一个字典，将特殊标记映射到对应的索引（从基础标记的数量开始）
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        # 使用 tiktoken 库初始化模型编码器（Encoder）
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,  # 设置编码器名称为模型路径的文件名
            pat_str=self.pat_str,  # 使用定义的正则表达式模式
            mergeable_ranks=mergeable_ranks,  # 使用加载的 BPE 合并规则
            special_tokens=self.special_tokens,  # 使用特殊标记字典
        )

        # 计算总的词汇大小（基础标记数量 + 特殊标记数量）
        self.n_words: int = num_base_tokens + len(special_tokens)

        # 为一些特殊标记设置 ID（如开始标记、结束标记等）
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]  # 开始标记 ID
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]  # 结束标记 ID
        self.eot_id: int = self.special_tokens["<|eot_id|>"]  # 轮次结束标记 ID
        self.eom_id: int = self.special_tokens["<|eom_id|>"]  # 消息结束标记 ID
        self.python_tag_id = self.special_tokens["<|python_tag|>"]  # Python 标签标记 ID
        self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]  # 填充标记 ID

        # 定义一个特殊标记列表，用于标识停止的标记
        self.stop_tokens = [
            self.special_tokens["<|begin_of_text|>"],  # 开始标记
            self.special_tokens["<|end_of_text|>"],  # 结束标记
        ]

    def encode(
            self,
            s: str,
            *,
            bos: bool,  # 是否在编码的开头插入 "begin_of_text" 标记
            eos: bool,  # 是否在编码的结尾插入 "end_of_text" 标记
            allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,  # 允许的特殊标记，默认为 None，表示没有限制
            disallowed_special: Union[Literal["all"], Collection[str]] = (),  # 禁止的特殊标记，默认为空
    ) -> List[int]:
        if allowed_special is None:
            allowed_special = set()
        assert type(s) is str

        # 将字符串 s 按最大字符长度 TIKTOKEN_MAX_ENCODE_CHARS 进行分段，并根据条件分隔
        substrs = (
            substr  # 遍历每个分段，生成每一段的 substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)  # 按照 TIKTOKEN_MAX_ENCODE_CHARS 的大小划分字符串
            for substr in self._split_whitespaces_or_nonwhitespaces(
            s[i: i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS  # 调用方法将每段分割为白空格或非空格部分
        )
        )

        # 用于存储最终编码后的 token 列表
        t: List[int] = []

        # 对每个 substr 进行编码，并将结果追加到 t 列表中
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,  # 对当前分段进行编码
                    allowed_special=allowed_special,  # 传入允许的特殊标记
                    disallowed_special=disallowed_special,  # 传入禁止的特殊标记
                )
            )

        # 如果需要在编码开始处添加开始标记（bos），则在 t 列表的开头插入 bos_id
        if bos:
            t.insert(0, self.bos_id)

        # 如果需要在编码结尾处添加结束标记（eos），则在 t 列表的结尾添加 eos_id
        if eos:
            t.append(self.eos_id)
        return t
    def decode(self, t: Sequence[int]) -> str:
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
            s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        current_slice_len = 0  # 当前连续字符的长度
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False  # 当前字符是否是空格
        slice_start = 0  # 当前分片的起始位置

        for i in range(len(s)):
            is_now_space = s[i].isspace()  # 当前字符是否是空格

            if current_slice_is_space ^ is_now_space:  # 检查空格与非空格字符是否交替
                current_slice_len = 1  # 如果交替，重置当前切片长度为 1
                current_slice_is_space = is_now_space  # 更新当前字符是否为空格
            else:
                current_slice_len += 1  # 如果当前字符与前一个字符相同类型（空格或非空格），增加当前切片长度
                if current_slice_len > max_consecutive_slice_len:  # 如果连续的空格或非空格字符超过限制
                    yield s[slice_start:i]  # 返回当前切片
                    slice_start = i  # 更新切片起始位置
                    current_slice_len = 1  # 重置当前切片长度为 1
        yield s[slice_start:]  # 返回最后一个切片