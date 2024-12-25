import dataclasses

@dataclasses.dataclass
class GPTConfig:
    # 输入的tokens
    seq_len: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
