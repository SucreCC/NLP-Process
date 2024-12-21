from dataclasses import dataclass


@dataclass()
class GPT2Config():
    n_vacab: int
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int
