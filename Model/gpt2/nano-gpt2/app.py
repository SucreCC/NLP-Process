import tiktoken
import torch

from model import GPT
from config import GPTConfig



num_samples = 10 # number of samples to draw
model_name = 'gpt2'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
config = GPTConfig()
model = GPT.from_pretrained(model_name, dict(dropout=0.0))
model.eval()
model.to(config.device)
enc = tiktoken.get_encoding(model_name)
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


max_new_tokens = 100
temperature = 0.8
top_k = 200

start_ids = encode("讲一个中文笑话")
x = (torch.tensor(start_ids, dtype=torch.long, device=config.device)[None, ...])


with torch.no_grad():
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    print(decode(y[0].tolist()))
    print("\n")
    print('---------------')