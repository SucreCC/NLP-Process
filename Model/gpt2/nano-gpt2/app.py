from contextlib import nullcontext

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
device ='cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


max_new_tokens = 50
temperature = 0.8
top_k = 200

start_ids = encode("讲一个中文笑话")
x = (torch.tensor(start_ids, dtype=torch.long, device=config.device)[None, ...])


with torch.no_grad():
    with ctx:
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(decode(y[0].tolist()))
        print('---------------')