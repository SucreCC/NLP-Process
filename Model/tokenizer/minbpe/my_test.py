from basic import BasicTokenizer
from my_regex import RegexTokenizer
import pickle




def load_unicode_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()  # 分割行
            if len(parts) == 2:  # 确保每行只有两个部分
                token, idx = parts
                vocab[token] = int(idx)  # 将 ID 转换为整数
            else:
                print(f"Skipping malformed line: {line.strip()}")  # 打印异常行
    return vocab



tokenizer = RegexTokenizer()
tokenizer.vocab =load_unicode_vocab("./models/regex.vocab")
ids = tokenizer.encode("Hello world")
print(ids)