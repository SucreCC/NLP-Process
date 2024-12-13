import torch
from openai import batches
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd

PAD, SEP, CLS = '[PAD]', '[SEP]', '[CLS]'


def get_balance_corpus(corpus):
    positive_corpus = corpus[corpus['label'] == 1]
    negative_corpus = corpus[corpus['label'] == 0]
    sample_size = len(corpus) // 2
    balanced_corpus = pd.concat(
        [positive_corpus.sample(sample_size, replace=positive_corpus.shape[0] < sample_size),
         negative_corpus.sample(sample_size, replace=negative_corpus.shape[0] < sample_size)],
        ignore_index=True)
    balanced_corpus = balanced_corpus.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_corpus


def load_dataset(file_path, max_len, tokenizer):
    CLS_ids = tokenizer.convert_tokens_to_ids(CLS)

    corpus = pd.read_csv(file_path)
    balanced_corpus = get_balance_corpus(corpus)
    labels = [int(label) for label in balanced_corpus['label'].tolist()]
    inputs = balanced_corpus['review'].tolist()
    attention_masks = []
    seq_lens = []
    input_ids_list = []

    for input in tqdm(inputs):
        # add_special_tokens 在句首和句尾添加CLS 和 SEP 的id
        encoding = tokenizer(input, add_special_tokens=False)
        input_ids = [CLS_ids] + encoding['input_ids']
        input_ids_len = len(input_ids)

        if input_ids_len < max_len:
            attention_mask = [1] * input_ids_len + [0] * (max_len - input_ids_len)
            input_ids += [0] * (max_len - input_ids_len)
        else:
            attention_mask = [1] * max_len
            input_ids = input_ids[:max_len]
            input_ids_len = max_len

        input_ids_list.append(input_ids)
        attention_masks.append(attention_mask)
        seq_lens.append(input_ids_len)
    batches = [(input_ids_list, labels, attention_masks, seq_lens)]
    return batches


class DataIterator(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.device = device
        self.n_batches = len(batches) / batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, contents, labels, masks, seq_lens):
        x = torch.LongTensor(contents).to(self.device)
        y = torch.LongTensor(labels).to(self.device)
        mask = torch.LongTensor(masks).to(self.device)
        seq_len = torch.LongTensor(seq_lens).to(self.device)
        return x, y, mask, seq_len

    def __next__(self):
        if self.residue and self.index < self.n_batches:
            batches = self.batches[self.index * self.batch_size:len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return int(self.n_batches)


import model_config

model_config = model_config.ModelConfig()
batches = load_dataset(model_config.file_path, model_config.max_len, model_config.tokenizer)
print(batches)
