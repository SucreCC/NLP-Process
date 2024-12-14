import numpy as np
import torch
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
    CLS_ids = tokenizer.convert_tokens_to_ids('[CLS]')  # 改为标准 BERT CLS 标记
    contents = []

    # 读取数据
    corpus = pd.read_csv(file_path)
    balanced_corpus = get_balance_corpus(corpus)
    labels = [int(label) for label in balanced_corpus['label'].tolist()]
    inputs = balanced_corpus['review'].tolist()

    for input_text, label in tqdm(zip(inputs, labels), total=len(labels)):
        # 分词
        encoding = tokenizer(input_text, add_special_tokens=False)
        input_ids = [CLS_ids] + encoding['input_ids']
        input_ids_len = len(input_ids)

        # 截断或填充
        if input_ids_len < max_len:
            attention_mask = [1] * input_ids_len + [0] * (max_len - input_ids_len)
            input_ids += [0] * (max_len - input_ids_len)
            input_ids_len = max_len
        else:
            attention_mask = [1] * max_len
            input_ids = input_ids[:max_len]
            input_ids_len = max_len

        # 打包单个样本
        contents.append((input_ids, label, input_ids_len, attention_mask))

    return contents


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

    def _to_tensor(self, batches):
        x = torch.LongTensor([batch[0] for batch in batches]).to(self.device)
        y = torch.tensor([batch[1] for batch in batches], dtype=torch.float).to(self.device)
        # 把labels 由1维转换成2维 （batch_size, 1）
        y = y.view(-1, 1)
        attention_masks = torch.LongTensor([batch[2] for batch in batches]).to(self.device)
        seq_len = torch.LongTensor([batch[3] for batch in batches]).to(self.device)
        return (x, attention_masks, seq_len), y

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


def buildIterator(dataset, config):
    return DataIterator(dataset, config.batch_size, config.device)


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
