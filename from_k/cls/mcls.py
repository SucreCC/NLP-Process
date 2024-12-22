import time

import torch
from sklearn import metrics
from transformers import BertForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import time
from sklearn import metrics
import os
from tqdm import tqdm
from datetime import timedelta



def build_dataset():
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = line.split('\t')
                token = tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train = load_dataset(train_path, pad_size)
    dev = load_dataset(dev_path, pad_size)
    test = load_dataset(test_path, pad_size)

    return train, dev, test



class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) / batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
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


def build_iterator(dataset):
    iter = DatasetIterater(dataset, 16, device)
    return iter


class Model(nn.Module):
    def __init__(self, n_cls, hidden_size,model_name):
        super(Model, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_size, n_cls)

    def forward(self, x):
        input_ids = x[0]
        attention_mask = x[2]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.pooler_output
        out = self.fc(outputs)
        return out




def get_time_dif(start_time):
    end_time = time.time()
    time.dif = end_time - start_time
    return timedelta(seconds=int(round(time.dif)))



def test(n_cls, model, test_iter):
    # model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(n_cls, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(n_cls, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    label_all = []
    with torch.no_grad():
        for texts, labels in data_iter:
            # # 如果 texts 和 labels 不是张量，先转换为张量
            # if not isinstance(texts, torch.Tensor):
            #     texts = torch.tensor(texts, dtype=torch.float32)
            # if not isinstance(labels, torch.Tensor):

            #     labels = torch.tensor(labels, dtype=torch.long)
            #
            # # 将张量移动到指定设备（如 GPU）
            # texts, labels = texts.to(config.device), labels.to(config.device)

            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()
            predic = torch.max(outputs, 1)[1]
            predict_all.append(predic)
            label_all.append(labels)

    # 在 GPU 上拼接结果，并在最后转为 NumPy 数组
    predict_all = torch.cat(predict_all).cpu().numpy()
    label_all = torch.cat(label_all).cpu().numpy()

    acc = metrics.accuracy_score(label_all, predict_all)

    if test:
        report = metrics.classification_report(label_all, predict_all, target_names=n_cls, digits=4)
        confusion = metrics.confusion_matrix(label_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion

    return acc, loss_total / len(data_iter)




# 优化后的代码
def train(n_cls, model, train_iter, dev_iter, test_iter):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # 确保模型在正确的设备上
    model.train()  # 调整为只在外层调用一次

    start_time = time.time()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    total_steps = len(train_iter) * num_epochs
    warmup_ratio = 0.05
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    total_batch = 0
    dev_best_loss = float('inf')
    accumulate_steps = 4  # 梯度累积的步数

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        optimizer.zero_grad()

        for i, (trains, labels) in enumerate(train_iter):
            # 确保训练数据和标签在正确的设备上
            # trains, labels = trains.to(device), labels.to(device)

            outputs = model(trains)
            # time.sleep(0.1)  # time sleep, 减小由GPU利用率过高造成的显示器黑屏问题

            loss = F.binary_cross_entropy(outputs, labels) / accumulate_steps  # 累积梯度分摊损失
            loss.backward()

            if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_iter):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if total_batch % 500 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(n_cls, model, dev_iter)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2f}, Train Acc: {2:>6.2%}, Val Loss: {3:>5.2f}, Val Acc: {4:>6.2%}, Time: {5}, {6}'
                print(
                    msg.format(total_batch, loss.item() * accumulate_steps, train_acc, dev_loss, dev_acc, time_dif, improve))
            total_batch += 1

    test(n_cls, model, test_iter)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["TF_FORCE_GPU_ALLOW_CROWTH"] = "true"

    train_path = './dataset/train.txt'
    dev_path = './dataset/train.txt'
    test_path = './dataset/train.txt'

    hidden_size = 768
    n_cls = 2


    PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = Model(n_cls, hidden_size,PRE_TRAINED_MODEL_NAME )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    pad_size = 32

    PAD, CLS = '[PAD]', '[CLS]'

    learning_rate = 5e-5
    num_epochs = 3






    train_data, dev_data, test_data = build_dataset()
    train_iter = build_iterator(train_data)
    dev_iter = build_iterator(dev_data)
    test_iter = build_iterator(test_data)

    random_seed = 1221

    bert = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=2)
    for param in bert.parameters():
        param.requires_grad = True



    train(n_cls, model, train_iter, dev_iter, test_iter)