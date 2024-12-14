import time

import torch
from sklearn import metrics
from torch.optim._multi_tensor import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from data_loader import get_time_dif


def train(config, model, train_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    total_steps = train_iter.n_batches
    warmup_ratio = 0.05
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # dataset 包含 input_ids_list, seq_len, attention_mask
        for i, (dataset, labels) in enumerate(train_iter):
            outputs = model(dataset)
            model.zero_grad()

            # binary_cross_entropy 会直接接受概率值和真实标签，因此不需要将 outputs 转换成离散的标签格式（如 0 或 1）
            loss = F.binary_cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # 1个batch 128 条记录， 每10个batch打印一次
            if i % 10 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data,1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2f}, Train Acc: {2:>6.2%}, Time: {3}'
                print(msg.format(i, loss.item(), train_acc, time_dif))



