import time

import torch
from sklearn import metrics
from torch.optim._multi_tensor import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from data_loader import get_time_dif


import time
import torch
from sklearn import metrics
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import BCEWithLogitsLoss


def train(config, model, train_iter):
    start_time = time.time()
    model.train()

    # 优化器参数分组
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
    ]

    # 优化器和调度器
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    total_steps = train_iter.n_batches
    warmup_ratio = 0.05
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # 损失函数
    loss_fn = BCEWithLogitsLoss()

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))

        for i, (dataset, labels) in enumerate(train_iter):


            outputs = model(dataset)  # 模型输出 logits
            time.sleep(0.5)
            labels = labels.float()  # 确保标签是浮点类型

            # 计算损失
            outputs = outputs.squeeze()  # 将 [128, 1] 转为 [128]
            loss = loss_fn(outputs, labels.view(-1))  # 确保 labels 是一维

            optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率

            # 每 10 个 batch 打印一次信息
            if i % 10 == 0:
                true = labels.data.cpu().numpy()
                predic = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()  # 阈值 0.5
                train_acc = metrics.accuracy_score(true, predic)

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2f}, Train Acc: {2:>6.2%}, Time: {3}'
                print(msg.format(i, loss.item(), train_acc, time_dif))
