import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from torch.optim import AdamW, optimizer
from transformers import get_linear_schedule_with_warmup


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


# def train(config, model, train_iter, dev_iter, test_iter):
#     start_time = time.time()
#     model.train()
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
#     ]
#     optimizer = AdamW(
#         optimizer_grouped_parameters,
#         lr=config.learning_rate)
#
#     total_steps = int(train_iter.n_batches) * config.num_epochs
#     # warmup比例，例如0.05表示前5%的训练步骤用于warmup
#     warmup_ratio = 0.05
#     warmup_steps = int(total_steps * warmup_ratio)
#
#     # 使用调度器创建学习率schedule，随训练步数逐步线性下降，且前warmup_steps步数进行学习率预热
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=total_steps
#     )
#
#     total_batch = 0
#     dev_best_loss = float('inf')
#     last_improve = 0
#     flag = False
#     model.train()
#     for epoch in range(config.num_epochs):
#         print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
#         for i, (trains, labels) in enumerate(train_iter):
#             outputs = model(trains)
#             time.sleep(0.2)
#             # 每50个batch沉睡一次
#             # if total_batch % 10 == 0 and total_batch != 0:
#
#             model.zero_grad()
#             loss = F.cross_entropy(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             if total_batch % 500 == 0:
#                 true = labels.data.cpu()
#                 predic = torch.max(outputs.data, 1)[1].cpu()
#                 train_acc = metrics.accuracy_score(true, predic)
#                 dev_acc, dev_loss = evaluate(config, model, dev_iter)
#                 if dev_loss < dev_best_loss:
#                     dev_best_loss = dev_loss
#                     torch.save(model.state_dict(), config.save_path)
#                     improve = '*'
#                     last_improve = total_batch
#                 else:
#                     improve = ''
#                 time_dif = get_time_dif(start_time)
#                 msg = 'Iter: {0:>6}, Train Loss: {1:>5.2f}, Train Acc: {2:>6.2%}, Val Loss: {3:>5.2f}, Val Acc: {4:>6.2%}, Time: {5}, {6}'
#                 print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
#                 model.train()
#             total_batch += 1
#             # if total_batch - last_improve > config.require_improvement:
#             #     print("No optimization for a long time, auto-stopping...")
#             #     flag = True
#             #     break
#
#         # if flag:
#         #     break
#     test(config, model, test_iter)

def train(config, model, train_iter, dev_iter, test_iter):
    model.train()  # 调整为只在外层调用一次
    start_time = time.time()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}  # 无衰减
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    total_steps = len(train_iter) * config.num_epochs  # 修正以确保步数与数据集和批次大小一致
    warmup_ratio = 0.05
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False

    accumulate_steps = 4  # 梯度累积的步数
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        optimizer.zero_grad()  # 初始化梯度

        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            time.sleep(0.2)

            loss = F.cross_entropy(outputs, labels) / accumulate_steps  # 累积梯度分摊损失
            loss.backward()

            if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_iter):
                # 每 accumulate_steps 步进行一次优化
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()  # 重置梯度

            if total_batch % 500 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2f}, Train Acc: {2:>6.2%}, Val Loss: {3:>5.2f}, Val Acc: {4:>6.2%}, Time: {5}, {6}'
                print(msg.format(total_batch, loss.item() * accumulate_steps, train_acc, dev_loss, dev_acc, time_dif, improve))

            total_batch += 1

    test(config, model, test_iter)




def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    label_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            label_all = np.append(label_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(label_all, predict_all)
    if test:
        report = metrics.classification_report(label_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(label_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)




# def evaluate(config, model, data_iter, test=False):
#     model.eval()
#     loss_total = 0
#     predict_all = np.array([], dtype=int)
#     label_all = np.array([], dtype=int)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)  # 将模型移动到 GPU
#
#     with torch.no_grad():
#         for texts, labels in data_iter:
#             texts, labels = texts.to(device), labels.to(device)  # 将数据移动到 GPU
#             outputs = model(texts)
#             loss = F.cross_entropy(outputs, labels)
#             loss_total += loss.item()  # 将 loss 转为标量
#             labels = labels.cpu().numpy()  # 将 labels 移回 CPU
#             predic = torch.max(outputs.data, 1)[1].cpu().numpy()  # 将预测结果移回 CPU
#             label_all = np.append(label_all, labels)
#             predict_all = np.append(predict_all, predic)
#
#     acc = metrics.accuracy_score(label_all, predict_all)
#     if test:
#         report = metrics.classification_report(label_all, predict_all, target_names=config.class_list, digits=4)
#         confusion = metrics.confusion_matrix(label_all, predict_all)
#         return acc, loss_total / len(data_iter), report, confusion
#     return acc, loss_total / len(data_iter)
