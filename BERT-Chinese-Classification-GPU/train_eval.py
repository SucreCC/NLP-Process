import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

from public.GPU_monitor import GPUMonitor
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


# 原始的train 方法

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







# 优化后的代码
def train(config, model, train_iter, dev_iter, test_iter):
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    total_steps = len(train_iter) * config.num_epochs
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

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        optimizer.zero_grad()

        for i, (trains, labels) in enumerate(train_iter):
            # 确保训练数据和标签在正确的设备上
            # trains, labels = trains.to(device), labels.to(device)

            outputs = model(trains)
            time.sleep(0.15)  # time sleep, 减小由GPU利用率过高造成的显示器黑屏问题

            loss = F.cross_entropy(outputs, labels) / accumulate_steps  # 累积梯度分摊损失
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
                dev_acc, dev_loss = evaluate(config, model, dev_iter)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2f}, Train Acc: {2:>6.2%}, Val Loss: {3:>5.2f}, Val Acc: {4:>6.2%}, Time: {5}, {6}'
                print(
                    msg.format(total_batch, loss.item() * accumulate_steps, train_acc, dev_loss, dev_acc, time_dif, improve))
            total_batch += 1

    test(config, model, test_iter)




















# 使用 torch.cuda.memory_summary(device=config.device) 打印输入太多，不好观察
# 以下是观察结果
# 观察与分析
# 1. 模型训练前后对比
# Cur Usage (当前使用)：从 1187 MiB 增加到 1338 MiB，再到 1733 MiB，有小幅增长。
# Peak Usage (峰值使用)：始终为 2363 MiB，表明高峰内存使用稳定，可能来自模型加载和初始化。
# Tot Alloc (总分配内存)：约 769 GiB 至 1037 GiB，表明训练中多次分配和释放内存，增长较快。
# Tot Freed (总释放内存)：释放内存量也随着分配增长，表明 PyTorch 的动态内存管理较为高效。
# 结论：
#
# GPU 内存的增长主要来源于模型加载后的训练过程，但总体增长量较小，峰值内存并未改变。
# 怀疑点：动态内存分配可能引发短时的 GPU 利用率激增。
# 2. 损失计算前后对比
# 损失计算前后的内存变化很小（当前使用稳定在 1338 MiB）。
# 损失计算未显著引发 GPU 的高峰使用，表明可能是前向传播或其他阶段的内存分配引发了波动。
# 结论：
#
# 损失计算过程并未引发显著的内存需求或释放，因此激增问题很可能发生在前向传播或梯度反向传播。
# 3. 模型评估的影响
# 每 500 个 batch 进行一次模型评估：
# Cur Usage (当前使用)：从 1583 MiB 稳定到 1583 MiB。
# Tot Alloc (总分配)：显著增加，从 769 GiB 到 1037 GiB，进一步上升到 1271 GiB。
# Non-releasable Memory (不可释放内存)：从 160 KiB 增加到 181 KiB。
# 结论：
#
# 每 500 个 batch 的评估触发了一些额外的 GPU 内存分配，虽然释放了部分内存，但有少量不可释放内存（Non-releasable memory）逐渐累积。
# GPU 利用率激增可能与模型评估有关，尤其是多次验证集计算时。
# 4. 内存分配的主要特征
# 动态内存分配
# Large Pool vs. Small Pool:
# 大块分配（Large Pool）主导内存需求，占用 2592 MiB。
# 小块分配（Small Pool）使用较少，占用 102 MiB。
# 动态分配特点：
# 每次训练过程中总分配和释放内存量非常高（数百 GiB），表明有大量动态分配操作。
# 不可释放内存
# Non-releasable Memory: 有小幅增长，但未显著累积。
# 表明内存未完全释放的原因可能是由某些 GPU 操作的生命周期未结束（如缓存、未使用的临时张量）。


# def train(config, model, train_iter, dev_iter, test_iter):
#     model.train()  # 调整为只在外层调用一次
#     start_time = time.time()
#
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}  # 无衰减
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
#
#     total_steps = len(train_iter) * config.num_epochs  # 修正以确保步数与数据集和批次大小一致
#     warmup_ratio = 0.05
#     warmup_steps = int(total_steps * warmup_ratio)
#
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
#
#     accumulate_steps = 4  # 梯度累积的步数
#     for epoch in range(config.num_epochs):
#         print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
#         optimizer.zero_grad()  # 初始化梯度
#
#         for i, (trains, labels) in enumerate(train_iter):
#
#             print("模型训练前" + torch.cuda.memory_summary(device=config.device))
#             outputs = model(trains)
#             print("模型训练后" + torch.cuda.memory_summary(device=config.device))
#
#             time.sleep(0.2)
#
#             print("计算损失前" + torch.cuda.memory_summary(device=config.device))
#             loss = F.cross_entropy(outputs, labels) / accumulate_steps  # 累积梯度分摊损失
#             print("计算损失后" + torch.cuda.memory_summary(device=config.device))
#
#             loss.backward()
#             print(torch.cuda.memory_summary(device=config.device))
#
#             if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_iter):
#                 # 每 accumulate_steps 步进行一次优化
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()  # 重置梯度
#
#                 print("计算累计梯度损失后" + torch.cuda.memory_summary(device=config.device))
#
#             if total_batch % 500 == 0:
#                 print(torch.cuda.memory_summary(device=config.device))
#                 true = labels.data.cpu()
#                 predic = torch.max(outputs.data, 1)[1].cpu()
#                 train_acc = metrics.accuracy_score(true, predic)
#                 dev_acc, dev_loss = evaluate(config, model, dev_iter)
#                 print("每500batch进行一次模型评估" +  torch.cuda.memory_summary(device=config.device))
#
#                 if dev_loss < dev_best_loss:
#                     dev_best_loss = dev_loss
#                     torch.save(model.state_dict(), config.save_path)
#                     improve = '*'
#                     last_improve = total_batch
#                 else:
#                     improve = ''
#
#                 time_dif = get_time_dif(start_time)
#                 msg = 'Iter: {0:>6}, Train Loss: {1:>5.2f}, Train Acc: {2:>6.2%}, Val Loss: {3:>5.2f}, Val Acc: {4:>6.2%}, Time: {5}, {6}'
#                 print(msg.format(total_batch, loss.item() * accumulate_steps, train_acc, dev_loss, dev_acc, time_dif, improve))
#
#             total_batch += 1
#
#     test(config, model, test_iter)


# 停用在0.2的程序沉睡，会导致gpu的利用率一直大于90%
# 采用 torch.profiler 进行性能分析， 每500个epoch 就存储日志
# 采用 torch.profiler 写入日志需要很长时间

# from torch.profiler import tensorboard_trace_handler
# def train(config, model, train_iter, dev_iter, test_iter):
#     model.train()  # 调整为只在外层调用一次
#     start_time = time.time()
#
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}  # 无衰减
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
#
#     total_steps = len(train_iter) * config.num_epochs  # 修正以确保步数与数据集和批次大小一致
#     warmup_ratio = 0.05
#     warmup_steps = int(total_steps * warmup_ratio)
#
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=total_steps
#     )
#
#     total_batch = 0
#     dev_best_loss = float('inf')
#     accumulate_steps = 4  # 梯度累积的步数
#
#     # 创建 TensorBoard 日志记录器
#     trace_handler = tensorboard_trace_handler('./profiler_logs')
#
#     with torch.profiler.profile(
#             activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#             with_stack=True
#     ) as prof:
#         for epoch in range(config.num_epochs):
#             print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
#             optimizer.zero_grad()  # 初始化梯度
#
#             for i, (trains, labels) in enumerate(train_iter):
#                 with torch.profiler.record_function("Model Forward Pass"):
#                     outputs = model(trains)
#                     time.sleep(0.15)  # time sleep, 减小由gpu利用率 （>90%）过高造成的显示器黑屏的问题
#
#                 with torch.profiler.record_function("Loss Computation"):
#                     loss = F.cross_entropy(outputs, labels) / accumulate_steps  # 累积梯度分摊损失
#
#                 with torch.profiler.record_function("Backward Pass"):
#                     loss.backward()
#
#                 if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_iter):
#                     with torch.profiler.record_function("Optimizer Step"):
#                         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
#                         optimizer.step()
#                         scheduler.step()
#                         optimizer.zero_grad()  # 重置梯度
#
#                 if total_batch % 500 == 0:
#                     with torch.profiler.record_function("Evaluation"):
#                         true = labels.data.cpu()
#                         predic = torch.max(outputs.data, 1)[1].cpu()
#                         train_acc = metrics.accuracy_score(true, predic)
#                         dev_acc, dev_loss = evaluate(config, model, dev_iter)
#
#                     if dev_loss < dev_best_loss:
#                         dev_best_loss = dev_loss
#                         torch.save(model.state_dict(), config.save_path)
#                         improve = '*'
#                         last_improve = total_batch
#                     else:
#                         improve = ''
#
#                     time_dif = get_time_dif(start_time)
#                     msg = 'Iter: {0:>6}, Train Loss: {1:>5.2f}, Train Acc: {2:>6.2%}, Val Loss: {3:>5.2f}, Val Acc: {4:>6.2%}, Time: {5}, {6}'
#                     print(
#                         msg.format(total_batch, loss.item() * accumulate_steps, train_acc, dev_loss, dev_acc, time_dif,
#                                    improve))
#                 total_batch += 1
#
#                 # 手动步进 Profiler
#                 prof.step()
#                 # 在每个 iter 后手动调用 on_trace_ready
#                 trace_handler(prof)
#
#         # 延迟打印 Profiler 数据
#         print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#     test(config, model, test_iter)




# use GPU monitor to get GPU information for special step
# compute loss he backward 这两步对gpu的利用率和gpu内存的占用率不高

# def train(config, model, train_iter, dev_iter, test_iter):
#     # 引入 GPU Monitor 用来监控GPU的使用情况
#     monitor = GPUMonitor(log_dir="./save/logs/GPU-monitor")
#
#     model.train()  # 调整为只在外层调用一次
#     start_time = time.time()
#
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}  # 无衰减
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
#
#     total_steps = len(train_iter) * config.num_epochs  # 修正以确保步数与数据集和批次大小一致
#     warmup_ratio = 0.05
#     warmup_steps = int(total_steps * warmup_ratio)
#
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=total_steps
#     )
#
#     total_batch = 0
#     dev_best_loss = float('inf')
#     accumulate_steps = 4  # 梯度累积的步数
#
#     for epoch in range(config.num_epochs):
#         print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
#         optimizer.zero_grad()  # 初始化梯度
#
#         for i, (trains, labels) in enumerate(train_iter):
#
#             # 启动监控任务
#             monitor.start_monitoring(task_name="Model training step", interval=0.1)
#             outputs = model(trains)
#             # 停止监控任务
#             monitor.stop_monitoring()
#
#             time.sleep(0.15)  # time sleep, 减小由gpu利用率 （>90%）过高造成的显示器黑屏的问题
#
#             # 启动监控任务
#             # monitor.start_monitoring(task_name="Compute loss step", interval=0.1)
#             loss = F.cross_entropy(outputs, labels) / accumulate_steps  # 累积梯度分摊损失
#             # 停止监控任务
#             # monitor.stop_monitoring()
#
#             # 启动监控任务
#             # monitor.start_monitoring(task_name="Backward step", interval=0.1)
#             loss.backward()
#             # 停止监控任务
#             # monitor.stop_monitoring()
#
#             if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_iter):
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()  # 重置梯度
#
#             if total_batch % 500 == 0:
#                 true = labels.data.cpu()
#                 predic = torch.max(outputs.data, 1)[1].cpu()
#
#                 # 启动监控任务
#                 monitor.start_monitoring(task_name="calculate accuracy for each 500 step", interval=0.1)
#                 train_acc = metrics.accuracy_score(true, predic)
#                 dev_acc, dev_loss = evaluate(config, model, dev_iter)
#                 # 停止监控任务
#                 monitor.stop_monitoring()
#
#                 if dev_loss < dev_best_loss:
#                     dev_best_loss = dev_loss
#                     torch.save(model.state_dict(), config.save_path)
#                     improve = '*'
#                     last_improve = total_batch
#                 else:
#                     improve = ''
#
#                 time_dif = get_time_dif(start_time)
#                 msg = 'Iter: {0:>6}, Train Loss: {1:>5.2f}, Train Acc: {2:>6.2%}, Val Loss: {3:>5.2f}, Val Acc: {4:>6.2%}, Time: {5}, {6}'
#                 print(
#                     msg.format(total_batch, loss.item() * accumulate_steps, train_acc, dev_loss, dev_acc, time_dif,
#                                improve))
#             total_batch += 1
#
#     test(config, model, test_iter)




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



# 优化前的代码
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

