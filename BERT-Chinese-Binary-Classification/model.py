from torch import nn
from transformers import BertModel

import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(config.bert_path)

        # 设置BERT参数为可训练
        for param in self.bert.parameters():
            param.requires_grad = True

        # 最后一层全连接层，将hidden_size映射到单输出
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    # forward方法中，dataset包含 input_ids_list, seq_len, attention_mask
    def forward(self, dataset):
        input_ids_list = dataset[0]
        attention_mask = dataset[2]

        # 获取BERT的输出
        outputs = self.bert(input_ids=input_ids_list, attention_mask=attention_mask)

        # 提取池化后的向量
        pooled_output = outputs.pooler_output

        # 通过全连接层后，使用sigmoid激活函数，得到概率值
        logits = self.fc(pooled_output)
        probs = torch.sigmoid(logits)
        return probs