import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class Config(object):
    def __init__(self, dataset):
        # 模型名称，用于保存模型文件名和日志标识
        self.model_name = 'bert'

        # 训练集数据文件的路径
        self.train_path = dataset + '/others/data/THUCNews/train copy.txt'
        # 验证集数据文件的路径
        self.dev_path = dataset + '/others/data/THUCNews/dev copy.txt'
        # 测试集数据文件的路径
        self.test_path = dataset + '/others/data/THUCNews/test copy.txt'

        # 类别列表文件：其中每行一个类别名称，将其读取为列表
        # 用于分类任务时确定num_classes和对预测结果进行解码
        with open(dataset + '/others/data/THUCNews/class.txt', 'r', encoding='utf-8') as f:
            self.class_list = [x.strip() for x in f.readlines()]

        # 模型训练后参数保存的路径，训练完成或中途保存checkpoint时使用
        self.save_path = './save/' + self.model_name + '.ckpt'

        # 设备选择：如果有GPU可用，则使用GPU，否则使用CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 如果在连续的1000个batch中验证集性能（如准确率或损失等）无提升
        # 则提前终止训练，防止模型过度训练浪费时间
        self.require_improvement = 1000

        # 类别数量，根据class_list的长度得出
        # 用于输出层的线性映射和计算分类损失时的维度
        self.num_classes = len(self.class_list)

        # 训练过程的轮数（epoch）
        # 一个epoch是指完整地对训练集数据进行一遍训练
        self.num_epochs = 3

        # mini-batch大小，即每次参数更新时所使用的样本数
        # 决定了内存使用量和训练稳定性，通常GPU显存越大可以选用越大的batch_size
        self.batch_size = 8

        # 每句输入文本被统一填充或截断到的固定长度
        # 短的句子会padding填充到该长度，长的句子则截断
        # 确保模型输入的维度一致
        self.pad_size = 32

        # 学习率，用于控制参数更新的步幅大小
        # 通常建议fine-tuning Bert时使用较小的学习率，e.g., 5e-5或2e-5
        self.learning_rate = 5e-5

        # Bert预训练模型的路径
        # 可以是本地路径或Hugging Face提供的预训练模型名（如'bert-base-chinese'）
        # 该路径下应包含config.json、pytorch_model.bin、vocab.txt等文件
        self.bert_path = dataset + '/others/pretrain-model/bert-base-chinese'

        # 初始化BERT的分词器(Tokenizer)
        # 用于将原始文本转换为模型可接受的token序列
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        # Bert的隐藏层维度，一般为768，对于Bert-base是768维
        # 决定分类层全连接层输入维度的大小
        self.hidden_size = 768

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        input_ids = x[0]
        attention_mask = x[2]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        out = self.fc(pooled_output)
        return out
