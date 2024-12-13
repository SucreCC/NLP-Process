import torch
from transformers import BertTokenizer


class ModelConfig(object):
    def __init__(self):
        self.model_name = 'BERT-Chinese-Binary-Classification'
        self.file_path = '../others/data/binary-classification/waimai-10k/waimai_10k.csv'
        self.test_file_path = '../others/data/binary-classification/waimai-10k/test.csv'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 2
        self.num_epochs = 3
        self.batch_size = 128
        self.max_len = 32
        self.learning_rate = 5e-5
        self.bert_path = '../others/pretrain-model/bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
