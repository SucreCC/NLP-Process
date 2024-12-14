import time
from importlib import import_module

import numpy as np
import torch
from transformers import BertModel
import model_config
from data_loader import load_dataset, buildIterator, get_time_dif
from model import Model
from train import train

if __name__ == '__main__':
    model_name = 'model'
    bert_model = import_module(model_name)
    config = model_config.ModelConfig()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    train_data = load_dataset(config.file_path, config.max_len, config.tokenizer)
    train_iter =  buildIterator(train_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    model = Model(config).to(config.device)
    train(config, model, train_iter)


