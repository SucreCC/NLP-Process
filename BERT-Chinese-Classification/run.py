import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif


if __name__ == '__main__':
    dataset = '..'
    model_name = 'bert'
    bert_model = import_module(model_name)
    config = bert_model.Config(dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    #  train
    model = bert_model.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
