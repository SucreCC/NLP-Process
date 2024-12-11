import os
import time
import torch
import numpy as np

from public.wandbLoger import WandbLogger
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
from torch.cuda.amp import  autocast, GradScaler

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()

if __name__ == '__main__':

    logger = WandbLogger(
        project_name="BERT-Chinese-Classification",

        # api_key= os.environ.get("WANDB_API_KEY"),  # 如果已经设置了环境变量，可以省略
        run_name="experiment_1",
        config={
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 32,
        }
    )

    torch.cuda.set_per_process_memory_fraction(0.80)
    torch.backends.cudnn.benchmark = True
    # 默认是0异步进行， 设置为1 变成同步了
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


    dataset = '..'
    model_name = args.model
    x = import_module(model_name)
    config = x.Config(dataset)

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

    # print(train_iter.n_batches)
    # print(int(train_iter.n_batches))

    #  train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

    # wandb 结束运行
    logger.finish()
