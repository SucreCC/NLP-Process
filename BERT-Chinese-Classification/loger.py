import loger
import os
os.environ["WANDB_API_KEY"] = "f49fecc68453c53efcf05f3eef1ed2baddb92036"
wandb.login
wandb.init(project="BERT-Chinese-Classification")
