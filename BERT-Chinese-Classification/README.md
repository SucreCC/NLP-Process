# BERT 中文文本多分类 
中文文本多分类，使用BERT预训练模型。


---
## 如何运行
1. 这个项目使用到的文件夹是如下。
- BERT-Chinese-Classification  (存放主要的代码)
- others （存放数据集和预训练模型）
2. 去 hugging face 下载 pytorch_model.bin文件， 然后放入到路径 /others/pretrain-model/bert-base-chinese 中。
- 下载链接 https://huggingface.co/google-bert/bert-base-uncased/tree/main
3. 运行run.py 文件即可。
4. 如果代码运行崩溃，就去 bert.py 文件中调节 batch_size 的大小。
5. 为了降低笔记本的压力，各个数据集都调成只有 256 条记录， 原始记录在它们的 copy 文件中。


