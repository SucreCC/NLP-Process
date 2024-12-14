# BERT 中文文本二分类 
中文文本二分类，使用BERT预训练模型。
这个项目的目的是，在熟悉了 BERT-Chinese-Classification 的代码后。自己根据前面的代码仿写一个二分类的项目。

---
## 如何运行
1. 这个项目使用到的文件夹是如下。
- BERT-Chinese-Binary-Classification  (存放主要的代码)
- 'others/data/binary-classification/waimai-10k' （存放数据集和预训练模型）
2. 去 hugging face 下载 pytorch_model.bin文件， 然后放入到路径 /others/pretrain-model/bert-base-chinese 中。
- 下载链接 https://huggingface.co/google-bert/bert-base-uncased/tree/main
3. 运行run.py 文件即可。
4. 如果代码运行崩溃，就去 bert.py 文件中调节 batch_size 的大小。


