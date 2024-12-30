

# 文件
- config.py          llama3 配置文件
- tokenizer.py       llama3 tokenizer
- model.py           llama3 模型
- test_llama31.py    用于测试 （用于debug 查看代码）
- chat.py  用于开始会话（游玩）
- tiny_stories.py    下载tiny stories
- generation.py      llama3 构造器，用于创建一个llama3



# 如何下载meta的预训练模型
1. 去下面先填表格，选择模型
https://www.llama.com/llama-downloads/
2. 确认完之后会跳转到安装页面
按照指示进行下载，要下载的模型名字是 “meta-llama/Llama-3.1-8B”
3. 跳转的页面有一个 **Specify custom URL**, 在输入如下命令后需要输入这个 URL 才能开始下载

```bash
pip install llama-stack
bash llama model download --source meta --model-id  meta-llama/Llama-3.1-8B
```

4. 文件存放的位置
下载完成后会展示的