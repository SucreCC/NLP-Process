# 1 如何让英文大模型支持中文
训练并合并中文分词器与 LLaMA 分词器，确保 LLaMA 能处理中文文本。

# 目录结构
- data  存放语料库数据
- models 存放各种分词器
  - chinese 用中文语料训练生成的中文分词器
  - llama  llama 的英文分词器
  - llama_chinese llama 和 chinese 拼合后的分词器

# app.py
	1.数据准备：从中文小说《斗破苍穹》提取文本数据，清洗并保存为语料文件。
	2.训练中文分词器：使用 SentencePiece 库训练一个 BPE（字节对编码）模型，生成中文分词器。
	3.合并分词器：将自训练的中文分词器与 LLaMA 的分词器合并，确保 LLaMA 能处理中文。
	4.保存和加载分词器：保存合并后的分词器并加载进行测试。
	5.测试功能：通过测试文本，验证合并后的中文 LLaMA 分词器是否有效，并输出与原 LLaMA 分词器的对比结果。