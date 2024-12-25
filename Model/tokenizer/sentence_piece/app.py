import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Directory {dir} created.")

# 中文语料库准备
def convert_to_corpus():
    with open("data/《斗破苍穹》.txt", "r", encoding="utf-8") as fp:
        data = fp.read().strip().split("\n")
    sentences = []

    for d in data:
        d = d.strip()
        if "===" in d or len(d) == 0 or d == "《斗破苍穹》来自:":
            continue
        sentences.append(d)

    with open("data/corpus.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(sentences))

# 构建中文词库
def train_chinese_bpe():
    spm.SentencePieceTrainer.Train(
        input="data/corpus.txt",
        model_prefix="./models/chinese/tokenizer",
        vocab_size=50000,
        user_defined_symbols=['foo', 'bar'],
        character_coverage=1.0,
        model_type='bpe',
    )

# 把经由sentencepiece 训练出来的中文词库拼接到 llama 的词库上
def merge_chinese_vocab_into_llama():
    llama_tokenizer_dir = "./models/llama/tokenizer.model"
    chinese_sp_model_file = "./models/chinese/tokenizer.model"

    # load
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file)

    # 这里创建了一个空的 ModelProto 对象，ModelProto 是 SentencePiece 模型的 protobuf（协议缓冲区）格式的一部分。
    # 它通常用来存储和传输 SentencePiece 模型的结构信息（例如词汇表、分词器的训练配置等）。
    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

    # print number of tokens
    print(len(llama_tokenizer), len(chinese_sp_model))
    print(llama_tokenizer.all_special_tokens)
    print(llama_tokenizer.all_special_ids)
    print(llama_tokenizer.special_tokens_map)

    ## Add Chinese tokens to LLaMA tokenizer
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
    print(len(llama_spm_tokens_set))
    print(f"Before:{len(llama_spm_tokens_set)}")
    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
    print(f"New model pieces: {len(llama_spm.pieces)}")

    ## Save
    output_sp_dir = './models/llama_chinese/'
    file_name = 'tokenizer.model'  # the path to save Chinese-LLaMA tokenizer
    os.makedirs(output_sp_dir, exist_ok=True)
    with open(output_sp_dir + file_name, 'wb') as f:
        f.write(llama_spm.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + file_name)

    tokenizer.save_pretrained(output_sp_dir)
    print(f"Chinese-LLaMA tokenizer has been saved to {output_sp_dir}")

    # Test
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_sp_dir)
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    print(tokenizer.special_tokens_map)
    text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
    The primary use of LLaMA is research on large language models, including'''
    print("Test text:\n", text)
    print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
    print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")


def test(text):
    llama_tokenizer_dir = "./models/llama"
    llama_chinese_tokenizer_dir = "./models/llama_chinese"
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(llama_chinese_tokenizer_dir)

    print(chinese_llama_tokenizer.all_special_tokens)
    print(chinese_llama_tokenizer.all_special_ids)
    print(chinese_llama_tokenizer.special_tokens_map)


    print("Test text:\n", text)
    print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
    print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")



if __name__ == '__main__':
    # output_dir = "transformer"
    # make_dir(output_dir)
    # convert_to_corpus()
    # train_chinese_bpe()
    # merge_chinese_vocab_into_llama()

    text = '''本文详细介绍了如何利用Sentencepiece训练中文分词器，包括数据处理、构建训练语料、训练过程以及测试。
    接着，文章讲解了如何将自训练的分词器与LLaMA分词器进行合并，并测试了分词效率。提供了完整的GitHub资源链接。'''
    test(text)