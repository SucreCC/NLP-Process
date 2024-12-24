import os

import sentencepiece as spm

# 中文预料库准备
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
def train_bpe():
    spm.SentencePieceTrainer.Train(
        input="data/corpus.txt",
        model_prefix="transformer_output/tokenizer",
        vocab_size=50000,
        user_defined_symbols=['foo', 'bar'],
        character_coverage=1.0,
        model_type='bpe',
    )



if __name__ == '__main__':
    output_dir = "transformer_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory {output_dir} created.")
    convert_to_corpus()
    train_bpe()