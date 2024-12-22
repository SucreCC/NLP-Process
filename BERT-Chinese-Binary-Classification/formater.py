import re


def convert_text_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()

        for line in lines:
            # 移除行首和行尾的多余空格，并去掉 ,,（如果存在）
            line = line.strip().replace(',,', '')

            # 匹配文本和情感值之间的关系
            match = re.match(r'(.*?)(\d+)', line)
            if match:
                # 提取文本和情感值
                text = match.group(1).strip()
                sentiment = match.group(2)

                # 写入转换后的格式
                outfile.write(f"{text}\t{sentiment}\n")



# 使用方法
input_file = '../others/data/THUCNews/binary_cls/goods.txt'  # 输入文件路径
output_file = '../others/data/THUCNews/binary_cls/goods_ft.txt'  # 输出文件路径
convert_text_format(input_file, output_file)
