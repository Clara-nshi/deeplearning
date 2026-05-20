# 导包
import jieba
import torch
# 1. 准备文本语料
text = "北京东奥的进度条已经过半。小明考上了中国的大学"
# 2. jieba分词
words = jieba.lcut(text)
print(len(words))
# 3. 去重创建词表为了构建词表
word_list = list(set(words))
print(len(word_list))
# todo 4. 创建词嵌入层，num_embeddings传入的 >= 去重后的词数量，尽量相等
embedding = torch.nn.Embedding(num_embeddings=len(word_list), embedding_dim=4)
# todo 5. 使用词嵌入层生成词向量
for i, word in enumerate(word_list):
    embed = embedding(torch.tensor(i))
    print(i, word, embed)