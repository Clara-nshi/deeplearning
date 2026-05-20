# 1 导包
import torch
import jieba
# 2 生成语料
text = 'rose是一朵玫瑰花，rose是一个kpop艺人，她也叫澳洲野玫瑰。'
# 3 分词
text_jieba = jieba.lcut(text)
# 4 去重构建词表
word_list = list(set(text_jieba))
# 5 创建嵌入层
embedding = torch.nn.Embedding(num_embeddings=len(word_list), embedding_dim=4)
# 6.使用嵌入曾生成词向量
for i, word in enumerate(word_list):
    embed = embedding(torch.tensor(i))
    print(i, word, embed)