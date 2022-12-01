'''
Author: LiHang
Date: 2022-11-10 09:49:39
LastEditors: LiHang
LastEditTime: 2022-11-11 18:26:31
Description: 
'''
import os
import time
from jieba import cut
from tqdm import tqdm


def getVocab(path):
    def tokenizer(x): return [i for i in x]  # cut
    maxSize = 300000
    minFreq = 1
    unk = "<UNK>"
    pad = "<PAD>"
    vocab = {}
    with open(path, 'r', encoding="utf-8") as f:
        # k = 1
        for lines in tqdm(f):
            # time.sleep(0.05)
            # k += 1
            # if k == 100:
            #     break
            line = lines.split("\t")
            sentence = line[1].replace("\n", "")
            sentence = [i for i in tokenizer(sentence)]
            for i in sentence:
                if i in vocab:
                    vocab[i] += 1
                else:
                    vocab[i] = 1
    vocabList = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    vocab = {}  # 为了按照频率顺序，而不是扫描出现的顺序，这里置空一下
    # print(vocabList)
    for i in range(min(maxSize, len(vocabList))):
        # print(vocabList[i][1])
        # print(vocabList[i][0])
        if vocabList[i][1] >= minFreq:
            vocab[vocabList[i][0]] = i
        else:
            break
    vocab.update({unk: len(vocab), pad: len(vocab) + 1})
    return vocab


# 写入文件
def vocab2file(path, vocab):
    with open(path, 'w', encoding="utf-8") as f:
        for i in vocab.keys():
            f.write(str(i) + "\t" + str(vocab[i]) + "\n")


# path1_in = "dataset/1_nlpcc2013data-nlpcc2013_data/train.txt"
# vocab1 = getVocab(path1_in)
# print(len(vocab1))

#   word    char
# 1 33020   4341
# 2 33367   4106
# 3 124579  5533

# path1_out = "dataset/3_nlpcc2017_news_headline_categorization-master/wordVocab.txt"
# vocab2file(path1_out,vocab1)

# 统计句子长度


def getlength(path):
    def tokenizer(x): return [i for i in x]  # cut
    f = open(path, 'r', encoding="utf-8")

    list = {}
    for line in tqdm(f):
        line = line.split("\t")
        sentence = line[1].replace("\n", "")
        sentence = [i for i in tokenizer(sentence)]
        if len(sentence) in list.keys():
            list[len(sentence)] += 1
        else:
            list[len(sentence)] = 1
    # print(type(list))
    list = sorted(list.items(), key=lambda x: x[0])
    x = []
    y = []
    for i in list:
        x.append(i[0])
        y.append(i[1])
    sum = 0
    for i in y:
        sum += i
    print(sum)
    for i in range(len(y)):
        y[i] = 0.1 * y[i] / sum

    return x, y
# path = 'dataset/3_nlpcc2017_news_headline_categorization-master/train_1.txt'
# x, y = getlength(path)
# print(x)
# print(y)
