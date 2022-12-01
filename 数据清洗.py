'''
Author: LiHang
Date: 2022-11-08 11:10:18
LastEditors: LiHang
LastEditTime: 2022-11-10 09:49:17
Description: 
'''

import os
import matplotlib.pyplot as plt
import numpy as np


##### 数据清洗 #####
listCN = ["悲伤", "高兴", "喜好", "厌恶", "惊讶", "愤怒", "恐惧", "无"]
listEN = ["sadness", "happiness", "like", "disgust", "surprise", "anger", "fear", "none"]
listNum = ["1", "2", "3", "4", "5", "6", "7", "8"]
path1 = "dataset/1_nlpcc2013data-nlpcc2013_data/test微博情绪标注语料.xml"
path2 = "dataset/1_nlpcc2013data-nlpcc2013_data/train微博情绪样例数据V5-13.xml"
pathTest = "dataset/1_nlpcc2013data-nlpcc2013_data/test.txt"
pathTrain = "dataset/1_nlpcc2013data-nlpcc2013_data/train.txt"

def dealemo(s, flag):
    '''
    输入：<sentence id="2" opinionated="Y" emotion-1-type="厌恶" emotion-2-type="无">念得我各种烦躁……</sentence>
    输出: disgust,念得我各种烦躁……
    '''
    list = s.strip().split(">")
    emo = list[0].split(" ")[3][16:-1]
    if flag:
        emo = listNum[listEN.index(emo)]
    else:
        emo = listNum[listCN.index(emo)]
    sentence = list[1][:-10]
    return emo, sentence

def dealnone(s, flag):
    '''
    输入：<sentence id="4" opinionated="N">她之前还做了天使光圈翅膀的动作。</sentence>
    输出：无,她之前还做了天使光圈翅膀的动作。
    '''
    
    if flag:
        emo = "none"
        emo = listNum[listEN.index(emo)]
    else:
        emo = "无"
        emo = listNum[listCN.index(emo)]
    list = s.strip().split(">")
    sentence = list[1][:-10]
    return emo, sentence


def data1test():
    sentences = []
    i = 1
    with open(path1,"r", encoding = "utf-8") as f:
        for lines in f:
            if "opinionated=\"Y\"" in lines:#有情绪的
                emo, sentence = dealemo(lines, False)
                emo_sentence = emo + "	" + sentence
                sentences.append(emo_sentence)
            if "opinionated=\"N\"" in lines:#没有情绪的
                emo, sentence = dealnone(lines, False)
                emo_sentence = emo + "	" + sentence
                sentences.append(emo_sentence)
                        
    with open(pathTest, "w", encoding = "utf-8") as f:
        for i in sentences:
            f.write(i + "\n")
#数据1test
#data1test()

def data1train():
    sentences = []
    i = 1
    with open(path2,"r", encoding = "utf-8") as f:
        for lines in f:
            # i += 1
            # if i == 1000:
            #     break
            if "emotion_tag=\"Y\"" in lines:#有情绪的
                emo, sentence = dealemo(lines, True)
                emo_sentence = emo + "	" + sentence
                #print(emo_sentence)
                sentences.append(emo_sentence)
            if "emotion_tag=\"N\"" in lines:#没有情绪的
                emo, sentence = dealnone(lines, True)
                emo_sentence = emo + "	" + sentence
                #print(emo_sentence)
                sentences.append(emo_sentence)
                        
    with open(pathTrain, "w", encoding = "utf-8") as f:
        for i in sentences:
            f.write(i + "\n")
#数据1train
#data1train()

path3 = "dataset/2_NLPCC2014_sentiment-master/sample.negative.txt"
path4 = "dataset/2_NLPCC2014_sentiment-master/sample.positive.txt"
pathTrain = "dataset/2_NLPCC2014_sentiment-master/train.txt"
path5 = "dataset/2_NLPCC2014_sentiment-master/test.label.cn.txt"
pathTest = "dataset/2_NLPCC2014_sentiment-master/test.txt"
#  "0", "1" = neg, pos
def data2train(path,label):
    with open(path,"r", encoding = "utf-8") as f:
        i = 1
        s = ""
        sentences = []
        for lines in f:
            if "<review id=\"" in lines:
                s = ""
                continue
            elif "</review>" in lines:
                sentences.append(label + "	" + s)
            else:
                s = s[:-1] + lines

    with open(pathTrain, "a+", encoding = "utf-8") as f:
        for i in sentences:
            f.write(i)       

#数据2train
# data2train(path3, "0")
# data2train(path4, "1")

def data2test():
    sentences = []
    with open(path5, 'r', encoding ="utf-8") as f:
        for lines in f:
            if "<review id=" in lines:
                label = lines[-4:-3]
                s = ""
            elif "</review>" in lines:
                sentences.append(label + "	" + s)
            else:
                s = s + lines

    with open(pathTest, "w", encoding = "utf-8") as f:
        for i in sentences:
            f.write(i)
#数据2trest
#data2test()        

path6 = "dataset/3_nlpcc2017_news_headline_categorization-master/train.txt"
path7 = "dataset/3_nlpcc2017_news_headline_categorization-master/dev.txt"
path8 = "dataset/3_nlpcc2017_news_headline_categorization-master/test.txt"
pathTrain = "dataset/3_nlpcc2017_news_headline_categorization-master/train_1.txt"
pathDev = "dataset/3_nlpcc2017_news_headline_categorization-master/dev_1.txt"
pathTest = "dataset/3_nlpcc2017_news_headline_categorization-master/test_1.txt"
label = ["1", "2", "3", "4", "5", "6", "7", "8","9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
category = ["baby", "car", "discovery", "entertainment", "essay", "fashion", "finance", "food",
        "game", "history", "military", "regimen", "society", "sports", "story", "tech", "travel", "world"
]

def data3(pathA, pathB):
    sentences = []
    with open(pathA, 'r', encoding = "utf-8") as f:
        i =1
        for lines in f:
            line = lines.strip().split("	")
            labelNum = label[category.index(line[0])]
            sentence = line[1].replace(" ","")
            sentences.append(labelNum + "	" + sentence + '\n')
        
    with open(pathB, 'w', encoding = "utf-8") as f:
        for i in sentences:
            f.write(i)

#数据3
# data3(path6, pathTrain)
# data3(path7, pathDev)
# data3(path8, pathTest)




#数据量可视化
#扫描数据集
def scan(path):
    data1 = []
    for i in range(20):
        data1.append(0)
    
    with open(path, 'r',encoding = "utf-8") as f:
        for lines in f:
            l= lines.split("\t")
            data1[ int(l[0]) -1 ] += 1
    i = data1.index(0)
    data1 = data1[:i]
    data2 = [str(i+1) for i in range(len(data1))]
    return data1, data2
# dset1_1, dset1_2 = scan("dataset/3_nlpcc2017_news_headline_categorization-master/dev_1.txt")
# print("s")
# print(dset1_1)
# print(dset1_2)
# print("e")






# dset2_1, dset2_2, dset3_1, dset3_2

# def my_plots(ax,data1,data2):
#     out = ax.bar(data1, data2)
#     return out
# fig, (ax1,ax2,ax3) = plt.subplots(3,1)
# dset1_1, dset1_2, dset2_1, dset2_2, dset3_1, dset3_2 = dset1_1, dset1_2, dset2_1, dset2_2, dset3_1, dset3_2
# print(data1)
# print(data2)
# my_plotter(ax1, data1, data2)#, {'marker': 's'})

 

