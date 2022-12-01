'''
Author: LiHang
Date: 2022-11-10 20:25:22
LastEditors: LiHang
LastEditTime: 2022-11-19 20:33:59
Description:
'''
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from jieba import cut
import requests
import copy
from d2l import torch as d2l
import os

from CST_CCS_attention import CatAttention, SlefAttention
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# input("input")


class Config():
    def __init__(self):
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        self.word_tokenizer = cut  #
        self.char_tokenizer = lambda x: [i for i in x]  # cut
        self.maxToken = 64  # 一句话最长有几个词
        self.data1_path = "dataset/1_nlpcc2013data-nlpcc2013_data/"
        self.data2_path = "dataset/2_NLPCC2014_sentiment-master/"
        self.data3_path = "dataset/3_nlpcc2017_news_headline_categorization-master/"
        self.data1_class_num = 8
        self.data2_class_num = 2
        self.data3_class_num = 18
        self.data4_class_num = 5
#   word    char    class_num
# 1 33020   4341    8
# 2 33367   4106    2
# 3 124579  5533    18
        self.data1_word_vocablen = 33020 
        self.data1_char_vocablen = 4341 
        self.data2_word_vocablen = 33367
        self.data2_char_vocablen = 4106
        self.data3_word_vocablen = 124579
        self.data3_char_vocablen = 5533
        self.window_sizes = [2, 3, 4]
        self.device = torch.device("cuda:0")
        self.epoch = 2
        self.api = "http://shuyantech.com/api/cnprobase/ment2ent?q="
        
        #需要指定
        self.drop = 0.1
        self.batch_size = 16
        self.num_consept = 64#一句话有几个概念

def get_concept(entity_set, num_entity):

    # # 获取实体概念

    # urls = [config.api + i for i in word_sentence]

    # # api返回的数据
    # res = []
    # for i in urls:
    #     response = requests.get(i)
    #     content = response.text
    # #     print(type(content))
    #     # print(content)
    #     content = content.replace("{\"status\": \"ok\", \"ret\": [", "")
    #     content = content.replace("]}", "")
    #     content = content.replace("\"", "")
    #     # print(content)
    #     res.append(content)
    # #     print(content)

    # concept_set = []
    # for i in res:
    #     re = i.split(", ")
    #     concept_set.append(re)
    # # for i in concept_set:
    #     # print(i)
    #word_sentence = word_sentence.unsqueeze(0)#
    concepts_set = []
    #num_concepts = 0

    for i in entity_set:
        # entity_now = entity_set[i]
        # # 获取i实体对应的所有概念
        # concept_now = entity_now#这里要修改成多个概念下，一个实体最多对应16个概念，这里简单将当前实体当做概念
        # concept_now.append(entity_now)
        concepts_set.append(i)
        #num_concepts += 1
        
    return concepts_set, num_entity  # concept_set和概念数（目前就是实体数） 现在用不了接口

def get_entity(sentence):
    config = Config()
    word_sentence = [i for i in config.word_tokenizer(sentence)]
    # 把分词作为实体集，获取概念集
    num_entity = len(word_sentence)
    #实体对齐
    if len(word_sentence) < config.num_consept:#最多64个实体
        for i in range(config.num_consept - len(word_sentence)):
            word_sentence.append(config.pad)
    elif len(word_sentence) > config.num_consept:
        word_sentence = word_sentence[:config.num_consept]
        num_entity = config.num_consept

    entity_set = word_sentence  # copy.deepcopy(word_sentence)
    # print("实体集：")
    # print(entity_set)
    #concept_set = get_concept(entity_set)
    # print("概念集：")
    # print(concept_set)
    # print("++++++++++++++++++++++++++")
    # for i in concept_set:
    #     print(i)
    # print(concept_set)
    return entity_set, num_entity#, concept_set


# word_sentence = ["马云", "周杰伦"]
# concept_set = get_concept(word_sentence)
# for i in concept_set:
#     print(i)
# print(concept_set)


class myData(Dataset):
    def __init__(self, data_path, word_vocab_path, char_vocab_path):  # 加载词表，计算数据集大小，所有句子转换为token表示

        self.config = Config()
        #self.temp = 0
        self.word_tokenizer = self.config.word_tokenizer
        self.char_tokenizer = self.config.char_tokenizer
        self.word_vocab = {}
        self.char_vocab = {}

        with open(word_vocab_path, 'r', encoding="utf-8") as f:
            i = 0
            for lines in f:
                # i += 1
                # if i == 100:
                #     break
                line = lines.split("\t")
                word, token = line[0], int(line[1])
                self.word_vocab.update({word: token})

        with open(char_vocab_path, 'r', encoding="utf-8") as f:
            i = 0
            for lines in f:
                # i += 1
                # if i == 100:
                #     break
                line = lines.split("\t")
                char, token = line[0], int(line[1])
                self.char_vocab.update({char: token})
        # 数据集大小
        self.set_size = 0
        self.labels = []
        self.sentences = []
        with open(data_path, "r", encoding="utf-8") as f:
            i = 0
            for lines in f:
                # i += 1
                # if i == 4:
                #     break  # 修改这里控制读取数据量
                self.set_size += 1

                line = lines.split("\t")
                label, sentence = int(line[0]), line[1]
                self.sentences.append(sentence)
                self.labels.append(label)
        # print("++++++++++")
        # print(self.vocab)
        # print(self.labels)
        # print(self.sentences)
        # print("--------")

    def __len__(self):
        return self.set_size

    def __getitem__(self, index):
        # print("^^^^^^{}".format(self.temp))
        # print(index)
        #self.temp += 1

        sentence = self.sentences[index].replace("\n", "")  # str
        
        #获取概念：1、切分实体 2、获取实体对应的概念
        #1、切分实体#2、获取实体对应的概念
        row = sentence
        entity_set, num_entity = get_entity(row)#列表[e1,e2]
        concepts_set, num_consept = get_concept(entity_set, num_entity)#二维列表[[c11,c12],[c21,c22]]
        # print("^^^^^^^^^^^^^^^^^^^^")
        # print(entity_set)
        # print(num_entity)
        # print(concepts_set)
        # print(num_consept)
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        #概念的token表示
        concepts_sets = concepts_set
        concepts_set = []
        for i in concepts_sets:
            concepts_set.append(self.word_vocab.get(i, self.config.unk))

        



        # 获取token表示

        label = self.labels[index]
        # 分词
        word_sentence = [i for i in self.word_tokenizer(sentence)]  # list
        char_sentence = [i for i in self.char_tokenizer(sentence)]  # list
        #print(sentence, len(sentence))

       
        # 长度处理
        word_length = len(word_sentence)
        char_length = len(char_sentence)
        #sentence_length = len(sentence)
        if len(word_sentence) < self.config.maxToken:
            for i in range(self.config.maxToken - len(word_sentence)):
                word_sentence.append(self.config.pad)
        elif len(word_sentence) > self.config.maxToken:
            word_length = self.config.maxToken
            word_sentence = word_sentence[:self.config.maxToken]

        if len(char_sentence) < self.config.maxToken:
            for i in range(self.config.maxToken - len(char_sentence)):
                char_sentence.append(self.config.pad)
        elif len(char_sentence) > self.config.maxToken:
            char_length = self.config.maxToken
            char_sentence = char_sentence[:self.config.maxToken]
        # 根据词表，转token表示
        # print(sentence)
        word_sentence_token = []
        char_sentence_token = []
        word_concept_token = []
        

        for word in word_sentence:
            # print(word)
            # print(self.vocab.get(word), type(self.vocab.get(word)))
            word_sentence_token.append(self.word_vocab.get(word, self.config.unk))

        for char in char_sentence:
            # print(word)
            # print(self.vocab.get(word), type(self.vocab.get(word)))
            char_sentence_token.append(self.char_vocab.get(char, self.config.unk))
        # 用张量表示
        sentence_word = torch.Tensor(word_sentence_token).long()
        sentence_char = torch.Tensor(char_sentence_token).long()
        label = torch.tensor(label).long()

        concepts_set = torch.Tensor(concepts_set).long()
        num_consept = torch.tensor(num_consept).long()
        # length = int(sentence_length)
        # print("*****")
        # print(sentence)
        # print(label)
        # print(length)
        # print("!!!!!!")
        # 必须返回两个数据，前一个是data，后一个是label
        return (sentence_word, word_length, sentence_char, char_length,concepts_set,num_consept), label


# config = Config()
# wordVocab_path = config.data1_path + "wordVocab.txt"
# charVocab_path = config.data1_path + "charVocab.txt"
# train_data_path = config.data1_path + "train.txt"
# ##test_data_path = config.data1_path + "test.txt"


# data = myData(train_data_path, wordVocab_path, charVocab_path)
# ## char_data = myData(train_data_path, charVocab_path)
# train_data = DataLoader(data, 2)  # , shuffle=True)
# ## train_char_data = DataLoader(char_data, 2)

# for j, (data, label) in enumerate(train_data):
#     # print("=====================")
#     # print(data, type(data))
#     # print("#####################")
#     print("tensor的读取：")
#     print(data[0], type(data[0]), data[0].shape)
#     # print(data[1], type(data[1]), data[1].shape)
#     # print(data[2], type(data[2]), data[2].shape)

#     # print(data[3], type(data[3]), data[3].shape)
#     print("最后的读取结果：")
#     # [('源海',), ('都',), ('学',), ('愤怒',), ('鸟',), ('的',), ('声音',), ('，',), ('好像',), ('好',), ('厉害',), ('…',)] <class 'list'>
#     print(data[4], type(data[4]))
#     print(data[5], type(data[5]))
#     print(data[4][1], type(data[4][1]))  # ('都',) <class 'tuple'>
#     print(data[5][1], type(data[5][1]))
#     print(data[4][1][0], type(data[4][1][0]))  # 都 <class 'str'>
#     print(data[5][1][0], type(data[5][1][0]))
#     # print(label, type(label), label.shape)

# queries = torch.normal(0, 1, (2, 1, 2))
# keys = torch.ones((2, 10, 2))
# values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
# valid_lens = torch.tensor([2, 6])

# attention = DotProductAttention(dropout=0.5)
# attention.eval()
# re = attention(queries, keys, values, valid_lens)

# d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
#                   xlabel='Keys', ylabel='Queries')


class STCKA(nn.Module):
    def __init__(self):
        super(STCKA, self).__init__()
        self.config = Config()

        self.char_embedding = nn.Embedding(self.config.data1_char_vocablen, 256,  # 不确定多少维度
                                           padding_idx=self.config.data1_char_vocablen - 1)
        self.word_embedding = nn.Embedding(self.config.data1_word_vocablen, 256,
                                           padding_idx=self.config.data1_word_vocablen - 1)
        # KB还没有实现 用word代替concept
        self.concept_embedding = nn.Embedding(self.config.data1_word_vocablen, 256,
                                            padding_idx=self.config.data1_word_vocablen - 1)
        #
        # self.convs = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv1d(512, 50, h),  # 因为卷积是在也就是第二个维度。
        #         nn.ReLU(),  # torch.Size([16, 50, 64-h+1])
        #         nn.MaxPool1d(64-h+1)  # ！！！不确定用得对不对。但是很多地方都是这么写的  torch.Size([16, 50, 1])
        #         # 句子长度，在一个卷积核上只要最大值（感觉是把当前kernel觉得整句话（但是论文说是一个单词的表示）在当前维度（总共50个维度）最重要的挑出来），50个卷积核。
        #         # 一个kernel上：现在只用一个数值表示这个句子的映射（原本是每个单词都要有，也就是64-h+1个数）
        #         # ！！！ 卷积之后一句话已经是用150个数来表示了。[16,50*3]
        #     )
        #     for h in self.config.window_sizes]
        # )
        # self.BiLSTM = nn.LSTM(150, 64, num_layers=2, bidirectional=True)  # u = 64


        # 卷积另一种实现#参考的代码是二维卷积，我改成一维卷积
        self.convs1 = nn.ModuleList([nn.Conv1d(512, 50, K) for K in self.config.window_sizes])
        

        self.BiLSTM = nn.LSTM(1, 64, num_layers=2, bidirectional=True, batch_first = True)
        self.fc = nn.Linear(256, self.config.data1_class_num)

        self.dropout = nn.Dropout(self.config.drop)

        self.pre_attention = nn.Linear(512, 128)

        self.nothing = nn.Linear(186, 128)

        self.self_att = SlefAttention()
        
        self.cst_ccs_attention = CatAttention()

    def forward(self, char_sentence, word_sentence, concept_sentence, word_lens,consept_set, num_consept):
        # print("~~~~~~~~~~~~~~~~~~~")
        # # torch.Size([16, 64]) <class 'torch.Tensor'>
        # # torch.Size([16]) <class 'torch.Tensor'>
        # # torch.Size([16, 32]) <class 'torch.Tensor'>
        # print(char_sentence.shape, type(char_sentence))
        # print(num_consept.shape, type(num_consept))
        # print(consept_set.shape, type(consept_set))
        # input("in")
        pq_char = self.char_embedding(char_sentence)
        q_word = self.word_embedding(word_sentence)
        p_concept = self.concept_embedding(consept_set)
        # print(pq_char.shape)
        # print(q_word.shape)
        # print(p_concept.shape)
        # input("input")
        # print(q_word.size(1))
        q_ = torch.cat((q_word, pq_char), 2)
        p_ = torch.cat((p_concept, pq_char), 2)
        p_temp = p_
        # q, p = q.permute(0, 2, 1), p.permute(0, 2, 1)  # torch.Size([16, 512, 64])# 因为卷积是在也就是第二个维度。
        # print("1q维度：")#torch.Size([16, 64, 512])
        # print(q.shape)
        # print(p.shape)
        #q = [conv(q) for conv in self.convs]
        # p = [conv(p) for conv in self.convs]
        #print(q[0].shape, q[1].shape, q[2].shape)
        # print(p[0].shape, p[1].shape, p[2].shape)  # torch.Size([16, 50, 1])


        #另一种卷积
        q = q_.permute(0, 2, 1)
        p = p_.permute(0, 2, 1)
        # print("2q维度：")#torch.Size([16, 64, 512])
        # print(p.shape)
        #q = [F.relu(conv(q)).squeeze(3) for conv in self.convs1] #[(N, Co, L), ...]*len(Ks)
        q = [F.relu(conv(q)) for conv in self.convs1]   #torch.Size([16, 50, 511]) torch.Size([16, 50, 510]) torch.Size([16, 50, 509])   
        p = [F.relu(conv(p)) for conv in self.convs1]
        # print("3q维度：")
        # print(p[0].shape, p[1].shape, q[2].shape)#torch.Size([16, 50, 511]) torch.Size([16, 50, 510]) torch.Size([16, 50, 509])
        #
        q = [i.permute(0, 2, 1) for i in q]
        p = [i.permute(0, 2, 1) for i in p]
        # print("3.1q维度：")
        # print(q[0].shape, q[1].shape, q[2].shape)
        q = [F.max_pool1d(i, i.size(2)) for i in q]  # [(N, Co), ...]*len(Ks)
        p = [F.max_pool1d(i, i.size(2)) for i in p]
        # print("4q维度：")
        # print(q[0].shape, q[1].shape, q[2].shape)#torch.Size([16, 50]) torch.Size([16, 50]) torch.Size([16, 50])
        q = [i.squeeze(2) for i in q]
        p = [i.squeeze(2) for i in p]
        # print("4.1q维度：")
        # print(q[0].shape, q[1].shape, q[2].shape)
        q = torch.cat(q, 1) #torch.Size([16, 150])
        p = torch.cat(p, 1)
        # print(q.shape)


        #没有提到这个
        #x = self.dropout(x)




        # new_q = torch.cat((q[0], q[1], q[2]), 1)
        # new_p = torch.cat((p[0], p[1], p[2]), 1)
        # print(new_q.shape, new_p.shape)  # torch.Size([16, 150, 1])
        # new_q, new_p = new_q.permute(0, 2, 1), new_p.permute(0, 2, 1)
        # # # print(new_q.shape, new_p.shape)
        # new_q = torch.squeeze(new_q, 1)
        # new_q = new_q.view(16, -1)
        #print(new_q.shape, new_p.shape)
        



        q = q.unsqueeze(2)
        p = p.unsqueeze(1)#torch.Size([64, 186, 1])
        # print("@@")
        # print(q.shape) 
        # print(p.shape)
        q, (hn, cn) = self.BiLSTM(q)
       
        # print("=====")
        # print(q.shape, type(q))  # torch.Size([16, 186, 128]) <class 'torch.Tensor'>  2u
        # print(p_temp.shape)
        # q = q[:,-1,:]
        # print("_______")
        # print(q.shape)

        #没有提到这个
        #q = self.dropout(q)


        #自注意力
        Q, K, V = q, q, q
        #self_attention = SlefAttention()
        q = self.self_att(Q, K, V, valid_lens = word_lens)

        
        #print("_______")#torch.Size([16, 64, 512]) torch.Size([16, 1, 128])
        #print(p_temp.shape, q.shape)
        # print(q.shape)#torch.Size([64, 128])
        p_temp = self.pre_attention(p_temp)
        q = q[:,-1,:].unsqueeze(1)
        q_temp = q
        #print(p_temp.shape, q.shape)#torch.Size([16, 64, 128]) torch.Size([16, 1, 128])
        q = q.repeat(1,64,1)
        #print(p_temp.shape, q.shape)
        Q, K, V = p_temp, q, q
        #cat注意力
       # print(Q.device)
        #cst_ccs_attention = CatAttention().to(self.config.device)
        #print(cst_attention.device)
        p_temp_ = self.cst_ccs_attention(Q, K, V,word_lens)#torch.Size([64, 186, 128])
        # print("******************")
        # print(q_temp.shape)#torch.Size([16, 1, 128])
        #没有提到这个
        # q = torch.cat((q[0], q[-1]), -1)
        

        # 注意力
        p = self.nothing(p)
        # print(p.shape)
        # print(p_temp_.shape)
        p= p + p_temp_
        re = torch.cat((q_temp, p), -1)
        #print(re.shape)

        # att_output = 
        re = re.squeeze(1)
        #print(re.shape)#
        # p = att_output + resdual
        re = self.fc(re)
        # print("$$")
        # print(re.shape)#torch.Size([64, 8])
        #print(q)
        return re



class STCKA_dense(nn.Module):
    def __init__(self):
        super(STCKA_dense, self).__init__()
        self.config = Config()

        self.char_embedding = nn.Embedding(self.config.data1_char_vocablen, 256,  # 不确定多少维度
                                           padding_idx=self.config.data1_char_vocablen - 1)
        self.word_embedding = nn.Embedding(self.config.data1_word_vocablen, 256,
                                           padding_idx=self.config.data1_word_vocablen - 1)
        # KB还没有实现 用word代替concept
        self.concept_embedding = nn.Embedding(self.config.data1_word_vocablen, 256,
                                            padding_idx=self.config.data1_word_vocablen - 1)
        #
        # self.convs = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv1d(512, 50, h),  # 因为卷积是在也就是第二个维度。
        #         nn.ReLU(),  # torch.Size([16, 50, 64-h+1])
        #         nn.MaxPool1d(64-h+1)  # ！！！不确定用得对不对。但是很多地方都是这么写的  torch.Size([16, 50, 1])
        #         # 句子长度，在一个卷积核上只要最大值（感觉是把当前kernel觉得整句话（但是论文说是一个单词的表示）在当前维度（总共50个维度）最重要的挑出来），50个卷积核。
        #         # 一个kernel上：现在只用一个数值表示这个句子的映射（原本是每个单词都要有，也就是64-h+1个数）
        #         # ！！！ 卷积之后一句话已经是用150个数来表示了。[16,50*3]
        #     )
        #     for h in self.config.window_sizes]
        # )
        # self.BiLSTM = nn.LSTM(150, 64, num_layers=2, bidirectional=True)  # u = 64


        # 卷积另一种实现#参考的代码是二维卷积，我改成一维卷积
        self.convs1 = nn.ModuleList([nn.Conv1d(512, 50, K) for K in self.config.window_sizes])
        

        self.BiLSTM = nn.LSTM(1, 64, num_layers=2, bidirectional=True, batch_first = True)
        self.fc = nn.Linear(256, self.config.data1_class_num)

        self.dropout = nn.Dropout(self.config.drop)

        self.pre_attention = nn.Linear(512, 128)

        self.nothing = nn.Linear(186, 128)

        self.dense_self_att = nn.Linear(128,128)
        self.dense_ccst_att = nn.Linear(64,1)


    def forward(self, char_sentence, word_sentence, concept_sentence, word_lens,consept_set, num_consept):
        # print("~~~~~~~~~~~~~~~~~~~")
        # # torch.Size([16, 64]) <class 'torch.Tensor'>
        # # torch.Size([16]) <class 'torch.Tensor'>
        # # torch.Size([16, 32]) <class 'torch.Tensor'>
        # print(char_sentence.shape, type(char_sentence))
        # print(num_consept.shape, type(num_consept))
        # print(consept_set.shape, type(consept_set))
        # input("in")
        pq_char = self.char_embedding(char_sentence)
        q_word = self.word_embedding(word_sentence)
        p_concept = self.concept_embedding(consept_set)
        # print(pq_char.shape)
        # print(q_word.shape)
        # print(p_concept.shape)
        # input("input")
        # print(q_word.size(1))
        q_ = torch.cat((q_word, pq_char), 2)
        p_ = torch.cat((p_concept, pq_char), 2)
        p_temp = p_
        # q, p = q.permute(0, 2, 1), p.permute(0, 2, 1)  # torch.Size([16, 512, 64])# 因为卷积是在也就是第二个维度。
        # print("1q维度：")#torch.Size([16, 64, 512])
        # print(q.shape)
        # print(p.shape)
        #q = [conv(q) for conv in self.convs]
        # p = [conv(p) for conv in self.convs]
        #print(q[0].shape, q[1].shape, q[2].shape)
        # print(p[0].shape, p[1].shape, p[2].shape)  # torch.Size([16, 50, 1])


        #另一种卷积
        q = q_.permute(0, 2, 1)
        p = p_.permute(0, 2, 1)
        # print("2q维度：")#torch.Size([16, 64, 512])
        # print(p.shape)
        #q = [F.relu(conv(q)).squeeze(3) for conv in self.convs1] #[(N, Co, L), ...]*len(Ks)
        q = [F.relu(conv(q)) for conv in self.convs1]   #torch.Size([16, 50, 511]) torch.Size([16, 50, 510]) torch.Size([16, 50, 509])   
        p = [F.relu(conv(p)) for conv in self.convs1]
        # print("3q维度：")
        # print(p[0].shape, p[1].shape, q[2].shape)#torch.Size([16, 50, 511]) torch.Size([16, 50, 510]) torch.Size([16, 50, 509])
        #
        q = [i.permute(0, 2, 1) for i in q]
        p = [i.permute(0, 2, 1) for i in p]
        # print("3.1q维度：")
        # print(q[0].shape, q[1].shape, q[2].shape)
        q = [F.max_pool1d(i, i.size(2)) for i in q]  # [(N, Co), ...]*len(Ks)
        p = [F.max_pool1d(i, i.size(2)) for i in p]
        # print("4q维度：")
        # print(q[0].shape, q[1].shape, q[2].shape)#torch.Size([16, 50]) torch.Size([16, 50]) torch.Size([16, 50])
        q = [i.squeeze(2) for i in q]
        p = [i.squeeze(2) for i in p]
        # print("4.1q维度：")
        # print(q[0].shape, q[1].shape, q[2].shape)
        q = torch.cat(q, 1) #torch.Size([16, 150])
        p = torch.cat(p, 1)
        # print(q.shape)


        #没有提到这个
        #x = self.dropout(x)




        # new_q = torch.cat((q[0], q[1], q[2]), 1)
        # new_p = torch.cat((p[0], p[1], p[2]), 1)
        # print(new_q.shape, new_p.shape)  # torch.Size([16, 150, 1])
        # new_q, new_p = new_q.permute(0, 2, 1), new_p.permute(0, 2, 1)
        # # # print(new_q.shape, new_p.shape)
        # new_q = torch.squeeze(new_q, 1)
        # new_q = new_q.view(16, -1)
        #print(new_q.shape, new_p.shape)
        



        q = q.unsqueeze(2)
        p = p.unsqueeze(1)#torch.Size([64, 186, 1])
        # print("@@")
        # print(q.shape) 
        # print(p.shape)
        q, (hn, cn) = self.BiLSTM(q)
       
        # print("=====")
        # print(q.shape, type(q))  # torch.Size([16, 186, 128]) <class 'torch.Tensor'>  2u
        # print(p_temp.shape)
        # q = q[:,-1,:]
        # print("_______")
        # print(q.shape)

        #没有提到这个
        #q = self.dropout(q)


        #自注意力
        Q, K, V = q, q, q
        self_attention = SlefAttention()
        # print("**********")
        # print(q.shape)
        #q = self_attention(Q, K, V, valid_lens = word_lens)
        q = self.dense_self_att(q)
        # print(q.shape)
        
        #print("_______")#torch.Size([16, 64, 512]) torch.Size([16, 1, 128])
        #print(p_temp.shape, q.shape)
        # print(q.shape)#torch.Size([64, 128])
        p_temp = self.pre_attention(p_temp)
        q = q[:,-1,:].unsqueeze(1)
        q_temp = q
        #print(p_temp.shape, q.shape)#torch.Size([16, 64, 128]) torch.Size([16, 1, 128])
        q = q.repeat(1,64,1)
        #print(p_temp.shape, q.shape)
        Q, K, V = p_temp, q, q
        #cat注意力
       # print(Q.device)
        cst_ccs_attention = CatAttention().to(self.config.device)
        #print(cst_attention.device)
        # print("###########")
        # print(p_temp.shape)
        #p_temp_ = cst_ccs_attention(Q, K, V,word_lens)#torch.Size([64, 186, 128])
        p_temp = p_temp.permute(0,2,1)
        p_temp_ = self.dense_ccst_att(p_temp)
        p_temp_ = p_temp_.permute(0,2,1)
        # print(p_temp_.shape)
        # print("******************")
        # print(q_temp.shape)#torch.Size([16, 1, 128])
        #没有提到这个
        # q = torch.cat((q[0], q[-1]), -1)
        

        # 注意力
        p = self.nothing(p)
        # print(p.shape)
        # print(p_temp_.shape)
        p= p + p_temp_
        re = torch.cat((q_temp, p), -1)
        #print(re.shape)

        # att_output = 
        re = re.squeeze(1)
        #print(re.shape)#
        # p = att_output + resdual
        re = self.fc(re)
        # print("$$")
        # print(re.shape)#torch.Size([64, 8])
        #print(q)
        return re



        # print(q.shape, type(q))  # torch.Size([16, 64, 8]) <class 'torch.Tensor'>

# torch.Size([16, 512, 64])
# torch.Size([16, 512, 64])
# torch.Size([16, 50, 63]) torch.Size([16, 50, 62]) torch.Size([16, 50, 61])
# torch.Size([16, 50, 63]) torch.Size([16, 50, 62]) torch.Size([16, 50, 61])
