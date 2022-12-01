import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math
from d2l import torch as d2l

#@save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

#@save
class CatAttention(nn.Module):
    """连接注意力"""
    def __init__(self):
        super(CatAttention, self).__init__()
        # self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        # self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        #self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(0.1)
        #self.config = Config()
        # self.softmax = nn.Softmax(dim=1)
        # self.linear1 = nn.Linear(128+186, 128)
        self.w_v = nn.Linear(256, 1, bias=False)
        # self.softmax = nn.Softmax(dim=2)
        self.w2 = nn.Linear(128,128,bias= False)
        self.w_v_2 = nn.Linear(128, 1)

    def forward(self, queries, keys, values,valid_lens):
        # queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # # 使用广播方式进行求和
        # for j in range(batch_size):
        #     key, value = keys[j], values[j]
        #     for i in num_concept:
        #         querie = queries[j][i]
        #         key, value = key.unsqueeze(1), value.unsqueeze(1)
        #         key, value = key.repeat(1, num_concept, 1), value.repeat(1, num_concept, 1)
        #         print(key.shape, value.shape)
        #         print(querie.shape)
        #         features = torch.cat((queries, key), -1)
        #         features = torch.tanh(features)
        #         # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        #         # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        #         scores = self.w_v(features).squeeze(-1)
        #         self.attention_weights = masked_softmax(scores, valid_lens)
        #         # value的形状：(batch_size，“键－值”对的个数，值的维度)
        #         re = torch.bmm(self.dropout(self.attention_weights), value)
        # queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = torch.cat((queries, keys), -1)
       # print(features.shape)
        features = torch.tanh(features)
        #print(features.shape)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores1 = self.w_v(features)#.squeeze(-1)

      # print(queries.shape)
        features_2 = self.w2(queries)
        features_2 = torch.tanh(features_2)
        scores2 = nn.functional.softmax(self.w_v_2(features_2), dim = -1)

        scores1 = scores1.squeeze(2)
        scores2 = scores2.squeeze(2)

        # print("###########")
        # print(scores1.shape, scores2.shape)

        scores = nn.functional.softmax(0.5 * scores1.detach() + 0.5 * scores2.detach(), dim = -1)


        scores = scores.unsqueeze(1)
        #print(scores.shape)
        
        self.attention_weights = masked_softmax(scores,valid_lens)
        # print("&&&&&")
        # print(self.attention_weights.shape)
        #self.attention_weights = self.attention_weights#.permute(0,2,1)
        # print(values.shape)
        # print(self.attention_weights.shape)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)
        


class SlefAttention(nn.Module):
    #@save
    """缩放点积注意力"""
    def __init__(self):
        super(SlefAttention, self).__init__()
        self.dropout = nn.Dropout(0.1)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=64):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(2*d)
        self.attention_weights = masked_softmax(scores, valid_lens)
       # print(type(self.attention_weights),self.attention_weights.shape)
       # print("^^^^^^^^^^")
        return torch.bmm(self.dropout(self.attention_weights), values)