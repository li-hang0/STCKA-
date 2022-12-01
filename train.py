'''
Author: LiHang
Date: 2022-11-10 20:28:12
LastEditors: LiHang
LastEditTime: 2022-11-19 22:34:26
Description: 
'''
import time

from sklearn import metrics
from config_data_model import STCKA, Config, STCKA_dense, myData
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_gpu():
    config = Config()
    model = STCKA()
    #model = STCKA_dense()
    model = model.to(config.device)
    wordVocab_path = config.data1_path + "wordVocab.txt"
    charVocab_path = config.data1_path + "charVocab.txt"
    train_data_path = config.data1_path + "train.txt"
    test_data_path = config.data1_path + "test.txt"

    data = myData(train_data_path, wordVocab_path, charVocab_path)
    # train_word_data = DataLoader(word_data, 1)  # , shuffle=True)
    data_test = myData(test_data_path, wordVocab_path, charVocab_path)
    train_data = DataLoader(data, config.batch_size)
    test_data = DataLoader(data_test, config.batch_size)
    crossentropyloss = nn.CrossEntropyLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    average_loss = []
    average_f1 = []
    average_acc = []
    model_time = []
    zero_grad_time =[]
    loss_time = []
    backward_time =[]
    step_time = []
    for i in tqdm(range(config.epoch)):
        train_loss_all = []
        train_acc_all = []
        train_loss = 0
        train_num = 0
        train_f1_all = []

        model_timei = []
        zero_grad_timei =[]
        loss_timei = []
        backward_timei =[]
        step_timei = []

        model.train()
        for j, (data, label) in enumerate(train_data):
            data[0], data[2], word_lens, label, consept_set, num_consept = \
                data[0].to(config.device), data[2].to(config.device), \
                data[1].to(config.device), label.to(config.device),\
                data[4].to(config.device), data[5].to(config.device)

            # print("~~~~~~~~~~~~~~~")torch.Size([16, 64]) torch.Size([16, 64])
            # print(data[0].shape, consept_set.shape)
            time0 = time.time()
            out = model(data[2], data[0], data[0], word_lens, consept_set, num_consept)  # 概念参数暂时没有修改
            time1 = time.time()
            # print("标签维度")
            # out = torch.softmax(out, dim=1)
            # predict = torch.argmax(out,1)
            # print("预测标签", predict)
            # print(label.shape, type(label))
            
            optimizer.zero_grad()
            #print("*******")
            time2 = time.time()
            loss = crossentropyloss(out, label)
            time3 = time.time()
            loss.backward()
            time4 = time.time()
            #print("&&&&&")
            optimizer.step()
            time5 = time.time()
            #print(type(loss))
            loss = loss.detach()
            loss = loss.cpu().numpy()
            train_loss_all.append(loss)
            predict = out.detach()
            predict = predict.cpu().numpy()
            label = label.cpu().numpy()
            #print()
            predict = np.argmax(predict, axis=1)
            # print(out)
            # print(predict)
            #print(type(label), type(predict))
            f1 = f1_score(label, predict, average='weighted')
            train_f1_all.append(f1)
            acc = metrics.accuracy_score(label, predict)
            train_acc_all.append(acc)
            #各步骤时间
            # model_timei.append(time1 - time0)
            # zero_grad_timei.append(time2 - time1)
            # loss_timei.append(time3 - time2)
            # backward_timei.append(time4 - time3)
            # step_timei.append(time5 - time4)
            if j == 2:
                break
        #验证、测试
        # test_f1_all = []
        # test_loss_all = []
        # test_acc_all = []
        # model.eval()
        # with torch.no_grad():
        #     for j, (data, label) in enumerate(test_data):
        #         data[0], data[2], word_lens, label, consept_set, num_consept = \
        #             data[0].to(config.device), data[2].to(config.device), \
        #             data[1].to(config.device), label.to(config.device),\
        #             data[4].to(config.device), data[5].to(config.device)

        #         # print("~~~~~~~~~~~~~~~")torch.Size([16, 64]) torch.Size([16, 64])
        #         # print(data[0].shape, consept_set.shape)
        #         time0 = time.time()
        #         out = model(data[2], data[0], data[0], word_lens, consept_set, num_consept)  # 概念参数暂时没有修改
        #         time1 = time.time()
        #         # print("标签维度")
        #         # out = torch.softmax(out, dim=1)
        #         # predict = torch.argmax(out,1)
        #         # print("预测标签", predict)
        #         # print(label.shape, type(label))
                
        #         #optimizer.zero_grad()
        #         #print("*******")
        #         # time2 = time.time()
        #         loss = crossentropyloss(out, label)
        #         # time3 = time.time()
        #         # loss.backward()
        #         # time4 = time.time()
        #         # #print("&&&&&")
        #         # optimizer.step()
        #         # time5 = time.time()
        #         # #print(type(loss))
        #         # loss = loss.detach()
        #         loss = loss.cpu().numpy()
        #         test_loss_all.append(loss)
        #         predict = out.detach()
        #         predict = predict.cpu().numpy()
        #         label = label.cpu().numpy()
        #         #print()
        #         predict = np.argmax(predict, axis=1)
        #         # print(out)
        #         # print(predict)
        #         #print(type(label), type(predict))
        #         f1 = f1_score(label, predict, average='weighted')
        #         test_f1_all.append(f1)
        #         acc = metrics.accuracy_score(label, predict)
        #         test_acc_all.append(acc)
                #各步骤时间
                # model_timei.append(time1 - time0)
                # zero_grad_timei.append(time2 - time1)
                # loss_timei.append(time3 - time2)
                # backward_timei.append(time4 - time3)
                # step_timei.append(time5 - time4)
                # if j == 2:
                #     break


        #计算总值
        sum = 0
        for i in train_loss_all:
            sum += i
        average_loss.append(sum / len(train_loss_all))

        sum = 0
        for i in train_f1_all:
            sum += i
        average_f1.append(sum / len(train_f1_all))

        sum = 0
        for i in train_acc_all:
            sum += i
        average_acc.append(sum / len(train_acc_all))

        # sum = 0
        # for i in model_timei:
        #     sum += i
        # model_time.append(sum / len(model_timei))

        # sum = 0
        # for i in zero_grad_timei:
        #     sum += i
        # zero_grad_time.append(sum / len(zero_grad_timei))

        # sum = 0
        # for i in loss_timei:
        #     sum += i
        # loss_time.append(sum / len(loss_timei))

        # sum = 0
        # for i in backward_timei:
        #     sum += i
        # backward_time.append(sum / len(backward_timei))

        # sum = 0
        # for i in step_timei:
        #     sum += i
        # step_time.append(sum / len(step_timei))

        #直接绘图
        # xaxis = [d for d in range(len(train_loss_all))]
        # yaxis = train_loss_all
        # fig, ax = plt.subplots()
        # ax.plot(xaxis, yaxis) 
        # ax.set_xlabel("batch_num") 
        # ax.set_ylabel("loss") 
        # fig.tight_layout()
        # # saving the file.Make sure you  
        # # use savefig() before show(). 
        # plt.savefig("epoch{}.png".format(i)) 
        
        # plt.show()



    # 绘图
    xaxis = [d for d in range(len(average_loss))]
    yaxis = average_loss
    fig, ax = plt.subplots()
    ax.plot(xaxis, yaxis) 
    ax.set_xlabel("epoch") 
    ax.set_ylabel("averageloss") 
    fig.tight_layout()
    # saving the file.Make sure you  
    # use savefig() before show(). 
    plt.savefig("average_loss.png") 
    plt.show()
    #绘图
    xaxis = [d for d in range(len(average_f1))]
    yaxis = average_f1
    fig, ax = plt.subplots()
    ax.plot(xaxis, yaxis) 
    ax.set_xlabel("epoch") 
    ax.set_ylabel("averagef1") 
    fig.tight_layout()
    # saving the file.Make sure you  
    # use savefig() before show(). 
    plt.savefig("average_f1.png") 
    plt.show()
    #绘图
    xaxis = [d for d in range(len(average_acc))]
    yaxis = average_acc
    fig, ax = plt.subplots()
    ax.plot(xaxis, yaxis) 
    ax.set_xlabel("epoch") 
    ax.set_ylabel("averageacc") 
    fig.tight_layout()
    # saving the file.Make sure you  
    # use savefig() before show(). 
    plt.savefig("average_acc.png") 
    plt.show()


    #绘图
    # xaxis = [d for d in range(len(model_time))]
    # yaxis = model_time
    # fig, ax = plt.subplots()
    # ax.plot(xaxis, yaxis) 
    # ax.set_xlabel("epoch") 
    # ax.set_ylabel("model_time") 
    # fig.tight_layout()
    # # saving the file.Make sure you  
    # # use savefig() before show(). 
    # plt.savefig("model_time.png") 
    # plt.show()

    # xaxis = [d for d in range(len(zero_grad_time))]
    # yaxis = zero_grad_time
    # fig, ax = plt.subplots()
    # ax.plot(xaxis, yaxis) 
    # ax.set_xlabel("epoch") 
    # ax.set_ylabel("zero_grad_time") 
    # fig.tight_layout()
    # # saving the file.Make sure you  
    # # use savefig() before show(). 
    # plt.savefig("zero_grad_time.png") 
    # plt.show()

    # xaxis = [d for d in range(len(loss_time))]
    # yaxis = loss_time
    # fig, ax = plt.subplots()
    # ax.plot(xaxis, yaxis) 
    # ax.set_xlabel("epoch") 
    # ax.set_ylabel("loss_time") 
    # fig.tight_layout()
    # # saving the file.Make sure you  
    # # use savefig() before show(). 
    # plt.savefig("loss_time.png") 
    # plt.show()

    # xaxis = [d for d in range(len(backward_time))]
    # yaxis = backward_time
    # fig, ax = plt.subplots()
    # ax.plot(xaxis, yaxis) 
    # ax.set_xlabel("epoch") 
    # ax.set_ylabel("backward_time") 
    # fig.tight_layout()
    # # saving the file.Make sure you  
    # # use savefig() before show(). 
    # plt.savefig("backward_time.png") 
    # plt.show()

    # xaxis = [d for d in range(len(step_time))]
    # yaxis = step_time
    # fig, ax = plt.subplots()
    # ax.plot(xaxis, yaxis) 
    # ax.set_xlabel("epoch") 
    # ax.set_ylabel("step_time") 
    # fig.tight_layout()
    # # saving the file.Make sure you  
    # # use savefig() before show(). 
    # plt.savefig("step_time.png") 
    # plt.show()




# start_time = time.time()
# train_cpu()
# end_time = time.time()
# print("cpu耗时：{}".format(end_time - start_time))


# print(torch.__version__)

torch.backends.cudnn.benchmark=True
start_time = time.time()
train_gpu()
end_time = time.time()
print("gpu耗时：{}".format(end_time - start_time))
# for j, (data, label) in enumerate(train_data):
#     print("=====================")
#     print(data, type(data))
#     print(data[0], type(data[0]))
#     print(data[1], type(data[1]))
#     print(data[2], type(data[2]))
#     print(data[3], type(data[3]))
#     print(label, type(label))
