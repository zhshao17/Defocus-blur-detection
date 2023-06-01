import os
import dataloder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision import models
import test
import click
import cv2
import numpy as np
from deeplabv3_plus import DeepLabV3Plus
from datetime import datetime
import logging
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def build_network(snapshot, backend):
    epoch = 0
    net = DeepLabV3Plus(nclass=2,backbone='xception',pretrained_base=True)
    # net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    if torch.cuda.is_available():
        net = net.cuda(0)
    else:
        net.to(torch.device('cpu'))

    return net, epoch



# <---------------------------------------------->
# 下面开始训练网络

def train(epo_num=1):
    # 实例化数据集
    os.makedirs('./models', exist_ok=True)
    save_epoch = 1


    train_data = dataloder.BagDataset(dataloder.transform,'./DUT-DBD_Dataset/DUT600S_Training/')
    test_data = dataloder.BagDataset(dataloder.transform,'./DUT-DBD_Dataset/DUT500S-Testing/')
    # 利用DataLoader生成一个分batch获取数据的可迭代对象
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    #开始训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net, _ = build_network(None, 'resnet18')
    # 这里只有两类，采用二分类常用的损失函数BCE
    criterion = nn.BCELoss().to(device)
    # 随机梯度下降优化，学习率0.001，惯性分数0.7
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # 记录训练过程相关指标
    all_train_iter_loss = []
    all_test_iter_loss = []
    # start timing
    prev_time = datetime.now()

    for epo in range(epo_num):

        # 训练
        train_loss = 0
        net.train()
        for index, (bag, bag_msk,ind) in enumerate(train_dataloader):

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = net(bag)
            output = torch.sigmoid(output[0])  # output.shape is torch.Size([5, 2, 160, 160])
            loss = criterion(output, bag_msk)
            loss.backward()  # 需要计算导数，则调用backward
            iter_loss = loss.item()  # .item()返回一个具体的值，一般用于loss和acc
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1)
            bag_msk_np = bag_msk.cpu().detach().numpy().copy()
            bag_msk_np = np.argmin(bag_msk_np, axis=1)

            # 每15个bacth，输出一次训练过程的数据
            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                #break

        # 验证
        test_loss = 0
        net.eval()
        with torch.no_grad():
            for index, (bag, bag_msk, ind) in enumerate(test_dataloader):
                bag = bag.to(device)
                bag_msk = bag_msk.to(device)
                optimizer.zero_grad()
                output = net(bag)
                output = torch.sigmoid(output[0])  # output.shape is torch.Size([5, 2, 160, 160])

                loss = criterion(output, bag_msk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np_float = output.cpu().detach().numpy().copy()
                output_np = np.argmin(output_np_float, axis=1)

                bag_msk_np_float = bag_msk.cpu().detach().numpy().copy()
                bag_msk_np = np.argmin(bag_msk_np_float, axis=1)
                #break


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('<---------------------------------------------------->')
        print('epoch: %f' % epo)
        print('epoch train loss = %f, epoch test loss = %f, %s'
              % (train_loss / len(train_dataloader), test_loss / len(test_dataloader), time_str))
        print('<---------------------------------------------------->')

        # 每5个epoch存储一次模型
        if np.mod(epo, 1) == 0:
            # 只存储模型参数
            torch.save(net.state_dict(), './net_model_{}.pth'.format(epo))
            stict = './net_model_{}.pth'.format(epo)
            print(stict)
            print('saveing ./net_model_{}.pth'.format(epo))

        if np.mod(epo, 1) == 0:

            test.test(stict,net, "./models_result", test_dataloader)
            mae1, fmeasure1, _, _ = test.eval1("./models_result", './DUT-DBD_Dataset/DUT500GT-Testing', 0.5)
            print('====================================================================================================================')
            print("mae:", mae1, "fmeasure:", fmeasure1)
            print('=====================================================================================================================')



if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn((200, 300, 200, 20), device=device)
    x = torch.randn((200, 300, 200, 20), device=device)
    x = torch.randn((200, 300, 200, 20), device=device)
    x = torch.randn((200, 300, 200, 20), device=device)
    x = torch.randn((200, 300, 200, 20), device=device)
    x = torch.randn((200, 300, 200, 20), device=device)
    x = torch.randn((200, 300, 200, 20), device=device)
    x = torch.randn((200, 300, 200, 20), device=device)
    x = torch.randn((200, 300, 200, 20), device=device)
    x = torch.randn((200, 300, 200, 20), device=device)
    x = torch.randn((200, 300, 200, 20), device=device)
    x = torch.randn((200, 300, 200, 20), device=device)
    x = 1

    train(epo_num=2)
