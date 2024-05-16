from __future__ import print_function, division
import os
import numpy as np
import torch.nn as nn
# import SimpleITK as sitk
import cv2
import torch.nn
import matplotlib.pyplot as plt
import torchsummary
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import gc
import shutil
import random
import time

from model import LUHAONet
import dataloder
from torch.utils.data import DataLoader, random_split
import test
import torch

from pretrain import pretrain
from mask import *


#######################################################
# Training loop
#######################################################
def train(epoch, epochs):
    # 记录训练过程相关指标
    all_train_iter_loss = []
    mae = []
    fmeasure = []
    recall = []
    precesion = []
    self_criterion = nn.L1Loss().cuda(0)
    while epoch <= epochs:

        train_loss = 0.0
        valid_loss = 0.0
        since = time.time()
        #######################################################
        # Training Data
        #######################################################
        model.train()
        k = 1
        for index, (x, y, idx) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            x_mask = creat_mask(0.1, 7, 7, x, batch_size, 3, 224, 224)
            opt.zero_grad()
            # output = model(x)
            # output = torch.sigmoid(output)
            # output = torch.flip(output, dims=[1])
            # output = output[:, 1,0 , :, :]
            outputsL2H0, outputsL2H1, outputsL2H2, outputsL2H3, outputsL2H4, \
            outputsH2L0, outputsH2L1, outputsH2L2, outputsH2L3, outputsH2L4, \
            outputsFusion = model(x)

            _, _, _, _, _, \
            _, _, _, _, _, \
            outputsFusion_mask = model(x_mask)
            y = y[:, 0, :, :]  # batch * label * H * W
            # y = y[:, 1, :, :]
            y = y.reshape(y.shape[0], 1, y.shape[1], y.shape[2])
            lossL2H0 = criterion(outputsL2H0, y)
            lossL2H1 = criterion(outputsL2H1, y)
            lossL2H2 = criterion(outputsL2H2, y)
            lossL2H3 = criterion(outputsL2H3, y)
            lossL2H4 = criterion(outputsL2H4, y)

            lossH2L0 = criterion(outputsH2L0, y)
            lossH2L1 = criterion(outputsH2L1, y)
            lossH2L2 = criterion(outputsH2L2, y)
            lossH2L3 = criterion(outputsH2L3, y)
            lossH2L4 = criterion(outputsH2L4, y)

            lossFusion = criterion(outputsFusion, y)

            lossMask = criterion(outputsFusion_mask, y)
            lossSelf = self_criterion(outputsFusion, outputsFusion_mask)

            loss = lossFusion + lossL2H0 + lossL2H1 + lossL2H2 + lossL2H3 + lossL2H4 + lossH2L0 + lossH2L1 + lossH2L2 + lossH2L3 + lossH2L4 + lossMask + 0.2 * lossSelf


            loss.backward()
            train_loss += loss.item()
            all_train_iter_loss.append(loss.item())
            opt.step()
            # 每15个bacth，输出一次训练过程的数据
            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epoch, index, len(train_loader), loss.item()))
                # break

        gc.collect()
        torch.cuda.empty_cache()
        #######################################################
        # test Step
        #######################################################
        if epoch % 5 == 0:
            stict = "model_mask" + str(epoch) + ".pth"
            torch.save(model.state_dict(), stict)
            test.test(stict, model, "model_mask/result", test_loader)
            mae1, fmeasure1, recall1, precesion1 = test.eval1("model_mask/result",
                                                              '../data/DUT-DBD_Dataset/DUT500GT-Testing/', 1.5)
            print('epoch {}, mae={}, fmeature={}'.format(epoch, mae1, fmeasure1))
            mae.append(mae1)
            fmeasure.append(fmeasure1)
            recall.append(recall1)
            precesion.append(precesion1)

        print("----------------------------------------------------")

        epoch = epoch + 1


if __name__ == '__main__':
    #######################################################
    # Checking if GPU is used
    #######################################################

    if not torch.cuda.is_available():
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################################################
    # Setting the basic paramters of the model
    #######################################################

    batch_size = 4
    print('batch_size = ' + str(batch_size))
    epochs = 50
    print('epoch = ' + str(epochs))
    epochs_pretrained = 20
    print('epochs_pretrained = ' + str(epochs_pretrained))

    random_seed = random.randint(1, 100)
    print('random_seed = ' + str(random_seed))

    train_data = dataloder.BagDataset(dataloder.transform, '../data/DUT-DBD_Dataset/DUT600S_Training/')
    testdata = dataloder.BagDataset(dataloder.transform, '../data/DUT-DBD_Dataset/DUT500S-Testing/')
    # 利用DataLoader生成一个分batch获取数据的可迭代对象
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=0)

    #######################################################
    # Setting up the model
    #######################################################

    epoch = 0

    model = LUHAONet()
    # model.load_state_dict(torch.load("model" + str(epoch) + ".pth"))
    model.load_state_dict(torch.load("model50.pth"))
    model = model.cuda()
    model.train()
    # torchsummary.summary(model, input_size=(3, 224, 224))

    initial_lr = 0.001
    opt = torch.optim.Adam(model.parameters(), lr=initial_lr)  # try SGD
    # opt = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99)
    criterion = nn.BCELoss().to(device)

    # pretrain(model, epochs=epochs_pretrained, batch_size=batch_size)
    train(epoch=epoch, epochs=epochs)
