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

import shutil
import random
from Models import NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from ploting import input_images
import time

import dataloder
from torch.utils.data import DataLoader, random_split
import test


#######################################################
# Training loop
#######################################################
def train(epochs=1):
    # 记录训练过程相关指标
    all_train_iter_loss = []
    mae = []
    fmeasure = []
    recall = []
    precesion = []
    for i in range(epochs):

        train_loss = 0.0
        valid_loss = 0.0
        since = time.time()
        #######################################################
        # Training Data
        #######################################################
        model.train()
        k = 1
        for index, (x, y, idx) in enumerate(train_loader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])
            x = x.to(device)
            y = y.to(device)
            # If want to get the input images with their Augmentation - To check the data flowing in net
            # input_images(x, y, i, n_iter, k)

            # grid_img = torchvision.utils.make_grid(x)
            # writer1.add_image('images', grid_img, 0)

            # grid_lab = torchvision.utils.make_grid(y)

            opt.zero_grad()
            output = model(x)
            output = torch.sigmoid(output)
            output = torch.flip(output, dims=[1])
            # output = output[:, 1,0 , :, :]
            loss = criterion(output, y)
            loss.backward()
            train_loss += loss.item()
            all_train_iter_loss.append(loss.item())
            opt.step()
            k = 2
            # 每15个bacth，输出一次训练过程的数据
            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(i, index, len(train_loader), loss.item()))
                # break

        #######################################################
        # test Step
        #######################################################

        test.test(model, "model/result", test_loader)
        mae1, fmeasure1, recall1, precesion1 = test.eval1("model/result", '../DUT-DBD_Dataset/DUT500GT-Testing/', 1.5)
        print('epoch {}, mae={}, fmeature={}'.format(i, mae1, fmeasure1))
        print("----------------------------------------------------")
        mae.append(mae1)
        fmeasure.append(fmeasure1)
        recall.append(recall1)
        precesion.append(precesion1)
        # x = len(mae)
        # plt.figure(1)
        # plt.plot(x, mae)


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
    epochs = 100
    print('epoch = ' + str(epochs))

    random_seed = random.randint(1, 100)
    print('random_seed = ' + str(random_seed))

    n_iter = 1

    #######################################################
    # Setting up the model
    #######################################################

    model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]


    def model_unet(model_input, in_channel=3, out_channel=1):
        return model_input(in_channel, out_channel)


    model = model_unet(model_Inputs[0], in_channel=3, out_channel=2)  # 使用Unet

    model.to(device)

    torchsummary.summary(model, input_size=(3, 128, 128))

    train_data = dataloder.BagDataset(dataloder.transform, '../DUT-DBD_Dataset/DUT600S_Training/')
    testdata = dataloder.BagDataset(dataloder.transform, '../DUT-DBD_Dataset/DUT500S-Testing/')
    # 利用DataLoader生成一个分batch获取数据的可迭代对象
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=0)

    initial_lr = 0.001
    opt = torch.optim.Adam(model.parameters(), lr=initial_lr)  # try SGD
    # opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)
    criterion = nn.BCELoss().to(device)

    #######################################################
    # Creating a Folder for every data of the program
    #######################################################

    New_folder = './model'

    if os.path.exists(New_folder) and os.path.isdir(New_folder):
        shutil.rmtree(New_folder)

    try:
        os.mkdir(New_folder)
    except OSError:
        print("Creation of the main directory '%s' failed " % New_folder)
    else:
        print("Successfully created the main directory '%s' " % New_folder)

    #######################################################
    # Setting the folder of saving the predictions
    #######################################################
    read_pred = './model/pred'
    #######################################################
    # Checking if prediction folder exixts
    #######################################################

    if os.path.exists(read_pred) and os.path.isdir(read_pred):
        shutil.rmtree(read_pred)

    try:
        os.mkdir(read_pred)
    except OSError:
        print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
    else:
        print("Successfully created the prediction directory '%s' of dice loss" % read_pred)

    #######################################################
    # checking if the model exists and if true then delete
    #######################################################

    read_model_path = './model/Unet_D_' + str(epochs) + '_' + str(batch_size)

    if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
        shutil.rmtree(read_model_path)
        print('Model folder there, so deleted for newer one')

    try:
        os.mkdir(read_model_path)
    except OSError:
        print("Creation of the model directory '%s' failed" % read_model_path)
    else:
        print("Successfully created the model directory '%s' " % read_model_path)

    train(epochs=epochs)
