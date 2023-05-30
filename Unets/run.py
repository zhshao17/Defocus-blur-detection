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

# from ploting import VisdomLinePlotter
# from visdom import Visdom


import data
from torch.utils.data import DataLoader, random_split

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
epoch = 15
print('epoch = ' + str(epoch))

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

shuffle = True
valid_loss_min = np.Inf
num_workers = 0  # window不能多进程处理数据
epoch_valid = epoch - 2
n_iter = 1
i_valid = 0

#######################################################
# Setting up the model
#######################################################

model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]


def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test


model_test = model_unet(model_Inputs[0], in_channel=3, out_channel=2)  # 使用Unet

model_test.to(device)

#######################################################
# Getting the Summary of Model
#######################################################

torchsummary.summary(model_test, input_size=(3, 128, 128))

#######################################################
# Dataset of Images and Labels
#######################################################

bag = data.BagDataset(data.transform)

train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])
# 利用DataLoader生成一个分batch获取数据的可迭代对象
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#######################################################
# Using Adam as Optimizer
#######################################################

initial_lr = 0.001
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr)  # try SGD
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

read_model_path = './model/Unet_D_' + str(epoch) + '_' + str(batch_size)

if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)


#####################################################
# 在训练网络前定义函数用于计算Acc 和 mIou
####################################################
# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


# 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    return acc, acc_cls, mean_iu


#######################################################
# Training loop
#######################################################
# 记录训练过程相关指标
all_train_iter_loss = []
all_test_iter_loss = []
test_Acc = []
test_mIou = []
for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()

    #######################################################
    # Training Data
    #######################################################

    model_test.train()
    k = 1
    for index, (bag, bag_msk) in enumerate(train_loader):
        # bag.shape is torch.Size([4, 3, 160, 160])
        # bag_msk.shape is torch.Size([4, 2, 160, 160])

        bag = bag.to(device)
        bag_msk = bag_msk.to(device)

        # If want to get the input images with their Augmentation - To check the data flowing in net
        input_images(bag, bag_msk, i, n_iter, k)

        # grid_img = torchvision.utils.make_grid(x)
        # writer1.add_image('images', grid_img, 0)

        # grid_lab = torchvision.utils.make_grid(y)

        opt.zero_grad()

        output = model_test(bag)
        output = torch.sigmoid(output)
        loss = criterion(output, bag_msk)
        loss.backward()
        train_loss += loss.item()
        all_train_iter_loss.append(loss.item())
        opt.step()
        k = 2

        output_np = output.cpu().detach().numpy().copy()
        output_np = np.argmin(output_np, axis=1)
        bag_msk_np = bag_msk.cpu().detach().numpy().copy()
        bag_msk_np = np.argmin(bag_msk_np, axis=1)

        # 每15个bacth，输出一次训练过程的数据
        if np.mod(index, 15) == 0:
            print('epoch {}, {}/{},train loss is {}'.format(i, index, len(train_loader), loss.item()))
            # break

    #######################################################
    # test Step
    #######################################################

    test_loss = 0
    model_test.eval()
    torch.no_grad()  # to increase the validation process uses less memory

    for index, (bag, bag_msk) in enumerate(test_loader):
        bag = bag.to(device)
        bag_msk = bag_msk.to(device)

        output = model_test(bag)
        output = torch.sigmoid(output)
        loss = criterion(output, bag_msk)

        test_loss += loss.item()
        all_test_iter_loss.append(loss.item())
        output_np = output.cpu().detach().numpy().copy()
        output_np = np.argmin(output_np, axis=1)
        bag_msk_np = bag_msk.cpu().detach().numpy().copy()
        bag_msk_np = np.argmin(bag_msk_np, axis=1)

    #######################################################
    # To write in Tensorboard
    #######################################################

    train_loss = train_loss / train_size
    valid_loss = valid_loss / test_size

    if (i + 1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss))
    acc, acc_cls, mIou = label_accuracy_score(bag_msk_np, output_np, 2)
    test_Acc.append(acc)
    test_mIou.append(mIou)

    print('Acc = %f, mIou = %f' % (acc, mIou))
    #######################################################
    # Early Stopping
    #######################################################

    if valid_loss <= valid_loss_min and epoch_valid >= i:  # and i_valid <= 2:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model_test.state_dict(), './model/Unet_D_' +
                   str(epoch) + '_' + str(batch_size) + '/Unet_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth')
        # print(accuracy)
        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print(i_valid)
            i_valid = i_valid + 1
        valid_loss_min = valid_loss


# 绘制训练过程数据
    cv2.imshow("GT", bag_msk.cpu().numpy()[0][0])
    cv2.waitKey(0)
    cv2.imshow("out", output.cpu().numpy()[0][0])
    cv2.waitKey(0)
    plt.figure()
    plt.subplot(221)
    plt.title('train_loss')
    plt.plot(all_train_iter_loss)
    plt.xlabel('batch')
    plt.subplot(222)
    plt.title('test_loss')
    plt.plot(all_test_iter_loss)
    plt.xlabel('batch')
    plt.subplot(223)
    plt.title('test_Acc')
    plt.plot(test_Acc)
    plt.xlabel('epoch')
    plt.subplot(224)
    plt.title('test_mIou')
    plt.plot(test_mIou)
    plt.xlabel('epoch')
    plt.show()