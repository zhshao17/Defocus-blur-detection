import os
import data
import fcn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision import models
from torchvision.models.vgg import VGG

import cv2
import numpy as np



# <---------------------------------------------->
# 下面开始训练网络

# 在训练网络前定义函数用于计算Acc 和 mIou
# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


# 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds,label_trues_float, label_preds_float, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - F-meature
      - MAE  每张图的MAE 共160*160
    """
    # hist每列为预测为该类的数量，每行为该类的真实数量，对角线为预测对的数量
    hist = np.zeros((n_class, n_class))
    mae = label_trues_float-label_preds_float
    mae = np.maximum(mae,-mae)
    n=np.size(label_preds,0)
    MAE = mae.sum()/n
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    F1 = hist[0][0]*2/(hist[0][0]*2+hist[0][1]+hist[1][0])
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    return acc, acc_cls, mean_iu, F1, MAE


from datetime import datetime

import torch.optim as optim
import matplotlib.pyplot as plt


def train(epo_num=1, show_vgg_params=False):
    # 实例化数据集
    bag = data.BagDataset(data.transform)
    bag2 = data.BagDataset2(data.transform)
    train_dataset = bag
    test_dataset = bag2
    # train_size = int(0.9 * len(bag))
    # test_size = len(bag) - train_size
    # train_dataset, test_dataset = random_split(bag, [train_size, test_size])
    # 利用DataLoader生成一个分batch获取数据的可迭代对象
    batch_size = 10
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=4)
    #开始训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = fcn.VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = fcn.FCN8s(pretrained_net=vgg_model, n_class=2)
    fcn_model = fcn_model.to(device)
    # 这里只有两类，采用二分类常用的损失函数BCE
    criterion = nn.BCELoss().to(device)
    # 随机梯度下降优化，学习率0.001，惯性分数0.7
    optimizer = optim.Adam(fcn_model.parameters(), lr=1e-3)

    # 记录训练过程相关指标
    all_train_iter_loss = []
    all_test_iter_loss = []
    test_Acc = []
    test_mIou = []
    test_F1 = []
    test_MAE = []
    # start timing
    prev_time = datetime.now()

    for epo in range(epo_num):

        # 训练
        train_loss = 0
        fcn_model.train()
        for index, (bag, bag_msk) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])
            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(bag)
            output = torch.sigmoid(output)  # output.shape is torch.Size([5, 2, 160, 160])
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
        fcn_model.eval()
        with torch.no_grad():
            for index, (bag, bag_msk) in enumerate(test_dataloader):
                bag = bag.to(device)
                bag_msk = bag_msk.to(device)
                optimizer.zero_grad()
                output = fcn_model(bag)
                output = torch.sigmoid(output)  # output.shape is torch.Size([5, 2, 160, 160])

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

        acc, acc_cls, mIou, F1, MAE = label_accuracy_score(bag_msk_np, output_np,bag_msk_np_float, output_np_float, 2)
        test_Acc.append(acc)
        test_mIou.append(mIou)
        test_F1.append(F1)
        test_MAE.append(MAE)

        print('Acc = %f, mIou = %f, F-meature = %f, MAE = %f' % (acc, mIou, F1, MAE))
        # 每5个epoch存储一次模型
        if np.mod(epo, 5) == 0:
            # 只存储模型参数
            torch.save(fcn_model.state_dict(), './fcn_model_{}.pth'.format(epo))
            print('saveing ./fcn_model_{}.pth'.format(epo))
    # 绘制训练过程数据

    cv2.imshow("GT", bag_msk.cpu().numpy()[0][0])
    for i in range(batch_size):
        cv2.imwrite('./DUT-DBD_Dataset/gt'+str(i)+'.jpg', bag_msk.cpu().numpy()[i][0] * 255)
    cv2.waitKey(0)
    cv2.imshow("out", output.cpu().numpy()[0][0])
    for i in range(batch_size):
        cv2.imwrite('./DUT-DBD_Dataset/out'+str(i)+'.jpg', output.cpu().numpy()[i][0]*255)
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


if __name__ == "__main__":
    train(epo_num=10, show_vgg_params=False)
