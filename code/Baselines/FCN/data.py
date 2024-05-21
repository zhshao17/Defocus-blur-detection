import os
import data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision import models
from torchvision.models.vgg import VGG

import cv2
import numpy as np


# 将标记图（每个像素值代该位置像素点的类别）转换为onehot编码
def onehot(data, n):
    buf = np.zeros(data.shape + (n,))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk - 1] = 1
    return buf


# 利用torchvision提供的transform，定义原始图片的预处理步骤（转换为tensor和标准化处理）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# 利用torch提供的Dataset类，定义我们自己的数据集
class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('./DUT-DBD_Dataset/DUT600S_Training'))

    def __getitem__(self, idx):
        img_name = os.listdir('./DUT-DBD_Dataset/DUT600S_Training')[idx]
        #print(img_name)
        imgA = cv2.imread('./DUT-DBD_Dataset/DUT600S_Training/' + img_name)

        img_name = img_name.replace('jpg', 'bmp')
        #print(img_name)
        imgA = cv2.resize(imgA, (160, 160))

        imgB = cv2.imread('./DUT-DBD_Dataset./DUT600GT_Training/' + img_name, 0)
        imgB = cv2.resize(imgB, (160, 160))
        imgB = imgB / 255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2, 0, 1)
        imgB = torch.FloatTensor(imgB)
        # print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)

        return imgA, imgB


# 实例化数据集
bag = BagDataset(transform)

train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

# # 利用DataLoader生成一个分batch获取数据的可迭代对象
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
# test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)
