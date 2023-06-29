import os
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

    def __init__(self, transform=None, file_name=None):
        self.transform = transform
        self.file_name = file_name
        #self.file_name_1 = file_name.replace('image', 'gt')

    def __len__(self):
        # return len(os.listdir('./DUT-DBD_Dataset/DUT600S_Training'))
        return len(os.listdir(self.file_name))

    def __getitem__(self, idx):
        # img_name = os.listdir('./DUT-DBD_Dataset/DUT600S_Training')[idx]
        img_name = os.listdir(self.file_name)[idx]
        # print(img_name)
        # imgA = cv2.imread('./DUT-DBD_Dataset/DUT600S_Training/' + img_name)
        imgA = cv2.imread(self.file_name + img_name)
        img_name = img_name.replace('jpg', 'png').replace('JPG', 'png')

        # print(img_name)
        imgA = cv2.resize(imgA, (224, 224))

        # imgB = cv2.imread('./DUT-DBD_Dataset./DUT600GT_Training/' + img_name, 0)
        # imgB = cv2.imread(self.file_name_1 + img_name,0)
        # imgB = cv2.resize(imgB, (224, 224))
        # imgB = imgB / 255
        # imgB = imgB.astype('uint8')
        # imgB = onehot(imgB, 2)
        # imgB = imgB.transpose(2, 0, 1)
        # imgB = torch.FloatTensor(imgB)
        # print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)

        return imgA, img_name