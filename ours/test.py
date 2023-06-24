import os
import logging
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import click
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from PIL import Image

import warnings


# warnings.filterwarnings("ignore")

# test函数进行模型测试并储存测试结果

def test(stict, model, mask_save_path, dataloder):
    """
    model: well_train model
    mask_save_path: path to save the result, you can use  "./models_result/result_epoch"
    dataloder: dataloder of test data, which contain img, gt, img_name
    """
    # model: well_train model
    #
    ""
    model.load_state_dict(torch.load(stict))
    model.eval()
    # dataloder = Get_dataloader_test(image_path, 1)
    with torch.no_grad():
        for i, (img, _, name) in tqdm(enumerate(dataloder)):
            img = Variable(img).cuda()
            out = model(img)
            # out = torch.sigmoid(out)
            os.makedirs(mask_save_path, exist_ok=True)
            to_image_test(out, name, tag='', path=mask_save_path)


# 保存结果函数
def to_image_test(tensor, i, tag, path):
    mask = tensor.detach().cpu().numpy()[0, 0, :, :]  # [i].cpu().clone()
    # print(mask.shape)
    if not os.path.isdir(path):
        os.makedirs(path)
    fake_samples_file = path + '/' + i[0]
    mask = Image.fromarray(mask * 255).convert('L')
    mask.save(fake_samples_file)


# 利用保存的结果与GT计算Fmeasure
def eval1(mask_path, gt_path, m):
    """
    mask_path: path to save the result, you can use  "./models_result/result_epoch"
    gt_path: GT path
    """
    files = os.listdir(gt_path)
    maes = 0
    precesions = 0
    recalls = 0
    fmeasures = 0.0
    for file in files:
        mask1 = mask_path + '/' + file
        gt1 = gt_path + '/' + file
        # mask=np.array(Image.open(mask1))
        mask1 = Image.open(mask1)
        mask1 = mask1.resize((320, 320))
        mask = np.array(mask1)
        mask = mask.astype(float) / 255.0
        mask_1 = mask
        # print(mask_1)

        (w, h) = mask.shape
        zeros = np.zeros((w, h))
        if m > 1:
            mean = np.mean(mask) * m
        else:
            mean = m
        if mean > 1:
            mean = 1
        # print('mean', mean)
        for i in range(w):
            for j in range(h):
                if mask_1[i, j] >= mean:
                    zeros[i, j] = 1.0
                else:
                    zeros[i, j] = 0.0

        gt = (np.array(Image.open(gt1)).astype(float)) / 255.0
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        for i in range(w):
            for j in range(h):
                if gt[i, j] > 0.1:
                    gt[i, j] = 1.0
                else:
                    gt[i, j] = 0.0

        mae = np.mean(np.abs((gt - mask)))
        maes += mae
        precesion = metrics.precision_score(gt.reshape(-1), zeros.reshape(-1))
        # print('pre',precesion)
        precesions += precesion
        recall = metrics.recall_score(gt.reshape(-1), zeros.reshape(-1))
        # print('recall',recall)
        recalls += recall
        if precesion == 0.0 and recall == 0.0:
            fmeasure = 0.0
        else:
            fmeasure = ((1 + 0.3) * precesion * recall) / (0.3 * precesion + recall)
        fmeasures += fmeasure
    mae1 = maes / len(files)
    fmeasure1 = fmeasures / len(files)
    recall1 = recalls / len(files)
    precesion1 = precesions / len(files)
    return mae1, fmeasure1, recall1, precesion1
