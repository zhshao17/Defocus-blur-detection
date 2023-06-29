import os
import logging
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics


def creat_mask(mask_ratio, patch_h, patch_w, x, b, c, h, w):
    num_patches = (h // patch_h) * (w // patch_w)
    # (b, c=3, h, w)->(b, n_patches, patch_size**2 * c)
    patches = x.view(
        b, c,
        h // patch_h, patch_h,
        w // patch_w, patch_w
    ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

    # 根据 mask 比例计算需要 mask 掉的 patch 数量
    # num_patches = (h // self.patch_h) * (w // self.patch_w)
    num_masked = int(mask_ratio * num_patches)

    # Shuffle:生成对应 patch 的随机索引
    # torch.rand() 服从均匀分布(normal distribution)
    # torch.rand() 只是生成随机数，argsort() 是为了获得成索引
    # (b, n_patches)
    shuffle_indices = torch.rand(b, num_patches, device='cuda').argsort()
    # mask 和 unmasked patches 对应的索引
    mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

    # 对应 batch 维度的索引：(b,1)
    batch_ind = torch.arange(b, device='cuda').unsqueeze(-1)
    # 利用先前生成的索引对 patches 进行采样，分为 mask 和 unmasked 两组
    patches[batch_ind, mask_ind] = patches[batch_ind, mask_ind] * 0
    patches= patches.reshape(b, h // patch_h, w // patch_w, patch_h, patch_w, c)
    patches= patches.permute(0, 5, 1, 3, 2, 4)
    x= patches.reshape(b, c, h, w)
    # patches = x.view(
    #     b, c,
    #     h // patch_h, patch_h,
    #     w // patch_w, patch_w
    # ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)
    return x
