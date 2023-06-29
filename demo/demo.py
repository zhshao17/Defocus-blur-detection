import os
import demodata
import LUHAONet
import ViT
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from PIL import Image
import warnings
import gc
warnings.filterwarnings("ignore")


def to_image_test(tensor, i, path):
    mask = tensor.detach().cpu().numpy()[0, 0, :, :]  # [i].cpu().clone()
    # print(mask.shape)
    if not os.path.isdir(path):
        os.makedirs(path)
    fake_samples_file = path + str(i)+'.jpg'
    mask = Image.fromarray(mask * 255).convert('L')
    mask.save(fake_samples_file)


def test(model, input, name, net):
    input = input.cpu()
    with torch.no_grad():
        out = model(input)
    os.makedirs('./result/', exist_ok=True)
    if net == 1:
        to_image_test(out, name, path='./result/1-')
    elif net == 2:
        to_image_test(out, name, path='./result/2-')
    # gc.collect()
    # torch.cuda.empty_cache()

