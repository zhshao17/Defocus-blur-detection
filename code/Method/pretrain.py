import torch
import numpy as np
import dataloder
from torch.utils.data import DataLoader, random_split
import random
import torch.nn as nn
import gc


def pretrain(model, epochs=10, batch_size=4):
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = dataloder.BagDataset(dataloder.transform, '../data/data_augmented/data00GT/')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    epoch = 0

    initial_lr = 0.001
    opt = torch.optim.Adam(model.parameters(), lr=initial_lr)  # try SGD
    # opt = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99)
    criterion = nn.BCELoss().to(device)

    while epoch <= epochs:
        model.train()
        for index, (x, y, idx) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            outputsL2H0, outputsL2H1, outputsL2H2, outputsL2H3, outputsL2H4, \
            outputsH2L0, outputsH2L1, outputsH2L2, outputsH2L3, outputsH2L4, \
            outputsFusion = model(x)

            y = y[:, 0, :, :]  # batch * label * H * W
            # y = y[:, 1, :, :]  # batch * label * H * W # label 反
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

            loss = lossFusion + lossL2H0 + lossL2H1 + lossL2H2 + lossL2H3 + lossL2H4 + lossH2L0 + lossH2L1 + lossH2L2 + lossH2L3 + lossH2L4

            loss.backward()
            opt.step()
            # 每15个bacth，输出一次训练过程的数据
            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epoch, index, len(train_loader), loss.item()))
                # break
        gc.collect()
        torch.cuda.empty_cache()

        print("----------------------------------------------------")

        epoch = epoch + 1

        torch.save(model.state_dict(), "model-pretrain-2-" + str(epoch) + ".pth")

