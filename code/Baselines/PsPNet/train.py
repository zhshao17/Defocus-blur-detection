import os
import logging
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
from tqdm import tqdm
import click
import torch.nn.functional as F
import numpy as np
from pspnet import PSPNet
import dataloder
from sklearn import metrics

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    # net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda(0)
    return net, epoch


@click.command()
@click.option('--data-path', type=str, help='Path to dataset folder', default='./data/dataset/')
@click.option('--trainxml', type=str, help='Path to xml file', default='./data/dataset/training.xml')
@click.option('--models-path', type=str, help='Path for storing model snapshots', default='./models')
@click.option('--backend', type=str, default='resnet18', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=256, help='Horizontal random crop size')
@click.option('--crop_y', type=int, default=256, help='Vertical random crop size')
@click.option('--batch-size', type=int, default=30)
@click.option('--alpha', type=float, default=0.4, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=100, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.001)
@click.option('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')
def train(data_path, trainxml, models_path, backend, snapshot, crop_x, crop_y, batch_size, alpha, epochs, start_lr,
          milestones, gpu):
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # net, starting_epoch = build_network(snapshot, backend)
    # data_path = os.path.abspath(os.path.expanduser(data_path))
    # models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(models_path, exist_ok=True)
    save_epoch = 10
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    # traindata = HeadSegData(data_path, trainxml, train=True)
    traindata = dataloder.BagDataset(dataloder.transform, './DUT-DBD_Dataset/DUT600S_Training/')
    testdata = dataloder.BagDataset(dataloder.transform, './DUT-DBD_Dataset/DUT500S-Testing/')
    train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=1)
    net, _ = build_network(None, backend)
    seg_criterion = nn.NLLLoss().cuda(0)
    cls_criterion = nn.BCEWithLogitsLoss().cuda(0)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])

    print("start training...")
    net.train()
    for epoch in range(epochs):
        if epoch % 6 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        for i, (x, y,idx) in enumerate(train_loader):
            x, y = x.cuda(0), y[:, 0, :, :].reshape(batch_size, 224, 224).cuda(0).long()  # , y_cls.cuda(0).float()

            out, out_cls = net(x)

            seg_loss = seg_criterion(out, y)
            # cls_loss = cls_criterion(out_cls, y_cls)
            loss = seg_loss  # + alpha*cls_loss

            if i % 10 == 0:
                status = '[batch:{0}/{1} epoch:{2}] loss = {3:0.5f}'.format(i, len(traindata) // batch_size, epoch + 1,
                                                                            loss.item())
                print(status)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % save_epoch == 0:
            torch.save(net.state_dict(), os.path.join(models_path, str(epoch) + ".pth"))
    test(net, "./models_result", test_loader)
    mae1, fmeasure1, _, _ = eval1("./models_result", './DUT-DBD_Dataset/DUT500GT-Testing', 1.5)
    print(
        '====================================================================================================================')
    print("mae:", mae1, "fmeasure:", fmeasure1)
    print(
        '=====================================================================================================================')


def test(model, mask_save_path, dataloder):
    # model=build_network(None, backend)
    model = model.cuda()
    # dataloder = Get_dataloader_test(image_path, 1)
    for i, (img,_,name) in tqdm(enumerate(dataloder)):
        img = img.cuda()
        out, _ = model(img)
        out = torch.sigmoid(out)
        os.makedirs(mask_save_path, exist_ok=True)

        to_image_test(out, name, tag='', path=mask_save_path)


def to_image_test(tensor, i, tag, path):
    mask = tensor.detach().cpu().numpy()[0, 1, :, :]  # [i].cpu().clone()
    # print(mask.shape)
    if not os.path.isdir(path):
        os.makedirs(path)
    fake_samples_file = path + '/'+i[0]
    mask = Image.fromarray(mask * 255).convert('L')
    mask.save(fake_samples_file)


def eval1(mask_path, gt_path, m):
    files = os.listdir(gt_path)
    maes = 0
    precesions = 0
    recalls = 0
    fmeasures = 0
    for file in files:
        mask1 = mask_path + '/' + file
        gt1 = gt_path + '/' + file
        # mask=np.array(Image.open(mask1))
        mask1 = Image.open(mask1)
        mask1 = mask1.resize((320, 320))
        mask = np.array(mask1)
        mask = mask.astype(float) / 255.0
        mask_1 = mask

        (w, h) = mask.shape
        zeros = np.zeros((w, h))
        if m > 1:
            mean = np.mean(mask) * 1.5
        else:
            mean = m
        if mean > 1:
            mean = 1
        for i in range(w):
            for j in range(h):
                if mask_1[i, j] >= mean:
                    zeros[i, j] = 1.0
                else:
                    zeros[i, j] = 0.0

        gt = (np.array(Image.open(gt1)).astype(float)) / 255.0
        if gt.ndim==3:
            gt=gt[:,:,0]
        for i in range(w):
            for j in range(h):
                if gt[i, j] > 0.1:
                    gt[i, j] = 1.0
                else:
                    gt[i, j] = 0.0

        mae = np.mean(np.abs((gt - mask)))
        maes += mae
        precesion = metrics.precision_score(gt.reshape(-1), zeros.reshape(-1))
        precesions += precesion
        recall = metrics.recall_score(gt.reshape(-1), zeros.reshape(-1))
        recalls += recall
        if precesion == 0 and recall == 0:
            fmeasure = 0.0
        else:
            fmeasure = ((1 + 0.3) * precesion * recall) / (0.3 * precesion + recall)
        fmeasures += fmeasure
    mae1 = maes / len(files)
    fmeasure1 = fmeasures / len(files)
    recall1 = recalls / len(files)
    precesion1 = precesions / len(files)
    return mae1, fmeasure1, recall1, precesion1


# def test(model,test_dataloader,device):
#     test_loss = 0
#     model.eval()
#     with torch.no_grad():
#         for index, (bag, bag_msk) in enumerate(test_dataloader):
#             bag = bag.to(device)
#             bag_msk = bag_msk.to(device)
#
#             optimizer.zero_grad()
#             output = fcn_model(bag)
#             output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
#             loss = criterion(output, bag_msk)
#             iter_loss = loss.item()
#             all_test_iter_loss.append(iter_loss)
#             test_loss += iter_loss
#
#             output_np = output.cpu().detach().numpy().copy()
#             output_np = np.argmin(output_np, axis=1)
#             bag_msk_np = bag_msk.cpu().detach().numpy().copy()
#             bag_msk_np = np.argmin(bag_msk_np, axis=1)
#
#     cur_time = datetime.now()
#     h, remainder = divmod((cur_time - prev_time).seconds, 3600)
#     m, s = divmod(remainder, 60)
#     time_str = "Time %02d:%02d:%02d" % (h, m, s)
#     prev_time = cur_time
#
#     print('<---------------------------------------------------->')
#     print('epoch: %f' % epo)
#     print('epoch train loss = %f, epoch test loss = %f, %s'
#           % (train_loss / len(train_dataloader), test_loss / len(test_dataloader), time_str))
#
#     acc, acc_cls, mIou = label_accuracy_score(bag_msk_np, output_np, 2)
#     test_Acc.append(acc)
#     test_mIou.append(mIou)
#
#     print('Acc = %f, mIou = %f' % (acc, mIou))


if __name__ == '__main__':
    train()
