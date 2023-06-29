import os
import dataloder2
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from PIL import Image
import warnings
from model import BR2Net


warnings.filterwarnings("ignore")


def to_image_test(tensor, i, tag, path):
    mask = tensor.detach().cpu().numpy()[0, 0, :, :]  # [i].cpu().clone()
    # print(mask.shape)
    if not os.path.isdir(path):
        os.makedirs(path)
    fake_samples_file = path + '/' + i[0]
    mask = Image.fromarray(mask * 255).convert('L')
    mask.save(fake_samples_file)


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
        # mask1 = mask1.resize((320, 320))
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

        gt = (np.array(Image.open(gt1).resize((224, 224))).astype(float)) / 255.0

        if gt.ndim == 3:
            gt = gt[:, :, 0]
        for i in range(w):
            for j in range(h):
                if gt[i, j] > 0.1:
                    gt[i, j] = 0  # 1.0
                else:
                    gt[i, j] = 1.0

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
    model = model.cuda()
    model.eval()
    # dataloder = Get_dataloader_test(image_path, 1)
    for i, (img, _, name) in tqdm(enumerate(dataloder)):
        # img = Variable(img).cuda()
        # print(name)
        img = img.cuda()
        out = model(img)
        # out = torch.sigmoid(out)
        os.makedirs(mask_save_path, exist_ok=True)

        to_image_test(out, name, tag='', path=mask_save_path)


def train():
    # train_data = dataloder.BagDataset(dataloder.transform,'./DUT-DBD_Dataset/DUT600S_Training/')
    test_data = dataloder2.BagDataset(dataloder2.transform, '../data/CUHK/image/')

    # 利用DataLoader生成一个分batch获取数据的可迭代对象

    # train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BR2Net()
    stict = './mask/model50.pth'
    test(stict, model, "./mask/models_CUHK_result", test_dataloader)
    mae1, fmeasure1, _, _ = eval1("./mask/models_CUHK_result", '../data/CUHK/gt', 1.5)

    print(
        '====================================================================================================================')
    print("mae:", mae1, "fmeasure:", fmeasure1)
    print(
        '=====================================================================================================================')


if __name__ == "__main__":
    train()
