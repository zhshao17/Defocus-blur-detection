import torch
import torch.nn.functional as F
from torch import nn
import resnext_101_32x4d_
from ViT import ViT, PatchEmbedding
from torchsummary import summary
from torchvision import models


class BR2Net(nn.Module):
    def __init__(self):
        super(BR2Net, self).__init__()
        # resnext = ResNeXt101()

        resnext = ResNet18()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        # self.PatchEmbedding0 = PatchEmbedding(in_channels=64, patch_size=16, emb_size=256, img_size=224)
        # self.PatchEmbedding1 = PatchEmbedding(in_channels=256, patch_size=16, emb_size=256, img_size=224)
        # self.PatchEmbedding2 = PatchEmbedding(in_channels=512, patch_size=16, emb_size=256, img_size=224)
        # self.PatchEmbedding3 = PatchEmbedding(in_channels=1024, patch_size=16, emb_size=256, img_size=224)
        # self.PatchEmbedding4 = PatchEmbedding(in_channels=2048, patch_size=16, emb_size=256, img_size=224)

        # self.VIT = ViT(emb_size=256, depth=12)

        # self.VIT0 = ViT(in_channels=64, patch_size=16, emb_size=256, img_size=224, depth=12)
        # self.VIT1 = ViT(in_channels=256, patch_size=16, emb_size=256, img_size=224, depth=12)
        # self.VIT2 = ViT(in_channels=512, patch_size=16, emb_size=256, img_size=224, depth=12)
        # self.VIT3 = ViT(in_channels=1024, patch_size=16, emb_size=256, img_size=224, depth=12)
        # self.VIT4 = ViT(in_channels=2048, patch_size=16, emb_size=256, img_size=224, depth=12)

        self.VIT0 = ViT(in_channels=64, patch_size=16, emb_size=256, img_size=224, depth=12)
        self.VIT1 = ViT(in_channels=64, patch_size=16, emb_size=256, img_size=224, depth=12)
        self.VIT2 = ViT(in_channels=128, patch_size=16, emb_size=256, img_size=224, depth=12)
        self.VIT3 = ViT(in_channels=256, patch_size=16, emb_size=256, img_size=224, depth=12)
        self.VIT4 = ViT(in_channels=512, patch_size=16, emb_size=256, img_size=224, depth=12)


        ## channel weighted feature maps
        # self.CAlayer0 = nn.Sequential(CALayer(64, 16))
        # self.CAlayer1 = nn.Sequential(CALayer(256, 16))
        # self.CAlayer2 = nn.Sequential(CALayer(512, 16))
        # self.CAlayer3 = nn.Sequential(CALayer(1024, 16))
        # self.CAlayer4 = nn.Sequential(CALayer(2048, 16))

        ## Low to High
        self.predictL2H0 = nn.Conv2d(64 + 1, 1, kernel_size=1)
        self.predictL2H1 = nn.Sequential(
            # nn.Conv2d(257 + 1, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(65 + 1, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictL2H2 = nn.Sequential(
            nn.Conv2d(129 + 1, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictL2H3 = nn.Sequential(
            nn.Conv2d(257 + 1, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictL2H4 = nn.Sequential(
            nn.Conv2d(513 + 1, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        ## High to Low
        self.predictH2L0 = nn.Conv2d(512 + 1, 1, kernel_size=1)
        self.predictH2L1 = nn.Sequential(
            nn.Conv2d(257 + 1, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictH2L2 = nn.Sequential(
            nn.Conv2d(129 + 1, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictH2L3 = nn.Sequential(
            nn.Conv2d(65 + 1, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictH2L4 = nn.Sequential(
            nn.Conv2d(65 + 1, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.Sigmoid = nn.Sigmoid()

        ###
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer0 = F.upsample(layer0, size=x.size()[2:], mode='bilinear')
        layer1 = F.upsample(layer1, size=x.size()[2:], mode='bilinear')
        layer2 = F.upsample(layer2, size=x.size()[2:], mode='bilinear')
        layer3 = F.upsample(layer3, size=x.size()[2:], mode='bilinear')
        layer4 = F.upsample(layer4, size=x.size()[2:], mode='bilinear')

        l0_size = layer0.size()[2:]
        l4_size = layer4.size()[2:]

        ## Compute CA weighted features
        ViTlayer0 = self.VIT0(layer0)
        ViTlayer1 = self.VIT1(layer1)
        ViTlayer2 = self.VIT2(layer2)
        ViTlayer3 = self.VIT3(layer3)
        ViTlayer4 = self.VIT4(layer4)

        # ViTlayer0 = self.PatchEmbedding0(layer0)
        # ViTlayer1 = self.PatchEmbedding1(layer1)
        # ViTlayer2 = self.PatchEmbedding2(layer2)
        # ViTlayer3 = self.PatchEmbedding3(layer3)
        # ViTlayer4 = self.PatchEmbedding4(layer4)
        #
        # ViTlayer0 = self.VIT(ViTlayer0)
        # ViTlayer1 = self.VIT(ViTlayer1)
        # ViTlayer2 = self.VIT(ViTlayer2)
        # ViTlayer3 = self.VIT(ViTlayer3)
        # ViTlayer4 = self.VIT(ViTlayer4)

        ViTlayer0 = torch.cat([ViTlayer0, layer0], dim=1)
        ViTlayer1 = torch.cat([ViTlayer1, layer1], dim=1)
        ViTlayer2 = torch.cat([ViTlayer2, layer2], dim=1)
        ViTlayer3 = torch.cat([ViTlayer3, layer3], dim=1)
        ViTlayer4 = torch.cat([ViTlayer4, layer4], dim=1)

        # CAlayer0 = self.CAlayer0(layer0)
        # CAlayer1 = self.CAlayer1(layer1)
        # CAlayer2 = self.CAlayer2(layer2)
        # CAlayer3 = self.CAlayer3(layer3)
        # CAlayer4 = self.CAlayer4(layer4)

        predictL2H0 = self.predictL2H0(ViTlayer0)
        predictL2H1 = self.predictL2H1(
            # torch.cat((predictL2H0, F.upsample(ViTlayer1, size=l0_size, mode='bilinear')), 1)) + predictL2H0
            torch.cat((predictL2H0, ViTlayer1), 1)) + predictL2H0
        predictL2H2 = self.predictL2H2(
            # torch.cat((predictL2H1, F.upsample(ViTlayer2, size=l0_size, mode='bilinear')), 1)) + predictL2H1
            torch.cat((predictL2H1, ViTlayer2), 1)) + predictL2H1
        predictL2H3 = self.predictL2H3(
            # torch.cat((predictL2H2, F.upsample(ViTlayer3, size=l0_size, mode='bilinear')), 1)) + predictL2H2
            torch.cat((predictL2H2, ViTlayer3), 1)) + predictL2H2
        predictL2H4 = self.predictL2H4(
            # torch.cat((predictL2H3, F.upsample(ViTlayer4, size=l0_size, mode='bilinear')), 1)) + predictL2H3
            torch.cat((predictL2H3, ViTlayer4), 1)) + predictL2H3
        predictH2L0 = self.predictH2L0(ViTlayer4)
        predictH2L1 = self.predictH2L1(
            # torch.cat((predictH2L0, F.upsample(ViTlayer3, size=l4_size, mode='bilinear')), 1)) + predictH2L0
            torch.cat((predictH2L0, ViTlayer3), 1)) + predictH2L0
        predictH2L2 = self.predictH2L2(
            # torch.cat((predictH2L1, F.upsample(ViTlayer2, size=l4_size, mode='bilinear')), 1)) + predictH2L1
            torch.cat((predictH2L1, ViTlayer2), 1)) + predictH2L1
        predictH2L3 = self.predictH2L3(
            # torch.cat((predictH2L2, F.upsample(ViTlayer1, size=l4_size, mode='bilinear')), 1)) + predictH2L2
            torch.cat((predictH2L2, ViTlayer1), 1)) + predictH2L2
        predictH2L4 = self.predictH2L4(
            # torch.cat((predictH2L3, F.upsample(ViTlayer0, size=l4_size, mode='bilinear')), 1)) + predictH2L3
            torch.cat((predictH2L3, ViTlayer0), 1)) + predictH2L3

        # predictL2H0 = F.upsample(predictL2H0, size=x.size()[2:], mode='bilinear')
        # predictL2H1 = F.upsample(predictL2H1, size=x.size()[2:], mode='bilinear')
        # predictL2H2 = F.upsample(predictL2H2, size=x.size()[2:], mode='bilinear')
        # predictL2H3 = F.upsample(predictL2H3, size=x.size()[2:], mode='bilinear')
        # predictL2H4 = F.upsample(predictL2H4, size=x.size()[2:], mode='bilinear')
        # predictH2L0 = F.upsample(predictH2L0, size=x.size()[2:], mode='bilinear')
        # predictH2L1 = F.upsample(predictH2L1, size=x.size()[2:], mode='bilinear')
        # predictH2L2 = F.upsample(predictH2L2, size=x.size()[2:], mode='bilinear')
        # predictH2L3 = F.upsample(predictH2L3, size=x.size()[2:], mode='bilinear')
        # predictH2L4 = F.upsample(predictH2L4, size=x.size()[2:], mode='bilinear')

        predictFusion = (predictL2H4 + predictH2L4) / 2

        if self.training:
            return self.Sigmoid(predictL2H0), self.Sigmoid(predictL2H1), self.Sigmoid(predictL2H2), self.Sigmoid(
                predictL2H3), self.Sigmoid(predictL2H4), \
                   self.Sigmoid(predictH2L0), self.Sigmoid(predictH2L1), self.Sigmoid(predictH2L2), self.Sigmoid(
                predictH2L3), self.Sigmoid(predictH2L4), \
                   self.Sigmoid(predictFusion)
        return self.Sigmoid(predictFusion)


# class Patch_Attention(nn.Model):
#     def __init__(self):
#
#     def forward(self, x):


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        net = models.resnet18(pretrained=False)
        net.load_state_dict(torch.load('resnet18-5c106cde.pth'))
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:4])
        self.layer1 = net[4]
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d_.resnext_101_32x4d
        state_dict = torch.load('resnext_101_32x4d.pth')
        net.load_state_dict(state_dict, False)

        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:4])
        self.layer1 = net[4]
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


if __name__ == '__main__':
    model = BR2Net()
    summary(model, input_size=[(3, 224, 224)], batch_size=2, device="cpu")
