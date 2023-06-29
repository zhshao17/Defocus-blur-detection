import os
import logging
from tqdm import tqdm
import click
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 256, img_size: int = 224):
        self.patch_size = patch_size
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Sequential(
            # 使用一个卷积层而不是一个线性层 -> 性能增加
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # 位置编码信息，一共有(img_size // patch_size)**2 + 1(cls token)个位置向量
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        # cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # 将cls token在维度1扩展到输入上
        # x = torch.cat([cls_tokens, x], dim=1)
        # 添加位置编码
        # print(x.shape, self.positions.shape)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 256, num_heads: int = 1, dropout: float = 0):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 使用单个矩阵一次性计算出queries,keys,values
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 将queries，keys和values划分为num_heads
        # print("1qkv's shape: ", self.qkv(x).shape)  # 使用单个矩阵一次性计算出queries,keys,values
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)  # 划分到num_heads个头上
        # print("2qkv's shape: ", qkv.shape)

        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # print("queries's shape: ", queries.shape)
        # print("keys's shape: ", keys.shape)
        # print("values's shape: ", values.shape)

        # 在最后一个维度上相加
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        # print("energy's shape: ", energy.shape)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        # print("scaling: ", scaling)
        att = F.softmax(energy, dim=-1) / scaling
        # print("att1' shape: ", att.shape)
        att = self.att_drop(att)
        # print("att2' shape: ", att.shape)

        # 在第三个维度上相加
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # print("out1's shape: ", out.shape)
        out = rearrange(out, "b h n d -> b n (h d)")
        # print("out2's shape: ", out.shape)
        out = self.projection(out)
        # print("out3's shape: ", out.shape)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 256,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super(TransformerEncoderBlock, self).__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super(TransformerEncoder, self).__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 256, patch_num: int = 14, patch_size: int = 16):
        super(ClassificationHead, self).__init__(

            Rearrange('(b c) (p1 p2) (s1 s2)  -> b c (p1 s1) (p2 s2)', p1=patch_num, p2=patch_num, s1=patch_size,
                      s2=patch_size, c=1)
            # Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            # Reduce('b n e -> b e', reduction='mean'),
            # nn.LayerNorm(emb_size),
            # nn.Linear(emb_size, n_classes)
        )


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 256,
                 img_size: int = 224,
                 depth: int = 12,
                 **kwargs):
        super(ViT, self).__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size)
        )

# if __name__ == '__main__':
#     model = ViT()
#     summary(model, input_size=[(3, 224, 224)], batch_size=10, device="cpu")
