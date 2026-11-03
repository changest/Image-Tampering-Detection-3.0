# -*- coding: utf-8 -*-
"""大骨干分支：DINOv2 ViT-L/14 作为语义分支。

注意：DINOv2 是 14×14 patch，输入尺寸必须是 14 的倍数（392=28×14 或 448=32×14）。
"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoViTBranch(nn.Module):
    def __init__(self, img_size=392, pretrained=True,
                 model_name="vit_large_patch14_dinov2.lvd142m"):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            img_size=img_size, dynamic_img_size=True,
        )
        try:
            self.backbone.set_grad_checkpointing(True)
        except Exception:
            pass
        dim = self.backbone.num_features
        # 先 1x1 降维再上采样，避免 1024 通道 × 448² 触发 CUDA 插值溢出
        self.reduce = nn.Conv2d(dim, 128, 1)
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):
        t = self.backbone.forward_features(x)      # (B, L, D)
        b, l, d = t.shape
        n = l
        if int(round(math.sqrt(n))) ** 2 != n:     # 去掉 cls token
            t = t[:, 1:]
            n = l - 1
        p = int(round(math.sqrt(n)))
        f = t.transpose(1, 2).reshape(b, d, p, p)
        f = self.reduce(f)                                  # (B,128,p,p)
        f = F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(f)
