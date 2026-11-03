# -*- coding: utf-8 -*-
"""FBFN 大骨干版：DINOv2 ViT-L/14 语义分支 + CNN/噪声/颜色分支 + 多任务头。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .branches import CNNBranch, NoiseBranch, ColorBranch
from .branches_large import DinoViTBranch
from .fusion import DualAttentionFusion


class FBFNTypeLarge(nn.Module):
    def __init__(self, pretrained=True, img_size=392, num_types=4):
        super().__init__()
        self.cnn_branch = CNNBranch(pretrained=pretrained)
        self.vit_branch = DinoViTBranch(img_size=img_size, pretrained=pretrained)
        self.noise_branch = NoiseBranch()
        self.color_branch = ColorBranch()
        self.fusion = DualAttentionFusion(in_channels=4)
        self.type_trunk = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.type_fc = nn.Linear(128, num_types)
        self.bin_fc = nn.Linear(128, 1)

    def forward(self, x, ela=None, entropy=None):
        p_cnn = self.cnn_branch(x)
        p_vit = self.vit_branch(x)
        p_ela = self.noise_branch(x, ela=ela)
        p_col = self.color_branch(x, entropy=entropy)
        final, att = self.fusion([p_cnn, p_vit, p_ela, p_col])

        feats = torch.cat([p_cnn, p_vit, p_ela, p_col], dim=1)
        tfeat = self.type_trunk(feats)
        type_logits = self.type_fc(tfeat)
        bin_logit = self.bin_fc(tfeat).squeeze(1)

        pool = lambda t: F.adaptive_avg_pool2d(t, 1).flatten(1).squeeze(1)
        return {
            "final_map": final,
            "final": bin_logit,
            "type": type_logits,
            "att": pool(att),
            "cnn": pool(p_cnn),
            "vit": pool(p_vit),
            "ela": pool(p_ela),
            "color": pool(p_col),
        }
