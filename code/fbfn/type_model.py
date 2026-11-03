# -*- coding: utf-8 -*-
"""FBFN + 篡改类型分类头（真实 / 拼接 / 复制移动 / AI 生成）。

四分支证据图（各 1 通道）拼接后：
  - 走原融合模块得到像素级真伪 logits；
  - 同时送入 type_head，输出 4 类 logits。
整图真伪分数由融合图全局平均池化得到。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .branches import CNNBranch, ViTBranch, NoiseBranch, ColorBranch
from .fusion import DualAttentionFusion


class FBFNType(nn.Module):
    def __init__(self, pretrained=True, img_size=256, num_types=4):
        super().__init__()
        self.cnn_branch = CNNBranch(pretrained=pretrained)
        self.vit_branch = ViTBranch(pretrained=pretrained, img_size=img_size)
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
        # 专用二分类头（真/假），避免对稀疏掩码池化导致信号被稀释
        self.bin_fc = nn.Linear(128, 1)

    def forward(self, x, ela=None, entropy=None):
        p_cnn = self.cnn_branch(x)
        p_vit = self.vit_branch(x)
        p_ela = self.noise_branch(x, ela=ela)
        p_col = self.color_branch(x, entropy=entropy)
        final, att = self.fusion([p_cnn, p_vit, p_ela, p_col])

        feats = torch.cat([p_cnn, p_vit, p_ela, p_col], dim=1)  # (B,4,H,W)
        tfeat = self.type_trunk(feats)                           # (B,128)
        type_logits = self.type_fc(tfeat)                        # (B,4)
        bin_logit = self.bin_fc(tfeat).squeeze(1)                # (B,) 真/假 logit

        pool = lambda t: F.adaptive_avg_pool2d(t, 1).flatten(1).squeeze(1)
        return {
            "final_map": final,          # (B,1,H,W) 像素级 logits，用于分割损失
            "final": bin_logit,          # (B,) 专用二分类头
            "type": type_logits,
            "att": pool(att),
            "cnn": pool(p_cnn),
            "vit": pool(p_vit),
            "ela": pool(p_ela),
            "color": pool(p_col),
        }
