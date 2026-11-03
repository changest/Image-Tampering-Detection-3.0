"""FBFN 整体组装：四分支 + 通道-空间双重注意力融合。"""
import torch
import torch.nn as nn

from .branches import CNNBranch, ViTBranch, NoiseBranch, ColorBranch
from .fusion import DualAttentionFusion


class FBFN(nn.Module):
    def __init__(self, pretrained=True, img_size=384):
        super().__init__()
        self.cnn_branch = CNNBranch(pretrained=pretrained)
        self.vit_branch = ViTBranch(pretrained=pretrained, img_size=img_size)
        self.noise_branch = NoiseBranch()
        self.color_branch = ColorBranch()
        self.fusion = DualAttentionFusion(in_channels=4)

    def forward(self, x):
        p_cnn = self.cnn_branch(x)
        p_vit = self.vit_branch(x)
        p_ela = self.noise_branch(x)
        p_color = self.color_branch(x)
        final, att = self.fusion([p_cnn, p_vit, p_ela, p_color])
        return {
            "final": final,
            "att": att,
            "cnn": p_cnn,
            "vit": p_vit,
            "ela": p_ela,
            "color": p_color,
        }
