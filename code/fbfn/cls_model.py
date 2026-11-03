"""FBFN 整图分类版：四分支证据图 + 图像级池化 + 注意力融合。

复用 branches.py 的四个分支（各自输出 (B,1,H,W) 证据图，未 sigmoid），
保留 fusion.py 的通道-空间双重注意力融合（像素级），最后把融合证据图做
全局平均池化得到「整图伪造得分」。

返回 dict：final/att/cnn/vit/ela/color 均为 (B,) 的图像级 logits。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .branches import CNNBranch, ViTBranch, NoiseBranch, ColorBranch
from .fusion import DualAttentionFusion


class FBFNClassifier(nn.Module):
    def __init__(self, pretrained=True, img_size=384, drop_branches=()):
        super().__init__()
        self.drop = set(drop_branches)
        self.branch_names = ["cnn", "vit", "noise", "color"]

        if "cnn" not in self.drop:
            self.cnn_branch = CNNBranch(pretrained=pretrained)
        if "vit" not in self.drop:
            self.vit_branch = ViTBranch(pretrained=pretrained, img_size=img_size)
        if "noise" not in self.drop:
            self.noise_branch = NoiseBranch()
        if "color" not in self.drop:
            self.color_branch = ColorBranch()

        self.fusion = DualAttentionFusion(in_channels=4 - len(self.drop))

    def _pool(self, x):
        """(B,1,H,W) 证据图 -> (B,) 图像级得分（全局平均池化）。"""
        return F.adaptive_avg_pool2d(x, 1).flatten(1).squeeze(1)

    def forward(self, x, ela=None, entropy=None):
        feats = []
        out = {}
        if "cnn" not in self.drop:
            p = self.cnn_branch(x)       # (B,1,H,W)
            feats.append(p)
            out["cnn"] = self._pool(p)
        if "vit" not in self.drop:
            p = self.vit_branch(x)
            feats.append(p)
            out["vit"] = self._pool(p)
        if "noise" not in self.drop:
            p = self.noise_branch(x, ela=ela)
            feats.append(p)
            out["ela"] = self._pool(p)
        if "color" not in self.drop:
            p = self.color_branch(x, entropy=entropy)
            feats.append(p)
            out["color"] = self._pool(p)
        final, att = self.fusion(feats)  # (B,1,H,W)
        out["final"] = self._pool(final)
        out["att"] = self._pool(att)
        return out
