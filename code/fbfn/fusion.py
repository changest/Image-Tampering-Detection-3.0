"""通道-空间双重注意力融合模块（论文式 4）。

四个分支各输出单通道特征图 f_cnn/f_vit/f_ela/f_color ∈ R^{HxWx1}。
拼接得 F ∈ R^{HxWx4}；通道注意力 alpha = Softmax(MLP(GAP(F))) ∈ R^4；
空间注意力经 3x3 卷积 + Sigmoid 生成逐像素权重；最后 1x1 卷积输出 1 通道。
返回 (final_logits, att_logits)，att 为通道注意力加权后的中间输出（用于 L_att）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttentionFusion(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # 通道注意力：GAP -> MLP -> Softmax
        self.ch_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )
        # 空间注意力：3x3 卷积 + Sigmoid
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        # 输出映射
        self.out_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.att_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, feats):
        # feats: list of 4 tensors (B, 1, H, W)
        x = torch.cat(feats, dim=1)                       # (B, 4, H, W)

        # 通道注意力
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)      # (B, 4)
        alpha = F.softmax(self.ch_mlp(gap), dim=1)        # (B, 4)
        x_ch = x * alpha.view(-1, x.shape[1], 1, 1)       # 通道加权

        # 空间注意力
        w_sp = torch.sigmoid(self.spatial_conv(x_ch))     # (B, 1, H, W)
        x_sp = x_ch * w_sp

        final = self.out_conv(x_sp)                        # (B, 1, H, W)
        att = self.att_conv(x_ch)                          # 通道注意力中间输出
        return final, att
