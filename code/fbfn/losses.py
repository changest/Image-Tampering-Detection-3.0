"""Focal + Dice + Tversky 组合损失（论文式 6）。

每个损失项 L_i = 1.0*Focal + 1.0*Dice + 0.5*Tversky。
总损失 L = L_final + 0.5*L_att + 0.5*(L_cnn+L_vit+L_ela+L_color)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits, target, gamma=2.0, alpha=0.25, eps=1e-7):
    """二分类 Focal Loss，logits 未经过 sigmoid。"""
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    focal = -alpha * (1 - pt).pow(gamma) * torch.log(pt + eps)
    return focal.mean()


def dice_loss(logits, target, smooth=1.0):
    """Dice Loss（逐图平均）。"""
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + smooth) / (union + smooth)
    return (1 - dice).mean()


def tversky_loss(logits, target, alpha=0.7, beta=0.3, smooth=1.0):
    """Tversky Loss（精确率/召回率非对称权衡）。"""
    p = torch.sigmoid(logits)
    tp = (p * target).sum(dim=(2, 3))
    fp = (p * (1 - target)).sum(dim=(2, 3))
    fn = ((1 - p) * target).sum(dim=(2, 3))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - tversky).mean()


def branch_loss(logits, target):
    """单分支损失：1.0*Focal + 1.0*Dice + 0.5*Tversky。"""
    return (focal_loss(logits, target)
            + dice_loss(logits, target)
            + 0.5 * tversky_loss(logits, target))


def total_loss(final, att, branches, target):
    """branches: dict{cnn,vit,ela,color} 各为 logits。

    L = L_final + 0.5*L_att + 0.5*(L_cnn+L_vit+L_ela+L_color)
    """
    l_final = branch_loss(final, target)
    l_att = branch_loss(att, target)
    l_branch = sum(branch_loss(b, target) for b in branches.values())
    return l_final + 0.5 * l_att + 0.5 * l_branch
