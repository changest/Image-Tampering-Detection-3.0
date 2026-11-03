"""FBFN 四个并行分支：CNN / ViT / ELA+SRM / Color。

约定：各分支 forward 输入为原始 RGB，取值 [0,1]。
CNN / ViT 分支内部自行做 ImageNet 归一化；噪声分支、颜色分支直接用原始像素。
各分支输出单通道 logits（未经过 sigmoid），尺寸与输入同分辨率。
"""
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _normalize_imagenet(x):
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


# ==================== 分支1: CNN (UNet++ / ResNet34) ====================
class CNNBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
            activation=None,   # 输出 logits
        )

    def forward(self, x):
        return self.model(_normalize_imagenet(x))


# ==================== 分支2: ViT (Swin-T + SegFormer MLP 解码器) ====================
class _MLP(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x):  # x: (B, C, H, W)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, H*W, C)
        x = self.proj(x)                   # (B, H*W, embed)
        return x.permute(0, 2, 1).reshape(b, -1, h, w)  # (B, embed, H, W)


class SegFormerHead(nn.Module):
    """SegFormer 轻量 MLP 解码器：4 个尺度特征融合上采样到 1/4，输出 1 通道。"""

    def __init__(self, in_channels=(96, 192, 384, 768), embed_dim=256, num_classes=1, dropout=0.1):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.linear_c4 = _MLP(c4, embed_dim)
        self.linear_c3 = _MLP(c3, embed_dim)
        self.linear_c2 = _MLP(c2, embed_dim)
        self.linear_c1 = _MLP(c1, embed_dim)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, feats):
        c1, c2, c3, c4 = feats            # 均为 NCHW，分辨率 1/4, 1/8, 1/16, 1/32
        target = c1.shape[2:]
        _c4 = F.interpolate(self.linear_c4(c4), size=target, mode="bilinear", align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=target, mode="bilinear", align_corners=False)
        _c2 = F.interpolate(self.linear_c2(c2), size=target, mode="bilinear", align_corners=False)
        _c1 = self.linear_c1(c1)
        fused = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        fused = self.dropout(fused)
        return self.linear_pred(fused)    # (B, 1, H/4, W/4)


class ViTBranch(nn.Module):
    def __init__(self, pretrained=True, img_size=384):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            img_size=(img_size, img_size),
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.head = SegFormerHead(in_channels=(96, 192, 384, 768), embed_dim=256, num_classes=1)

    def forward(self, x):
        feats = self.backbone(_normalize_imagenet(x))
        # Swin-T features_only 输出 NHWC，转为 NCHW
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        out = self.head(feats)            # (B, 1, H/4, W/4)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)


# ==================== 分支3: ELA + SRM (无训练参数的特征提取 + 1x1 头) ====================
class NoiseBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # 三个零和 5x5 高通 SRM 核（一/二/三阶边缘）
        filters = [
            [[0, 0, 0, 0, 0],
             [0, -1, 2, -1, 0],
             [0, 2, -4, 2, 0],
             [0, -1, 2, -1, 0],
             [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1],
             [2, -6, 8, -6, 2],
             [-2, 8, -12, 8, -2],
             [2, -6, 8, -6, 2],
             [-1, 2, -2, 2, -1]],
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 1, -2, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
        ]
        self.register_buffer("srm_filters", torch.tensor(filters, dtype=torch.float32).unsqueeze(1))
        # SRM 与 ELA 量级差约 29 倍，分别 InstanceNorm 对齐
        self.srm_norm = nn.InstanceNorm2d(3, affine=True)
        self.ela_norm = nn.InstanceNorm2d(1, affine=True)
        # 4 通道特征 (3 SRM + 1 ELA) -> 1 通道 logits（原 head 仅 1.2K 参数太弱，加深）
        self.head = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def compute_srm(self, x):
        b, c, h, w = x.shape
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return F.conv2d(gray, self.srm_filters, padding=2)   # (B, 3, H, W)

    def compute_ela(self, x):
        """ELA(x) = |I - JPEG(I, Q=90)|，逐 batch 在 CPU 用真实 JPEG 编解码。"""
        x_np = (x.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)  # (B,3,H,W)
        out = np.empty_like(x_np)
        for i in range(x_np.shape[0]):
            img_bgr = x_np[i].transpose(1, 2, 0)[:, :, ::-1]
            ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            comp = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
            out[i] = comp.transpose(2, 0, 1)
        comp = torch.from_numpy(out).to(x.device).float() / 255.0
        return (x - comp).abs().mean(dim=1, keepdim=True)   # (B, 1, H, W)

    def forward(self, x, ela=None):
        srm = self.compute_srm(x)
        if ela is None:
            with torch.no_grad():
                ela = self.compute_ela(x)
        srm = self.srm_norm(srm)
        ela = self.ela_norm(ela)
        return self.head(torch.cat([srm, ela], dim=1))


# ==================== 分支4: Color (Lab 熵 + MobileNetV3-Small) ====================
def _rgb_to_lab(x):
    """sRGB [0,1] -> CIE Lab (D65)。返回 (B,3,H,W)，各通道未归一化。"""
    x = x.clamp(0, 1)
    lin = torch.where(x > 0.04045, ((x + 0.055) / 1.055).pow(2.4), x / 12.92)
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    def _f(t):
        d = 6.0 / 29.0
        return torch.where(t > d ** 3, t.pow(1.0 / 3.0), t / (3 * d ** 2) + 4.0 / 29.0)

    fx, fy, fz = _f(X / 0.95047), _f(Y / 1.0), _f(Z / 1.08883)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.cat([L, a, bb], dim=1)


def _local_entropy_np(x_np, window=16, stride=8, bins=64):
    """16x16 滑窗（步长 8）直方图熵，CPU numpy 实现。

    x_np: (B,C,H,W) float32 取值 [0,1]。返回 (B,C,H',W') float32。
    """
    from numpy.lib.stride_tricks import sliding_window_view
    b, c, h, w = x_np.shape
    h_out = (h - window) // stride + 1
    w_out = (w - window) // stride + 1
    q = np.clip((x_np * (bins - 1)).astype(np.int32), 0, bins - 1)
    out = np.zeros((b, c, h_out, w_out), dtype=np.float32)
    for bi in range(b):
        for ci in range(c):
            win = sliding_window_view(q[bi, ci], (window, window))[::stride, ::stride]
            win = win.reshape(-1, window * window)                 # (L, 256)
            base = np.arange(win.shape[0], dtype=np.int64) * bins
            counts = np.bincount((win + base[:, None]).ravel(),
                                 minlength=win.shape[0] * bins)
            p = counts.reshape(win.shape[0], bins) / (window * window)
            ent = -(p * np.log2(p + 1e-8)).sum(axis=1)
            out[bi, ci] = ent.reshape(h_out, w_out)
    return out


class ColorBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # 熵图是低分辨率非自然图像，MobileNetV3 无法用 ImageNet 预训练，
        # 从零训练学不动。改用轻量浅 CNN（从零可学）。
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x, entropy=None):
        if entropy is None:
            lab = _rgb_to_lab(x)
            # 归一化到 [0,1]：L/100, a/b -> (v+128)/255
            L = lab[:, 0:1] / 100.0
            a = (lab[:, 1:2] + 128.0) / 255.0
            bb = (lab[:, 2:3] + 128.0) / 255.0
            lab_norm = torch.cat([L, a, bb], dim=1)
            # 熵是固定预处理（无可训练参数），在 CPU 上无梯度计算
            with torch.no_grad():
                ent_np = _local_entropy_np(lab_norm.detach().cpu().numpy())
            ent = torch.from_numpy(ent_np).to(x.device)        # (B,3,47,47)
        else:
            ent = entropy
        ent = F.interpolate(ent, size=(192, 192), mode="bilinear", align_corners=False)
        out = self.backbone(ent)
        out = self.head(out)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
