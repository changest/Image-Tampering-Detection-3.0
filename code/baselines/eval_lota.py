"""LOTA（ICCV 2025, Bit-Planes Guided AI-Generated Image Detection）跨生成器评估。

复刻 LOTA 官方推理管线（内联 model + bit_patch，无需 clone 官方仓库）：
  1. 提取 RGB 最低 3 个 bit-plane（& 0x07 后 *36 放大）
  2. 随机 crop 64 个 32×32 patch，选「梯度最大」的 patch（compute 计算水平+垂直+对角差之和）
  3. resize 回 256×256 → ToTensor → ImageNet 归一化 → ResNet-50 → sigmoid

标签方向：LOTA 的 sigmoid 高 = 真图（natural），与 FBFN（高=假）相反，故翻转成 fake_score。

测试集与 eval_baselines.py 完全一致（sd14 val + crossval 8 生成器），输出 Acc/AUC/AP。

用法：
    python -m baselines.eval_lota --ckpt baselines/lota/sd14_lota.pth
"""
import os
import argparse
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import roc_auc_score, average_precision_score

from .eval_baselines import GENS, _test_dirs, IMAGENET_MEAN, IMAGENET_STD

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 256
PATCH_SIZE = 32


# ---------------- LOTA 模型（等价官方 model.py：ResNet-50 + Linear(2048,1)） ----------------
class LOTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = resnet50(weights=None)
        self.disc.fc = nn.Linear(2048, 1)

    def forward(self, x):
        return self.disc(x)


# ---------------- LOTA bit_patch 预处理（复刻官方 bit_patch.py） ----------------
def _patch_gradient(patch):
    p = np.array(patch).astype(np.int64)
    dh = np.sum(np.abs(p[:, :-1, :] - p[:, 1:, :]))
    dv = np.sum(np.abs(p[:-1, :, :] - p[1:, :, :]))
    dd = np.sum(np.abs(p[:-1, :-1, :] - p[1:, 1:, :]))
    dd += np.sum(np.abs(p[1:, :-1, :] - p[:-1, 1:, :]))
    return int(dh + dv + dd)


def bit_patch_transform(img_pil):
    """最低 3 bit-plane 提取 + 最大梯度 patch 选择 + resize 回 256。"""
    img = np.array(img_pil).astype(np.uint8)  # RGB
    low3 = (img & 0x07) * (255 // 7)  # 每通道取最低 3 bit 后放大
    combined = cv2.merge((low3[:, :, 0], low3[:, :, 1], low3[:, :, 2])).astype(np.uint8)
    h, w = combined.shape[:2]
    pil = Image.fromarray(combined)
    if min(h, w) < PATCH_SIZE:
        pil = transforms.Resize((IMG_SIZE, IMG_SIZE))(pil)

    n_patch = (IMG_SIZE // PATCH_SIZE) ** 2
    crop = transforms.RandomCrop(PATCH_SIZE)
    best, best_grad = None, -1
    for _ in range(n_patch):
        p = crop(pil)
        g = _patch_gradient(p)
        if g > best_grad:
            best, best_grad = p, g
    patch = np.array(best)
    return cv2.resize(patch, (IMG_SIZE, IMG_SIZE))


TRANSFORM = transforms.Compose([
    transforms.Lambda(bit_patch_transform),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------- 数据加载 ----------------
class LotaDir(Dataset):
    def __init__(self, real_dir, fake_dir):
        self.items = []
        for d, lab in ((real_dir, 0), (fake_dir, 1)):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    self.items.append((os.path.join(d, f), lab))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        img = Image.open(p).convert("RGB")
        return TRANSFORM(img), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "lota", "sd14_lota.pth"))
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LOTA().to(device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    print(f"加载 {args.ckpt}", flush=True)

    print(f"\n=== LOTA (sd14) ===", flush=True)
    for gen in GENS:
        real_dir, fake_dir = _test_dirs(gen)
        ds = LotaDir(real_dir, fake_dir)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        scores, labels = [], []
        with torch.no_grad():
            for imgs, lab in loader:
                logits = model(imgs.to(device)).squeeze(1)
                fake_score = 1.0 - torch.sigmoid(logits)  # LOTA 高=真，翻转为高=假
                scores.append(fake_score.cpu().numpy())
                labels.append(lab.numpy())
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        acc = float(((scores > 0.5).astype(float) == labels).mean())
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
        print(f"  {gen:10s} 真 {int((labels == 0).sum())}/假 {int((labels == 1).sum())} | "
              f"Acc={acc * 100:.2f} AUC={auc * 100:.2f} AP={ap * 100:.2f}", flush=True)


if __name__ == "__main__":
    main()
