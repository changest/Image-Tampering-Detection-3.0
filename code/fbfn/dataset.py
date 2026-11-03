"""CASIA v2.0 篡改定位数据集（仅篡改图 Tp + 对应 GT 掩码）。

图像命名：Tp_D_..._xxx.{tif,jpg}，掩码命名：{stem}_gt.png。
增强（训练）：随机水平翻转、旋转 ±15°、亮度/对比度 ±20%、弹性变形。
"""
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def imread_unicode(path, flags):
    """OpenCV 在 Windows 下无法直接读取含中文的路径，用 numpy 中转。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


class CASIADataset(Dataset):
    def __init__(self, casia_root, gt_root, split="train", img_size=384,
                 val_ratio=0.2, seed=42, train=True):
        self.img_size = img_size
        self.train = train
        tp_dir = os.path.join(casia_root, "Tp")
        imgs = sorted(
            f for f in os.listdir(tp_dir)
            if f.lower().endswith((".tif", ".jpg", ".jpeg", ".png"))
        )
        # 仅保留有掩码的图像
        self.pairs = []
        for f in imgs:
            stem = os.path.splitext(f)[0]
            mask_path = os.path.join(gt_root, stem + "_gt.png")
            if os.path.exists(mask_path):
                self.pairs.append((os.path.join(tp_dir, f), mask_path))

        rng = random.Random(seed)
        rng.shuffle(self.pairs)
        n_val = int(len(self.pairs) * val_ratio)
        if split == "val":
            self.pairs = self.pairs[:n_val]
        else:
            self.pairs = self.pairs[n_val:]

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img, mask):
        # 水平翻转
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        # 旋转 ±15°
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_REPLICATE)
        # 弹性变形
        alpha, sigma = 34, 4
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        img = cv2.remap(img, (xs + dx).astype(np.float32), (ys + dy).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.remap(mask, (xs + dx).astype(np.float32), (ys + dy).astype(np.float32),
                         cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = imread_unicode(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)

        # 统一 resize 到 384x384
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.train:
            img, mask = self._augment(img, mask)
            # 亮度/对比度 ±20%
            alpha_c = 1.0 + random.uniform(-0.2, 0.2)
            beta_c = random.uniform(-0.2, 0.2) * 255.0
            img = np.clip(alpha_c * img.astype(np.float32) + beta_c, 0, 255).astype(np.uint8)

        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0      # (3,H,W) [0,1]
        mask = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)  # (1,H,W) 0/1
        return img, mask
