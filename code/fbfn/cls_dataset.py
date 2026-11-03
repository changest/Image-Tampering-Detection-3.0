"""整图分类数据集加载器（真 vs 假，图像级标签）。

目录结构约定：
    real_dir/  -> 真图（标签 0）
    fake_dir/  -> 生成/PS 图（标签 1）

兼容 GenImage（train/{nature,ai}）与 CIFAKE（train/{REAL,FAKE}）：
    只需分别把 real_dir 指到 nature/REAL、fake_dir 指到 ai/FAKE 即可。

熵图 + ELA 走预计算缓存（cache_dir），训练时直接加载，避免每 epoch 重复计算。
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


class ImageClsDataset(Dataset):
    def __init__(self, real_dir, fake_dir, img_size=256, train=True, cache_dir=None):
        self.img_size = img_size
        self.train = train
        self.cache_dir = cache_dir
        self.samples = []  # (path, label)

        for d, label in ((real_dir, 0), (fake_dir, 1)):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"目录不存在: {d}")
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".webp", ".bmp")):
                    self.samples.append((os.path.join(d, f), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"{real_dir} 与 {fake_dir} 下没有图像")

    def __len__(self):
        return len(self.samples)

    def _load_cache(self, path):
        """返回 (entropy, ela)，entropy (3,47,47) float32，ela (1,H,W) float32。

        缓存路径约定：{cache_dir}/{split}/{real|fake}/{stem}.npz，其中 split/sub
        取自图片路径的最后两级目录（如 train/real）。
        """
        sub = os.path.basename(os.path.dirname(path))
        split = os.path.basename(os.path.dirname(os.path.dirname(path)))
        stem = os.path.splitext(os.path.basename(path))[0]
        npz = os.path.join(self.cache_dir, split, sub, stem + ".npz")
        data = np.load(npz)
        return data["ent"].astype(np.float32), data["ela"].astype(np.float32)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = imread_unicode(path, cv2.IMREAD_COLOR)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        entropy, ela = self._load_cache(path)
        if self.train and random.random() < 0.5:
            img = cv2.flip(img, 1)
            entropy = np.ascontiguousarray(entropy[:, :, ::-1])
            ela = np.ascontiguousarray(ela[:, :, ::-1])

        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        entropy = torch.from_numpy(entropy)
        ela = torch.from_numpy(ela)
        label = torch.tensor(float(label))
        return img, entropy, ela, label
