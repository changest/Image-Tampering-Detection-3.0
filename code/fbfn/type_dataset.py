# -*- coding: utf-8 -*-
"""带类型标签 + 分割掩码的数据集（真实/拼接/复制移动/AI生成）。

读取 build_type_manifest.py 生成的 CSV（含 mask / has_mask 列）。
- mask 非空：读取 GT 掩码并二值化；
- mask 为空且 has_mask=1：真图，使用全零掩码；
- has_mask=0：AI 生成图，无掩码，跳过分割损失。
TIGAS 图用 cache256 缓存；CASIA 图无缓存则即时计算 ELA + 熵图。
"""
import csv
import hashlib
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import imread_unicode
from .precompute import compute_ela, compute_entropy

T_REAL, T_SPLICE, T_COPY, T_GEN = 0, 1, 2, 3


class TypeManifestDataset(Dataset):
    def __init__(self, manifest, img_size=256, train=True, cache_root=None,
                 cache_type_dir=None):
        self.img_size = img_size
        self.train = train
        self.cache_root = cache_root
        self.cache_type_dir = cache_type_dir
        self.rows = []
        with open(manifest, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append((
                    r["path"], int(r["label"]), int(r["type"]),
                    r.get("mask", ""), int(r.get("has_mask", 0)),
                ))

    def __len__(self):
        return len(self.rows)

    def _cache_for(self, path):
        if not self.cache_root:
            return None
        p = path.replace("\\", "/")
        if "/tigas/" not in p:
            return None
        rel = p.split("/tigas/", 1)[1].split("/")
        if len(rel) < 3:
            return None
        npz = os.path.join(self.cache_root, rel[0], rel[1],
                           os.path.splitext(rel[2])[0] + ".npz")
        return npz if os.path.exists(npz) else None

    def _cache_type_for(self, path):
        """按路径 sha1 查找预计算缓存（CASIA / synthetic / TIGAS 通用）。"""
        if not self.cache_type_dir:
            return None
        h = hashlib.sha1(path.encode("utf-8")).hexdigest()
        npz = os.path.join(self.cache_type_dir, h + ".npz")
        return npz if os.path.exists(npz) else None

    def __getitem__(self, idx):
        path, label, typ, mask_path, has_mask = self.rows[idx]
        img = imread_unicode(path, cv2.IMREAD_COLOR)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.rows))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        npz = self._cache_type_for(path) or self._cache_for(path)
        if npz is not None:
            d = np.load(npz)
            ent = d["ent"].astype(np.float32)
            ela = d["ela"].astype(np.float32)
            if ela.shape[-1] != self.img_size:
                ela = cv2.resize(ela[0], (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_LINEAR)[None]
        else:
            ela = compute_ela(rgb)
            ent = compute_entropy(rgb)
            # 缓存里的熵图是 256 输入下的 31×31；未缓存图像按当前尺寸算出的是 55×55，
            # 必须归一到同一分辨率，否则同一 batch 内无法 collate。
            if ent.shape[-1] != 31:
                ent = np.stack([cv2.resize(ent[c], (31, 31), interpolation=cv2.INTER_LINEAR)
                                for c in range(ent.shape[0])])

        # 分割掩码
        if has_mask:
            if mask_path:
                m = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    m = np.zeros((self.img_size, self.img_size), np.uint8)
                else:
                    m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                mask = (m > 127).astype(np.float32)[None]
            else:
                mask = np.zeros((1, self.img_size, self.img_size), np.float32)
        else:
            mask = np.zeros((1, self.img_size, self.img_size), np.float32)

        if self.train and random.random() < 0.5:
            img = img[:, ::-1, :]
            ela = ela[:, :, ::-1]
            ent = ent[:, :, ::-1]
            mask = mask[:, :, ::-1]

        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0
        ela = torch.from_numpy(np.ascontiguousarray(ela))
        ent = torch.from_numpy(np.ascontiguousarray(ent))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return (x, ent, ela, torch.tensor(float(label)),
                torch.tensor(typ, dtype=torch.long), mask, torch.tensor(float(has_mask)))


class RandPerturbDataset(Dataset):
    """随机施加 JPEG / 缩放 / 模糊扰动，并重算 ELA 与熵图。

    用于鲁棒性微调：模型训练时见到压缩、缩放后的样本，
    避免只在高质量原图上有效。
    """

    def __init__(self, base, p=0.5, img_size=448):
        self.base = base
        self.p = p
        self.img_size = img_size

    def __len__(self):
        return len(self.base)

    def _perturb(self, img):
        h, w = img.shape[:2]
        r = random.random()
        bgr = img[:, :, ::-1]
        if r < 0.55:
            q = random.randint(50, 95)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
        elif r < 0.85:
            s = random.uniform(0.5, 1.0)
            small = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))),
                               interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
        else:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(bgr, (k, k), 0)[:, :, ::-1]
        return img

    def __getitem__(self, idx):
        x, ent, ela, y, t, mask, hm = self.base[idx]
        if random.random() >= self.p:
            return x, ent, ela, y, t, mask, hm
        img = (x.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img = self._perturb(img)
        rgb = img.astype(np.float32) / 255.0
        rgb = rgb.transpose(2, 0, 1)
        ela = compute_ela(rgb)
        ent = compute_entropy(rgb)
        if ent.shape[-1] != 31:
            ent = np.stack([cv2.resize(ent[c], (31, 31), interpolation=cv2.INTER_LINEAR)
                            for c in range(ent.shape[0])])
        x = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        ela = torch.from_numpy(np.ascontiguousarray(ela))
        ent = torch.from_numpy(np.ascontiguousarray(ent))
        return x, ent, ela, y, t, mask, hm
