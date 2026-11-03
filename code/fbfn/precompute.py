"""预计算熵图 + ELA 缓存，加速训练（这些是固定预处理，无需每 epoch 重算）。

对 <研究项目>/data/tigas/{train,val}/{real,fake} 每张图：
  - ELA: |I - JPEG(I, Q=90)| 均值 -> (1,256,256) float32
  - 熵图: Lab 归一化 + 16x16 滑窗直方图熵 -> (3,31,31) float32
存到 <研究项目>/data/tigas/cache256/{train,val}/{real,fake}/{stem}.npz（断点续传）。
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch

from .branches import _rgb_to_lab, _local_entropy_np

# TIGAS 数据根目录：<研究项目>/data/tigas（随项目一起移动，不再硬编码盘符）
DST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "tigas",
)
CACHE = os.path.join(DST, "cache256")
IMG = 256
WORKERS = 8


def compute_ela(rgb):
    """rgb: (3,H,W) float32 [0,1] -> (1,H,W) float32"""
    x_np = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    bgr = x_np.transpose(1, 2, 0)[:, :, ::-1]
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    comp = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    comp = comp.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.abs(rgb - comp).mean(axis=0, keepdims=True)


def compute_entropy(rgb):
    """rgb: (3,H,W) float32 [0,1] -> (3,47,47) float32"""
    x = torch.from_numpy(rgb[None])  # (1,3,H,W)
    lab = _rgb_to_lab(x)
    L = lab[:, 0:1] / 100.0
    a = (lab[:, 1:2] + 128.0) / 255.0
    b = (lab[:, 2:3] + 128.0) / 255.0
    lab_norm = torch.cat([L, a, b], dim=1)
    ent = _local_entropy_np(lab_norm.numpy())  # (1,3,47,47)
    return ent[0]


def process_one(src, cache_path):
    if os.path.exists(cache_path):
        return "skip"
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return "bad"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    rgb = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    ela = compute_ela(rgb)
    ent = compute_entropy(rgb)
    np.savez_compressed(cache_path, ela=ela, ent=ent)
    return "ok"


def main():
    tasks = []
    for split in ("train", "val"):
        for sub in ("real", "fake"):
            src_dir = os.path.join(DST, split, sub)
            cache_dir = os.path.join(CACHE, split, sub)
            os.makedirs(cache_dir, exist_ok=True)
            for f in sorted(os.listdir(src_dir)):
                if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    continue
                stem = os.path.splitext(f)[0]
                tasks.append((os.path.join(src_dir, f), os.path.join(cache_dir, stem + ".npz")))

    print(f"待处理 {len(tasks)} 张", flush=True)
    ok = skip = bad = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(process_one, s, c): s for s, c in tasks}
        for i, fut in enumerate(futs, 1):
            r = fut.result()
            if r == "ok":
                ok += 1
            elif r == "skip":
                skip += 1
            else:
                bad += 1
            if i % 500 == 0:
                print(f"  进度 {i}/{len(tasks)} (ok {ok} skip {skip} bad {bad})", flush=True)
    print(f"完成：ok {ok}，skip {skip}，bad {bad}", flush=True)


if __name__ == "__main__":
    main()
