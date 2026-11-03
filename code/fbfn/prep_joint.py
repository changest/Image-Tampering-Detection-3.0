"""准备多生成器联合训练数据：下载新增生成器训练子集 + 链接 sd14 训练图到 joint 目录 + 预计算缓存。

训练池：sd14(3000+3000) + sd15_1(250+250) + sd15_2(250+250) + VQDM(500+500) + Midjourney(500+500)
留出验证：ADM / Glide / biggan（已在 <研究项目>/data/tigas/crossval/）

目录：
    <研究项目>/data/tigas/joint/{real,fake}/   <- 所有训练图（文件名加生成器前缀避免重名）
    <研究项目>/data/tigas/cache256/joint/{real,fake}/{stem}.npz
"""
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests

from .download_tigas import list_files, RES_BASE
from .precompute import compute_ela, compute_entropy

# TIGAS 数据根目录：<研究项目>/data/tigas（随项目一起移动，不再硬编码盘符）
DST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "tigas",
)
JOINT = os.path.join(DST, "joint")
CACHE = os.path.join(DST, "cache256")
IMG = 256
WORKERS = 16
NEW_GENS = [("sd15_1", 250), ("sd15_2", 250), ("VQDM", 500), ("Midjourney", 500)]


def _img_files(d):
    return [f for f in sorted(os.listdir(d))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]


def _download_renamed(path, dst_dir, prefix):
    dst = os.path.join(dst_dir, prefix + os.path.basename(path))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip"
    tmp = os.path.join(dst_dir, "_tmp_" + prefix + os.path.basename(path))
    for _ in range(3):
        try:
            r = requests.get(RES_BASE + path, timeout=180)
            r.raise_for_status()
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, dst)
            return "ok"
        except Exception:
            continue
    return "fail"


def download_new():
    for gen, n in NEW_GENS:
        for cls, sub in (("real", "0_real"), ("fake", "1_fake")):
            files = list_files(f"train/images/{gen}/{sub}")[:n]
            dst_dir = os.path.join(JOINT, cls)
            os.makedirs(dst_dir, exist_ok=True)
            prefix = gen.lower() + "_"
            ok = skip = fail = 0
            with ThreadPoolExecutor(WORKERS) as ex:
                futs = {ex.submit(_download_renamed, p, dst_dir, prefix): p for p in files}
                for fut in as_completed(futs):
                    r = fut.result()
                    ok += r == "ok"
                    skip += r == "skip"
                    fail += r == "fail"
            print(f"[{gen}/{cls}] 下载完成 ok {ok} skip {skip} fail {fail}", flush=True)


def link_sd14():
    for cls in ("real", "fake"):
        src_dir = os.path.join(DST, "train", cls)
        dst_dir = os.path.join(JOINT, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for f in _img_files(src_dir):
            dst = os.path.join(dst_dir, "sd14_" + f)
            if os.path.exists(dst):
                continue
            try:
                os.link(os.path.join(src_dir, f), dst)
            except OSError:
                shutil.copy2(os.path.join(src_dir, f), dst)
    n_real = len(_img_files(os.path.join(JOINT, "real")))
    n_fake = len(_img_files(os.path.join(JOINT, "fake")))
    print(f"[sd14] 链接完成，joint 现有 真 {n_real} / 假 {n_fake}", flush=True)


def precompute_joint():
    for cls in ("real", "fake"):
        src_dir = os.path.join(JOINT, cls)
        cache_dir = os.path.join(CACHE, "joint", cls)
        os.makedirs(cache_dir, exist_ok=True)
        files = _img_files(src_dir)
        for f in files:
            stem = os.path.splitext(f)[0]
            npz = os.path.join(cache_dir, stem + ".npz")
            if os.path.exists(npz):
                continue
            img = cv2.imread(os.path.join(src_dir, f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
            rgb = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            ela = compute_ela(rgb)
            ent = compute_entropy(rgb)
            np.savez_compressed(npz, ela=ela, ent=ent)
    print("[joint] 预计算完成", flush=True)


def main():
    download_new()
    link_sd14()
    precompute_joint()


if __name__ == "__main__":
    main()
