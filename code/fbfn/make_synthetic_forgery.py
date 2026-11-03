# -*- coding: utf-8 -*-
"""程序化合成 copy-move / splicing 篡改图（含像素级掩码），扩充稀有类。

- copy-move：从同一张图复制一块，粘贴到另一处；
- splicing ：从另一张图复制一块，粘贴过来；
- 三种融合方式：hard / feather / seamlessClone；
- 输出 JPG + 二值掩码 PNG，并生成 synth_manifest_{train,val}.csv。

只做数据生成，不训练。用法：
  python -m fbfn.make_synthetic_forgery            # 默认 4000+4000 训练、600+600 验证
  python -m fbfn.make_synthetic_forgery --n_train 10000 --n_val 1000
"""
import argparse
import csv
import os
import random

import cv2
import numpy as np

from .dataset import imread_unicode

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
C1_AU = os.path.join(ROOT, "data", "CASIA", "CASIA", "CASIA 1.0 dataset", "Au")
C2_AU = os.path.join(ROOT, "data", "CASIA", "CASIA", "CASIA 2.0", "Au")
TG_REAL = os.path.join(ROOT, "data", "tigas", "train", "real")
TG_VAL = os.path.join(ROOT, "data", "tigas", "val", "real")
OUT_ROOT = os.path.join(ROOT, "data", "synthetic")

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
T_SPLICE, T_COPY = 1, 2


def imwrite_unicode(path, img, ext, params=None):
    """OpenCV 在 Windows 下无法写入含中文的路径，用 imencode + tofile。"""
    ok, buf = cv2.imencode(ext, img, params or [])
    if ok:
        buf.tofile(path)
    return ok


def list_imgs(d):
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(EXTS)]


def load(path, size):
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def transform_patch(patch, rng):
    h, w = patch.shape[:2]
    s = rng.uniform(0.6, 1.2)
    ps = max(48, int(min(h, w) * s))
    patch = cv2.resize(patch, (ps, ps), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        ang = rng.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((ps / 2, ps / 2), ang, 1.0)
        patch = cv2.warpAffine(patch, M, (ps, ps), borderMode=cv2.BORDER_REPLICATE)
    return patch


def paste(base, patch, x, y, mode, rng):
    ps = patch.shape[0]
    h, w = base.shape[:2]
    x = max(0, min(x, w - ps)); y = max(0, min(y, h - ps))
    mask = np.zeros((h, w), np.uint8)
    mask[y:y + ps, x:x + ps] = 255

    if mode == "seamless":
        try:
            center = (x + ps // 2, y + ps // 2)
            out = cv2.seamlessClone(patch, base, mask, center, cv2.NORMAL_CLONE)
            return out, mask
        except Exception:
            mode = "feather"

    roi = base[y:y + ps, x:x + ps].astype(np.float32)
    if mode == "hard":
        base[y:y + ps, x:x + ps] = patch
    else:  # feather
        f = max(4, ps // 6)
        alpha = np.ones((ps, ps), np.float32)
        ramp = np.linspace(0, 1, f)
        alpha[:f, :] *= ramp[:, None]
        alpha[-f:, :] *= ramp[::-1][:, None]
        alpha[:, :f] *= ramp[None, :]
        alpha[:, -f:] *= ramp[::-1][None, :]
        blended = roi * (1 - alpha[..., None]) + patch.astype(np.float32) * alpha[..., None]
        base[y:y + ps, x:x + ps] = np.clip(blended, 0, 255).astype(np.uint8)
    return base, mask


def gen_one(src_paths, base_size, rng, typ):
    base = load(rng.choice(src_paths), base_size)
    if base is None:
        return None, None
    if typ == T_COPY:
        patch_src = base
    else:
        p = rng.choice(src_paths)
        patch_src = load(p, base_size)
        if patch_src is None:
            return None, None
    ph, pw = patch_src.shape[:2]
    ps = rng.randint(64, 160)
    px = rng.randint(0, pw - ps); py = rng.randint(0, ph - ps)
    patch = patch_src[py:py + ps, px:px + ps].copy()
    patch = transform_patch(patch, rng)
    ps = patch.shape[0]
    # copy-move 尽量不覆盖原位置
    for _ in range(10):
        x = rng.randint(0, base_size - ps); y = rng.randint(0, base_size - ps)
        if typ != T_COPY or not (abs(x - px) < ps and abs(y - py) < ps):
            break
    mode = rng.choice(["hard", "feather", "seamless"])
    out, mask = paste(base, patch, x, y, mode, rng)
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src_all = list_imgs(C1_AU) + list_imgs(C2_AU) + list_imgs(TG_REAL)
    val_src = list_imgs(TG_VAL) or src_all[:200]
    if not src_all:
        raise SystemExit("没有找到真实图源（CASIA Au / TIGAS real）")
    rng.shuffle(src_all)
    print(f"真实图源 {len(src_all)}，验证图源 {len(val_src)}", flush=True)

    rows = []
    for split, n, srcs in (("train", args.n_train, src_all), ("val", args.n_val, val_src)):
        for typ, name in ((T_COPY, "copy_move"), (T_SPLICE, "splice")):
            d_img = os.path.join(OUT_ROOT, split, name)
            d_msk = os.path.join(OUT_ROOT, split, name + "_mask")
            os.makedirs(d_img, exist_ok=True)
            os.makedirs(d_msk, exist_ok=True)
            for i in range(n):
                out, mask = gen_one(srcs, args.size, rng, typ)
                if out is None:
                    continue
                stem = f"synth_{name}_{i:06d}"
                ip = os.path.join(d_img, stem + ".jpg")
                mp = os.path.join(d_msk, stem + "_gt.png")
                q = rng.randint(90, 98)
                imwrite_unicode(ip, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), ".jpg",
                                [cv2.IMWRITE_JPEG_QUALITY, q])
                imwrite_unicode(mp, mask, ".png")
                rows.append((ip, 1, typ, 0, mp, 1, split))
            print(f"{split}/{name}: {n} 张完成", flush=True)

    for split in ("train", "val"):
        sel = [r for r in rows if r[6] == split]
        with open(os.path.join(HERE, f"synth_manifest_{split}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "type", "has_cache", "mask", "has_mask"])
            for p, lab, typ, has, m, hm, _ in sel:
                w.writerow([p, lab, typ, has, m, hm])
        print(f"{split} 合成清单: {len(sel)}", flush=True)


if __name__ == "__main__":
    main()
