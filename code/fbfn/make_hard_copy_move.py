# -*- coding: utf-8 -*-
"""生成更难的 copy-move 合成样本（相对于 make_synthetic_forgery 的 simple 版）。

难度来源：
  - 大角度旋转（±45°）、更宽缩放（0.5–1.5）、可选水平翻转；
  - 颜色/亮度轻微扰动，削弱"完全复制"的易检测特征；
  - 30% 样本含 2–3 个复制区域；
  - 羽化边缘（5–15 px），可按比例混合 seamlessClone。

输出 data/synthetic_hard/{train,val}/copy_move + copy_move_mask + 清单 CSV。
用法：
  python -m fbfn.make_hard_copy_move --n_train 8000 --n_val 1000
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
OUT_ROOT = os.path.join(ROOT, "data", "synthetic_hard")

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
T_COPY = 2


def imwrite_unicode(path, img, ext, params=None):
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


def make_patch(base, rng):
    h, w = base.shape[:2]
    ps = rng.randint(int(0.08 * min(h, w)), int(0.35 * min(h, w)))
    px = rng.randint(0, w - ps)
    py = rng.randint(0, h - ps)
    patch = base[py:py + ps, px:px + ps].copy()

    scale = rng.uniform(0.5, 1.5)
    nps = max(32, int(ps * scale))
    patch = cv2.resize(patch, (nps, nps), interpolation=cv2.INTER_LINEAR)
    ang = rng.uniform(-45, 45)
    M = cv2.getRotationMatrix2D((nps / 2, nps / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(nps * (cos + sin)), int(nps * (cos + sin))
    M[0, 2] += nw / 2 - nps / 2
    M[1, 2] += nh / 2 - nps / 2
    patch = cv2.warpAffine(patch, M, (nw, nh), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        patch = patch[:, ::-1]

    alpha = rng.uniform(0.92, 1.08)
    beta = rng.randint(-10, 10)
    patch = np.clip(patch.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    k = rng.choice([0, 0, 0, 3, 5])
    if k:
        patch = cv2.GaussianBlur(patch, (k, k), 0)
    return patch, (px, py, ps)


def paste_feather(base, patch, x, y, rng):
    h, w = base.shape[:2]
    ph, pw = patch.shape[:2]
    x = max(0, min(x, w - pw)); y = max(0, min(y, h - ph))
    f = rng.randint(5, 15)
    alpha = np.ones((ph, pw), np.float32)
    if f * 2 < ph and f * 2 < pw:
        ramp = np.linspace(0, 1, f)
        alpha[:f, :] *= ramp[:, None]
        alpha[-f:, :] *= ramp[::-1][:, None]
        alpha[:, :f] *= ramp[None, :]
        alpha[:, -f:] *= ramp[::-1][None, :]
    roi = base[y:y + ph, x:x + pw].astype(np.float32)
    blended = roi * (1 - alpha[..., None]) + patch.astype(np.float32) * alpha[..., None]
    base[y:y + ph, x:x + pw] = np.clip(blended, 0, 255).astype(np.uint8)
    return x, y, pw, ph


def gen_one(src_paths, size, rng):
    base = load(rng.choice(src_paths), size)
    if base is None:
        return None, None
    mask = np.zeros((size, size), np.uint8)
    n_regions = rng.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
    for _ in range(n_regions):
        patch, (px, py, ps) = make_patch(base, rng)
        ph, pw = patch.shape[:2]
        for _ in range(20):
            x = rng.randint(0, size - pw); y = rng.randint(0, size - ph)
            if not (abs(x - px) < ps and abs(y - py) < ps):
                break
        else:
            continue
        if rng.random() < 0.25 and pw > 8 and ph > 8:
            try:
                m = np.zeros((size, size), np.uint8)
                m[y:y + ph, x:x + pw] = 255
                out = cv2.seamlessClone(patch, base, m, (x + pw // 2, y + ph // 2),
                                        cv2.NORMAL_CLONE)
                base = out
                mask[y:y + ph, x:x + pw] = 255
                continue
            except Exception:
                pass
        xx, yy, ww, hh = paste_feather(base, patch, x, y, rng)
        mask[yy:yy + hh, xx:xx + ww] = 255
    return base, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src_all = list_imgs(C1_AU) + list_imgs(C2_AU) + list_imgs(TG_REAL)
    val_src = list_imgs(TG_VAL) or src_all[:200]
    if not src_all:
        raise SystemExit("没有找到真实图源（CASIA Au / TIGAS real）")
    print(f"真实图源 {len(src_all)}，验证图源 {len(val_src)}", flush=True)

    rows = []
    for split, n, srcs in (("train", args.n_train, src_all), ("val", args.n_val, val_src)):
        d_img = os.path.join(OUT_ROOT, split, "copy_move")
        d_msk = os.path.join(OUT_ROOT, split, "copy_move_mask")
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_msk, exist_ok=True)
        for i in range(n):
            out, mask = gen_one(srcs, args.size, rng)
            if out is None:
                continue
            stem = f"hard_cm_{i:06d}"
            ip = os.path.join(d_img, stem + ".jpg")
            mp = os.path.join(d_msk, stem + "_gt.png")
            q = rng.randint(85, 98)
            imwrite_unicode(ip, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), ".jpg",
                            [cv2.IMWRITE_JPEG_QUALITY, q])
            imwrite_unicode(mp, mask, ".png")
            rows.append((ip, 1, T_COPY, 0, mp, 1, split))
        print(f"{split}/copy_move: {n} 张完成", flush=True)

    for split in ("train", "val"):
        sel = [r for r in rows if r[6] == split]
        with open(os.path.join(HERE, f"hard_cm_manifest_{split}.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "type", "has_cache", "mask", "has_mask"])
            for p, lab, typ, has, m, hm, _ in sel:
                w.writerow([p, lab, typ, has, m, hm])
        print(f"{split} 硬样本清单: {len(sel)}", flush=True)


if __name__ == "__main__":
    main()
