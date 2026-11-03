# -*- coding: utf-8 -*-
"""按清单预计算 ELA + 熵图缓存（加速训练，断点续算）。

缓存按图片绝对路径的 sha1 命名：<out_dir>/<sha1>.npz，含 ela(1,H,W)、ent(3,h,w)。
用法：
  python -m fbfn.precompute_manifest --workers 16
"""
import argparse
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

from .dataset import imread_unicode
from .precompute import compute_ela, compute_entropy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IMG = 256


def cache_path(path, out_dir):
    h = hashlib.sha1(path.encode("utf-8")).hexdigest()
    return os.path.join(out_dir, h + ".npz")


def work(path, out_dir, img_size):
    # TIGAS 已有 cache256，数据集会回退使用，无需再算
    if "/tigas/" in path.replace("\\", "/"):
        return "skip"
    cp = cache_path(path, out_dir)
    if os.path.exists(cp):
        return "skip"
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        return "bad"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    ela = compute_ela(rgb)
    ent = compute_entropy(rgb)
    np.savez_compressed(cp, ela=ela.astype(np.float32), ent=ent.astype(np.float32))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", default=[
        os.path.join(HERE, "type_manifest_train.csv"),
        os.path.join(HERE, "type_manifest_val.csv")])
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data", "cache_type"))
    ap.add_argument("--img_size", type=int, default=IMG)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import csv
    paths = []
    for m in args.manifests:
        with open(m, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                paths.append(r["path"])
    paths = sorted(set(paths))
    todo = [p for p in paths if not os.path.exists(cache_path(p, args.out_dir))]
    print(f"共 {len(paths)} 张，待计算 {len(todo)}，workers={args.workers}", flush=True)

    ok = skip = bad = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, p, args.out_dir, args.img_size): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception:
                r = "fail"
            ok += r == "ok"; skip += r == "skip"; bad += r == "bad"; fail += r == "fail"
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}  ok={ok} skip={skip} bad={bad} fail={fail}", flush=True)
    print(f"完成: ok={ok} skip={skip} bad={bad} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
