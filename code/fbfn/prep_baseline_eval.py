# -*- coding: utf-8 -*-
"""把 CASIA 验证集复制为 ASCII 路径，供第三方基线（TruFor 等）推理。

输出：
  <out>/images/img_000001.jpg ...
  <out>/masks/img_000001.png   （真图为全零）
  <out>/index.csv              （name,label,orig）
"""
import argparse
import csv
import os
import shutil

import cv2
import numpy as np

from .dataset import imread_unicode

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "type_manifest_val.csv"))
    ap.add_argument("--out", default=r"E:\trufor_eval")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    msk_dir = os.path.join(args.out, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "CASIA" in r["path"].replace("\\", "/"):
                rows.append(r)
    print("CASIA val rows", len(rows))

    index = []
    for i, r in enumerate(rows, 1):
        stem = f"img_{i:06d}"
        ext = os.path.splitext(r["path"])[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        dst = os.path.join(img_dir, stem + ext)
        shutil.copyfile(r["path"], dst)

        img = imread_unicode(r["path"], cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if r["mask"]:
            m = imread_unicode(r["mask"], cv2.IMREAD_GRAYSCALE)
            if m is None:
                m = np.zeros((h, w), np.uint8)
            else:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                m = ((m > 127) * 255).astype(np.uint8)
        else:
            m = np.zeros((h, w), np.uint8)
        cv2.imwrite(os.path.join(msk_dir, stem + ".png"), m)

        index.append((stem + ext, int(r["label"]), int(r["type"])))
        if i % 500 == 0:
            print("copied", i, flush=True)

    with open(os.path.join(args.out, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "label", "type"])
        w.writerows(index)
    print("done", len(index), args.out)


if __name__ == "__main__":
    main()
