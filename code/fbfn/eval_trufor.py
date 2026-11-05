# -*- coding: utf-8 -*-
"""评测 TruFor 在 CASIA 验证集上的同协议结果（图级 AUC/F1 + 像素 mIoU）。"""
import argparse
import csv
import os

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=r"E:\trufor_eval")
ap.add_argument("--pred", default=r"E:\trufor_out_full")
args = ap.parse_args()

rows = list(csv.DictReader(open(os.path.join(args.root, "index.csv"), encoding="utf-8")))
ys, scores = [], []
P, M = [], []
for r in rows:
    npz = os.path.join(args.pred, r["name"] + ".npz")
    if not os.path.exists(npz):
        continue
    d = np.load(npz)
    ys.append(int(r["label"]))
    scores.append(float(d["score"]))
    m = cv2.imread(os.path.join(args.root, "masks", os.path.splitext(r["name"])[0] + ".png"),
                   cv2.IMREAD_GRAYSCALE)
    if m is None:
        continue
    pm = d["map"].astype(np.float32)
    pm = cv2.resize(pm, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)
    P.append(pm.astype(np.float16)); M.append((m > 127).astype(np.uint8))

ys = np.array(ys); scores = np.array(scores)
print(f"samples={len(ys)} real={int((ys==0).sum())} fake={int((ys==1).sum())}")
print(f"image AUC = {roc_auc_score(ys, scores):.4f}")
for th in (0.5, 0.4, 0.6):
    pred = (scores >= th).astype(int)
    print(f"  thr={th}: acc={accuracy_score(ys,pred):.4f} F1={f1_score(ys,pred):.4f}")

best = None
for th in np.arange(0.2, 0.86, 0.05):
    tp = fp = fn = 0.0
    for p, m in zip(P, M):
        pm = (p > th)
        tp += (pm * m).sum(); fp += (pm * (1 - m)).sum(); fn += ((1 - pm) * m).sum()
    miou = tp / (tp + fp + fn + 1e-7)
    if best is None or miou > best[1]:
        best = (float(th), float(miou))
    print(f"  mIoU th={th:.2f}: {miou:.4f}")
print(f"best mIoU={best[1]:.4f} @th={best[0]:.2f}")
