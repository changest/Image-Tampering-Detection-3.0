# -*- coding: utf-8 -*-
"""测试时集成：融合两个检查点在 CASIA 验证集上的 logits。

需要先运行 calibrate.py --out 保存的 npz（含 z / zt / y / type_true）。
用法：
  python -m fbfn.ensemble --a ens_ep20_ema.npz --b ens_ep30_raw.npz
"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def best_threshold(probs, y):
    best = (0.5, -1.0, 0.0)
    for th in np.linspace(0.05, 0.95, 91):
        pred = (probs >= th).astype(int)
        acc = accuracy_score(y, pred)
        if acc > best[1]:
            best = (float(th), float(acc), float(f1_score(y, pred)))
    return best


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def report(name, probs, y):
    auc = roc_auc_score(y, probs)
    th, acc, f1 = best_threshold(probs, y)
    print(f"{name:<28} AUC={auc:.4f}  best_thr={th:.2f} acc={acc:.4f} F1={f1:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    da = np.load(args.a)
    db = np.load(args.b)
    y = da["y"].astype(int)
    assert np.array_equal(y, db["y"].astype(int)), "两个 npz 的样本顺序不一致"
    yt = da["type_true"].astype(int)
    assert np.array_equal(yt, db["type_true"].astype(int)), "类型标签不一致"

    pa, pb = sigmoid(da["z"]), sigmoid(db["z"])
    report("A 单独", pa, y)
    report("B 单独", pb, y)

    for w in (0.25, 0.4, 0.5, 0.6, 0.75):
        report(f"融合 w(A)={w:.2f}", w * pa + (1 - w) * pb, y)

    ta, tb = softmax(da["zt"]), softmax(db["zt"])
    present = sorted(set(yt.tolist()))
    print()
    for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        pred = (w * ta + (1 - w) * tb).argmax(1)
        print(f"类型融合 w(A)={w:.2f}  acc={accuracy_score(yt, pred):.4f} "
              f"macroF1={f1_score(yt, pred, average='macro', labels=present):.4f}")


if __name__ == "__main__":
    main()
