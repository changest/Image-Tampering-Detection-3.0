# -*- coding: utf-8 -*-
"""冻结骨干，仅训练轻量二分类头。

特征（每张图，9 维）：
  - 原二分类 logit
  - final_map 的 max / top-1% 均值 / 均值 / 标准差
  - 4 类 type logits
在 CASIA 训练集上拟合 LogisticRegression / MLP，在 CASIA 验证集测试。

用法：
  python -m fbfn.fit_binary_head --ckpt fbfn/runs_type/ep20_ema.pth --out fbfn/runs_type/binhead.pkl
"""
import argparse
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
import joblib

from .type_dataset import TypeManifestDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def feats(ckpt, ds, idx, dev, bs=32, workers=12):
    from .type_model_large import FBFNTypeLarge
    m = FBFNTypeLarge(pretrained=False, img_size=448).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    m.eval()
    loader = DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False, num_workers=workers)
    F, y = [], []
    with torch.no_grad():
        for x, ent, ela, yy, t, mask, hm in loader:
            x, ent, ela = x.to(dev), ent.to(dev), ela.to(dev)
            out = m(x, ela=ela, entropy=ent)
            fm = out["final_map"].float().flatten(1)
            k = max(1, fm.shape[1] // 100)
            f = torch.cat([
                out["final"].float()[:, None],
                fm.max(1).values[:, None],
                fm.topk(k, dim=1).values.mean(1)[:, None],
                fm.mean(1)[:, None],
                fm.std(1)[:, None],
                out["type"].float(),
            ], dim=1)
            F.append(f.cpu().numpy())
            y.append(yy.numpy())
    return np.concatenate(F), np.concatenate(y).astype(int)


def report(name, y, s):
    p = s if s.min() >= 0 and s.max() <= 1 else 1 / (1 + np.exp(-s))
    th = 0.5
    print(f"{name:22s} AUC={roc_auc_score(y, s):.4f}  "
          f"acc={accuracy_score(y, p >= th):.4f}  F1={f1_score(y, p >= th):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_type", "ep20_ema.pth"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "runs_type", "binhead.pkl"))
    args = ap.parse_args()

    dev = torch.device("cuda")
    tr = TypeManifestDataset(os.path.join(HERE, "type_manifest_train.csv"), 448, False,
                             os.path.join(ROOT, "data", "tigas", "cache256"),
                             os.path.join(ROOT, "data", "cache_type"))
    va = TypeManifestDataset(os.path.join(HERE, "type_manifest_val.csv"), 448, False,
                             os.path.join(ROOT, "data", "tigas", "cache256"),
                             os.path.join(ROOT, "data", "cache_type"))
    itr = [i for i, r in enumerate(tr.rows) if "CASIA" in r[0].replace("\\", "/")]
    iva = [i for i, r in enumerate(va.rows) if "CASIA" in r[0].replace("\\", "/")]
    if args.limit:
        itr = itr[: args.limit]

    print(f"提取特征：train {len(itr)} / val {len(iva)}", flush=True)
    Xtr, ytr = feats(args.ckpt, tr, itr, dev)
    Xva, yva = feats(args.ckpt, va, iva, dev)
    print("train", Xtr.shape, "val", Xva.shape, flush=True)

    report("baseline bin_logit", yva, Xva[:, 0])
    report("maxpool", yva, Xva[:, 1])

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xt, Xv = (Xtr - mu) / sd, (Xva - mu) / sd

    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xt, ytr)
    report("logreg head", yva, lr.decision_function(Xv))

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                        early_stopping=True, random_state=0).fit(Xt, ytr)
    report("mlp head", yva, mlp.predict_proba(Xv)[:, 1])

    best = mlp
    joblib.dump({"model": best, "mu": mu, "sd": sd, "ckpt": args.ckpt}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
