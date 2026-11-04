# -*- coding: utf-8 -*-
"""记忆校准的级联推理评估（Jev-Mem 借鉴）。

流程：
1. 用 CASIA 训练集构建案例记忆（廉价特征，无需模型）；
2. CASIA 验证集上跑全模型，得到全模型分数 s_full 与系统一分数 s1
   （s1 = sigmoid(mean(noise_logit, color_logit))）；
3. 基线级联：|s1-0.5| < tau 时升级；
4. 记忆级联：用 K 个相似案例的标签把 s1 校准为 s_p，再按 |s_p-0.5| < tau 升级；
5. 可选一致性门控：邻居标签分歧大时强制升级；
6. 扫描阈值，报告升级比例 rho、AUC 与平均延迟（3 + rho*27 ms）。

用法：
  python -m fbfn.eval_memory_cascade --ckpt fbfn/runs_type/last_ema.pth --img_size 448
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .memory import CaseMemory, extract_feature
from .type_dataset import TypeManifestDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
T_SYS1 = 3.0
T_FULL = 30.0


def _load_model(ckpt, img_size, dev):
    from .type_model_large import FBFNTypeLarge
    m = FBFNTypeLarge(pretrained=False, img_size=img_size).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    m.eval()
    return m


def _is_casia(path):
    return "CASIA" in path.replace("\\", "/")


def build_memory(ds, img_size, db_path, limit=0, workers=8):
    idx = [i for i, r in enumerate(ds.rows) if _is_casia(r[0])]
    if limit:
        idx = idx[:limit]
    sub = Subset(ds, idx)
    loader = DataLoader(sub, batch_size=64, shuffle=False, num_workers=workers)
    t0 = time.time()
    feats, labels, types = [], [], []
    for x, ent, ela, y, t, mask, hm in loader:
        for i in range(x.shape[0]):
            feats.append(extract_feature(x[i].numpy(), ent[i].numpy(), ela[i].numpy()))
        labels.extend(y.numpy().astype(int).tolist())
        types.extend(t.numpy().astype(int).tolist())
    if os.path.exists(db_path):
        os.remove(db_path)
    mem = CaseMemory(db_path)
    mem.add_many(np.stack(feats), labels, types, ["casia_train"] * len(labels))
    print(f"记忆库构建完成: {len(mem)} 条，耗时 {time.time() - t0:.1f}s -> {db_path}", flush=True)
    return mem


def collect_scores(model, ds, img_size, dev, workers=8):
    idx = [i for i, r in enumerate(ds.rows) if _is_casia(r[0])]
    sub = Subset(ds, idx)
    loader = DataLoader(sub, batch_size=32, shuffle=False, num_workers=workers)
    feats, y_true, s_full = [], [], []
    with torch.no_grad():
        for x, ent, ela, y, t, mask, hm in loader:
            x, ent, ela = x.to(dev), ent.to(dev), ela.to(dev)
            out = model(x, ela=ela, entropy=ent)
            s_full.append(torch.sigmoid(out["final"]).float().cpu().numpy())
            y_true.append(y.numpy())
            xx = x.cpu().numpy(); ee = ent.cpu().numpy(); aa = ela.cpu().numpy()
            for i in range(xx.shape[0]):
                feats.append(extract_feature(xx[i], ee[i], aa[i]))
    return (np.stack(feats), np.concatenate(y_true).astype(int),
            np.concatenate(s_full))


def memory_calibrate(mem, feats, s1, k=10, lam=0.3):
    """用 K 近邻标签校准 s1，返回 (s_mem, agreement)。"""
    s_mem = np.empty_like(s1)
    agree = np.empty_like(s1)
    for i, f in enumerate(feats):
        sims, labs, _ = mem.query(f, k=k)
        if len(sims) == 0:
            s_mem[i], agree[i] = s1[i], 0.0
            continue
        w = np.clip(sims, 0, None) ** 2 + 1e-8
        lab_mean = float((w * labs).sum() / w.sum())
        signed = float((w * (2.0 * labs - 1.0)).sum() / w.sum())
        s_mem[i] = (1 - lam) * s1[i] + lam * lab_mean
        agree[i] = abs(signed)
    return s_mem, agree


def sweep(y, s1, s_full, taus, agree=None, gate=0.0):
    rows = []
    for tau in taus:
        up = np.abs(s1 - 0.5) < tau
        if agree is not None and gate > 0:
            up = up | (agree < gate)
        rho = float(up.mean())
        z = np.where(up, s_full, s1)
        rows.append((tau, rho, roc_auc_score(y, z), T_SYS1 + rho * (T_FULL - T_SYS1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_type", "last_ema.pth"))
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mem_limit", type=int, default=0, help="记忆库最多用多少训练样本（0=全部）")
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "memory", "casia_cases.sqlite"))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TypeManifestDataset(os.path.join(HERE, "type_manifest_train.csv"),
                                   args.img_size, False,
                                   os.path.join(ROOT, "data", "tigas", "cache256"),
                                   os.path.join(ROOT, "data", "cache_type"))
    val_ds = TypeManifestDataset(os.path.join(HERE, "type_manifest_val.csv"),
                                 args.img_size, False,
                                 os.path.join(ROOT, "data", "tigas", "cache256"),
                                 os.path.join(ROOT, "data", "cache_type"))

    if args.rebuild or not os.path.exists(args.db):
        mem = build_memory(train_ds, args.img_size, args.db, args.mem_limit, args.workers)
    else:
        mem = CaseMemory(args.db)
        print(f"载入记忆库 {len(mem)} 条: {args.db}", flush=True)

    # 用记忆库里的特征训练一个廉价线性系统一分类器（约 3ms 量级）
    _, F, labels, _, _ = mem._load()
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(F, labels)
    print(f"系统一线性分类器训练完成（{F.shape[0]} 条，{F.shape[1]} 维特征）", flush=True)

    model = _load_model(args.ckpt, args.img_size, dev)
    feats, y, s_full = collect_scores(model, val_ds, args.img_size, dev, args.workers)
    s1 = lr.predict_proba(feats)[:, 1]
    print(f"验证样本 {len(y)}  全模型AUC={roc_auc_score(y, s_full):.4f}  "
          f"系统一AUC={roc_auc_score(y, s1):.4f}", flush=True)

    s_mem, agree = memory_calibrate(mem, feats, s1, args.k, args.lam)
    print(f"记忆校准后系统一AUC={roc_auc_score(y, s_mem):.4f}", flush=True)

    taus = np.round(np.arange(0.0, 0.5, 0.05), 2)
    base = sweep(y, s1, s_full, taus)
    memc = sweep(y, s_mem, s_full, taus)
    gate = sweep(y, s_mem, s_full, taus, agree=agree, gate=0.2)

    print("\n tau   rho_base  AUC_base  ms_base | rho_mem  AUC_mem  ms_mem | rho_gate AUC_gate ms_gate")
    for (t, rb, ab, lb), (_, rm, am, lm), (_, rg, ag, lg) in zip(base, memc, gate):
        print(f"{t:5.2f} {rb:9.3f} {ab:9.4f} {lb:8.1f} | "
              f"{rm:7.3f} {am:8.4f} {lm:7.1f} | {rg:8.3f} {ag:8.4f} {lg:7.1f}")

    # 在相同升级比例下的对比
    print("\n[同 rho 对比] base vs mem（选最接近目标升级比例的阈值）")
    for rb_target in (0.2, 0.4, 0.6, 0.8):
        ib = int(np.argmin(np.abs(np.array([r[1] for r in base]) - rb_target)))
        im = int(np.argmin(np.abs(np.array([r[1] for r in memc]) - rb_target)))
        print(f"  目标rho~{rb_target:.1f}: base(rho={base[ib][1]:.2f}, AUC={base[ib][2]:.4f}) "
              f"mem(rho={memc[im][1]:.2f}, AUC={memc[im][2]:.4f})")


if __name__ == "__main__":
    main()
