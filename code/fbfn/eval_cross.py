"""跨生成器泛化评估：下载 TIGAS val 子集 -> 预计算熵图+ELA -> 用 best_cls.pth 推理。

对每个生成器输出同分布之外的 Acc/AUC/AP，检验 sd14 训练模型的跨库泛化能力。

用法：
    python -m fbfn.eval_cross --gens Midjourney ADM
"""
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from .cls_model import FBFNClassifier
from .cls_dataset import ImageClsDataset
from .precompute import compute_ela, compute_entropy
from .download_tigas import list_files, download_one

# TIGAS 数据根目录：<研究项目>/data/tigas（随项目一起移动，不再硬编码盘符）
DST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "tigas",
)
CROSS = os.path.join(DST, "crossval")
CACHE = os.path.join(DST, "cache256")
IMG = 256
WORKERS = 16


def download_gen(gen):
    for cls, sub in (("real", "0_real"), ("fake", "1_fake")):
        files = list_files(f"val/images/{gen}/{sub}")
        dst_dir = os.path.join(CROSS, gen, cls)
        os.makedirs(dst_dir, exist_ok=True)
        ok = skip = fail = 0
        with ThreadPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(download_one, p, dst_dir): p for p in files}
            for fut in as_completed(futs):
                r = fut.result()
                ok += r == "ok"
                skip += r == "skip"
                fail += r == "fail"
        print(f"[{gen}/{cls}] 下载完成 ok {ok} skip {skip} fail {fail}", flush=True)


def precompute_gen(gen):
    for cls in ("real", "fake"):
        src_dir = os.path.join(CROSS, gen, cls)
        cache_dir = os.path.join(CACHE, gen, cls)
        os.makedirs(cache_dir, exist_ok=True)
        for f in sorted(os.listdir(src_dir)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                continue
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
    print(f"[{gen}] 预计算完成", flush=True)


def eval_gen(model, device, gen):
    ds = ImageClsDataset(os.path.join(CROSS, gen, "real"), os.path.join(CROSS, gen, "fake"),
                         img_size=IMG, train=False, cache_dir=CACHE)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, ent, ela, lab in loader:
            imgs = imgs.to(device)
            out = model(imgs, ela=ela.to(device), entropy=ent.to(device))
            scores.append(torch.sigmoid(out["final"]).cpu().numpy())
            labels.append(lab.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    acc = float(((scores > 0.5).astype(float) == labels).mean())
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    print(f"[{gen}] 真 {int((labels == 0).sum())} / 假 {int((labels == 1).sum())} | "
          f"Acc={acc * 100:.2f} AUC={auc * 100:.2f} AP={ap * 100:.2f}", flush=True)
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+", required=True)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "runs_cls", "best_cls.pth")
    model = FBFNClassifier(pretrained=False, img_size=IMG).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"加载权重 {ckpt}，设备 {device}", flush=True)

    for gen in args.gens:
        download_gen(gen)
        precompute_gen(gen)
        eval_gen(model, device, gen)


if __name__ == "__main__":
    main()
