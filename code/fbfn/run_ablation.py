"""四分支 leave-one-out 消融实验。

流程：先评估 full（best_cls.pth）在 sd14 val 的 Acc/AUC/AP，再依次训练
w/o cnn / w/o vit / w/o noise / w/o color 四个变体（15 epoch，sd14 训练，
与 full 同协议），最后汇总成消融表并存 runs_ablation/ablation_results.json。

断点续训：某变体的 best_cls.pth 已存在则跳过训练，直接重新评估。

运行：python -m fbfn.run_ablation
"""
import os
import re
import json
import sys
import subprocess

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from .cls_model import FBFNClassifier
from .cls_dataset import ImageClsDataset
from .cls_train import _eval_scores

# Windows 控制台默认 GBK，子进程日志（tqdm/timm 进度条）可能含非 GBK 字符，统一 UTF-8
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_PROJECT = os.path.dirname(ROOT)          # <研究项目>/
TIGAS = os.path.join(_PROJECT, "data", "tigas")
REAL = os.path.join(TIGAS, "train", "real")
FAKE = os.path.join(TIGAS, "train", "fake")
VAL_REAL = os.path.join(TIGAS, "val", "real")
VAL_FAKE = os.path.join(TIGAS, "val", "fake")
CACHE = os.path.join(TIGAS, "cache256")
IMG_SIZE = 256
EPOCHS = 15

CONFIGS = [
    ("w-o-cnn", ["cnn"]),
    ("w-o-vit", ["vit"]),
    ("w-o-noise", ["noise"]),
    ("w-o-color", ["color"]),
]


def eval_ckpt(ckpt, drop):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FBFNClassifier(pretrained=False, img_size=IMG_SIZE, drop_branches=drop).to(device)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    val_ds = ImageClsDataset(VAL_REAL, VAL_FAKE, img_size=IMG_SIZE, train=False, cache_dir=CACHE)
    loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    scores, labels = _eval_scores(model, loader, device)
    acc = float(((scores > 0.5).astype(float) == labels).mean())
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    return {"acc": acc, "auc": auc, "ap": ap}


def run_one(name, drop):
    out_dir = os.path.join(HERE, "runs_ablation", name)
    ckpt = os.path.join(out_dir, "best_cls.pth")
    if os.path.exists(ckpt):
        print(f"===== {name}：已有 {ckpt}，跳过训练，直接评估 =====", flush=True)
        return eval_ckpt(ckpt, drop)

    log_path = os.path.join(HERE, "runs_ablation", name + ".log")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "fbfn.cls_train",
           "--real_dir", REAL, "--fake_dir", FAKE,
           "--val_real_dir", VAL_REAL, "--val_fake_dir", VAL_FAKE,
           "--cache_dir", CACHE, "--img_size", str(IMG_SIZE),
           "--epochs", str(EPOCHS), "--out_dir", out_dir]
    for d in drop:
        cmd += ["--drop", d]

    print(f"\n===== {name} (drop={drop}) =====", flush=True)
    best = None
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            f.write(line)
            f.flush()
            line = line.rstrip("\n")
            print(line, flush=True)
            m = re.search(r"val Acc=([\d.]+) AUC=([\d.]+) AP=([\d.]+)", line)
            if m:
                # 日志里已是百分数，除以 100 转回 0-1，与 eval_ckpt 量纲一致
                acc, auc, ap = (float(m.group(i)) / 100 for i in (1, 2, 3))
                if best is None or auc > best["auc"]:
                    best = {"acc": acc, "auc": auc, "ap": ap}
        proc.wait()
    if best is None:
        raise RuntimeError(f"{name} 训练失败，log 见 {log_path}")
    return best


def main():
    out_dir = os.path.join(HERE, "runs_ablation")
    os.makedirs(out_dir, exist_ok=True)

    full_ckpt = os.path.join(HERE, "runs_cls", "best_cls.pth")
    print(f"评估 full ({full_ckpt})...", flush=True)
    results = {"full": eval_ckpt(full_ckpt, [])}
    print(f"full: {results['full']}", flush=True)

    for name, drop in CONFIGS:
        results[name] = run_one(name, drop)
        print(f"-----> {name}: {results[name]}", flush=True)

    with open(os.path.join(out_dir, "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n===== 消融结果汇总（sd14 val）=====", flush=True)
    print(f"{'配置':14s} {'Acc':>8s} {'AUC':>8s} {'AP':>8s}", flush=True)
    for k in ["full", "w-o-cnn", "w-o-vit", "w-o-noise", "w-o-color"]:
        r = results[k]
        print(f"{k:14s} {r['acc']*100:7.2f}% {r['auc']*100:7.2f}% {r['ap']*100:7.2f}%", flush=True)


if __name__ == "__main__":
    main()
