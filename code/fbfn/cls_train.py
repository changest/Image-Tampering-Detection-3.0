"""FBFN 整图分类训练（真 vs 假）。

损失：L = BCE(final) + 0.5*BCE(att) + 0.5*(BCE(cnn)+BCE(vit)+BCE(ela)+BCE(color))
评估：Acc / AUC / AP（sklearn）。

用法示例（GenImage SD v1.4 子集）：
    python -m fbfn.cls_train \
        --real_dir <...>/stable_diffusion_v_1_4/train/nature \
        --fake_dir <...>/stable_diffusion_v_1_4/train/ai \
        --val_real_dir <...>/stable_diffusion_v_1_4/val/nature \
        --val_fake_dir <...>/stable_diffusion_v_1_4/val/ai
"""
import os
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from .cls_model import FBFNClassifier
from .cls_dataset import ImageClsDataset


def _bce(logits, labels):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def total_cls_loss(out, labels):
    """L = BCE(final) + 0.5*BCE(att) + 0.5*sum(BCE(branch))，branch 仅对保留分支求和。"""
    l_final = _bce(out["final"], labels)
    l_att = _bce(out["att"], labels)
    l_branch = sum(_bce(out[k], labels) for k in ("cnn", "vit", "ela", "color") if k in out)
    return l_final + 0.5 * l_att + 0.5 * l_branch


def _eval_scores(model, loader, device):
    """返回 (scores, labels) 两个 numpy 数组。"""
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, ent, ela, lab in loader:
            imgs = imgs.to(device)
            out = model(imgs, ela=ela.to(device), entropy=ent.to(device))
            scores.append(torch.sigmoid(out["final"]).cpu().numpy())
            labels.append(lab.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--val_real_dir", required=True)
    ap.add_argument("--val_fake_dir", required=True)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)  # Windows 下多进程 spawn 重导入 timm 反而慢
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--drop", nargs="+", default=[],
                    choices=["cnn", "vit", "noise", "color"],
                    help="消融：要去掉的分支名（可多个，如 --drop cnn vit）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_cls")
    os.makedirs(out_dir, exist_ok=True)

    train_ds = ImageClsDataset(args.real_dir, args.fake_dir, img_size=args.img_size, train=True,
                               cache_dir=args.cache_dir)
    val_ds = ImageClsDataset(args.val_real_dir, args.val_fake_dir, img_size=args.img_size, train=False,
                             cache_dir=args.cache_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.num_workers > 0,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.num_workers > 0)
    print(f"训练 {len(train_ds)} 张（真 {sum(1 for _, l in train_ds.samples if l == 0)} / "
          f"假 {sum(1 for _, l in train_ds.samples if l == 1)}），"
          f"验证 {len(val_ds)} 张", flush=True)

    model = FBFNClassifier(pretrained=True, img_size=args.img_size, drop_branches=args.drop).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"drop={args.drop} 可训练参数 {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auc = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        for imgs, ent, ela, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            ent, ela = ent.to(device), ela.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(imgs, ela=ela, entropy=ent)
                loss = total_cls_loss(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
        scheduler.step()
        train_loss = run_loss / len(train_loader)

        scores, labels = _eval_scores(model, val_loader, device)
        acc = float(((scores > 0.5).astype(float) == labels).mean())
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
        print(f"Epoch {epoch + 1}/{args.epochs} | train {train_loss:.4f} | "
              f"val Acc={acc * 100:.2f} AUC={auc * 100:.2f} AP={ap * 100:.2f} | "
              f"{time.time() - t0:.0f}s", flush=True)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(out_dir, "best_cls.pth"))

    print(f"最佳 val AUC={best_auc * 100:.2f}，保存 best_cls.pth", flush=True)


if __name__ == "__main__":
    main()
