"""冻结 cnn/vit，快速重训修复后的 ela/color/fusion，验证融合增益。

加载旧 best.pth 的 cnn/vit 权重（架构未变），ela/color/fusion 随机初始化。
每 epoch 打印五个输出的 val F1，观察 ela/color 是否从 ~10% 提升、final 是否超过 vit。
"""
import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

from .config import Config
from .model import FBFN
from .losses import total_loss
from .dataset import CASIADataset


def _binary_metrics(pred, target):
    pred = (pred > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    iou = (tp / (tp + fp + fn + 1e-7)).item()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = (2 * precision * recall / (precision + recall + 1e-7)).item()
    return iou, f1


def main(epochs=15):
    cfg = Config()
    device = torch.device("cuda")
    model = FBFN(pretrained=False).to(device)

    ckpt = torch.load(os.path.join(cfg.out_dir, "best.pth"), map_location="cpu", weights_only=True)
    model.cnn_branch.load_state_dict(
        {k[len("cnn_branch."):]: v for k, v in ckpt.items() if k.startswith("cnn_branch.")},
        strict=True)
    model.vit_branch.load_state_dict(
        {k[len("vit_branch."):]: v for k, v in ckpt.items() if k.startswith("vit_branch.")},
        strict=True)
    for p in model.cnn_branch.parameters():
        p.requires_grad = False
    for p in model.vit_branch.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数 {sum(p.numel() for p in trainable) / 1e6:.2f}M（仅 ela/color/fusion）", flush=True)

    train_ds = CASIADataset(cfg.casia_root, cfg.gt_root, split="train",
                            img_size=cfg.img_size, val_ratio=cfg.val_ratio, seed=cfg.seed, train=True)
    val_ds = CASIADataset(cfg.casia_root, cfg.gt_root, split="val",
                          img_size=cfg.img_size, val_ratio=cfg.val_ratio, seed=cfg.seed, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                out = model(imgs)
                branches = {k: out[k] for k in ("cnn", "vit", "ela", "color")}
                loss = total_loss(out["final"], out["att"], branches, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
        train_loss = run_loss / len(train_loader)

        model.eval()
        heads = ("cnn", "vit", "ela", "color", "final")
        acc_f1 = {k: 0.0 for k in heads}
        nb = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                for k in heads:
                    _, f1 = _binary_metrics(torch.sigmoid(out[k]), masks)
                    acc_f1[k] += f1
                nb += 1
        parts = " ".join(f"{k}={acc_f1[k] / nb * 100:.1f}" for k in heads)
        print(f"Epoch {epoch + 1}/{epochs} | train {train_loss:.4f} | {parts} | {time.time() - t0:.0f}s",
              flush=True)

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "finetuned.pth"))
    print("保存 finetuned.pth", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    main(epochs=args.epochs)
