"""FBFN 训练：AdamW + warmup + CosineAnnealing + AMP + 早停 + 断点续训。

每 epoch 结束保存完整训练状态到 checkpoint.pt（模型/优化器/调度器/混合精度状态），
`--resume` 时自动加载并接着训。最佳模型权重存 best.pth。
"""
import os
import math
import time
import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .model import FBFN
from .losses import total_loss
from .dataset import CASIADataset


def _binary_metrics(pred, target):
    """pred/target: (B,1,H,W) 二值。返回 (mIoU, F1) 按像素聚合。"""
    pred = (pred > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    iou = (tp / (tp + fp + fn + 1e-7)).item()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = (2 * precision * recall / (precision + recall + 1e-7)).item()
    return iou, f1


def build_loaders(cfg):
    train_ds = CASIADataset(cfg.casia_root, cfg.gt_root, split="train",
                            img_size=cfg.img_size, val_ratio=cfg.val_ratio,
                            seed=cfg.seed, train=True)
    val_ds = CASIADataset(cfg.casia_root, cfg.gt_root, split="val",
                          img_size=cfg.img_size, val_ratio=cfg.val_ratio,
                          seed=cfg.seed, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def train(cfg, resume=False):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)

    train_loader, val_loader = build_loaders(cfg)
    print(f"训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}", flush=True)

    model = FBFN(pretrained=cfg.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.2f} M", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs

    def lr_at_step(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_at_step)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    start_epoch = 0
    global_step = 0

    ckpt_path = os.path.join(cfg.out_dir, "checkpoint.pt")
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val = ckpt["best_val"]
        best_epoch = ckpt["best_epoch"]
        no_improve = ckpt["no_improve"]
        model = model.to(device)
        print(f"已恢复 checkpoint：epoch {start_epoch}，step {global_step}，best val loss {best_val:.4f}",
              flush=True)

    csv_path = os.path.join(cfg.out_dir, "log.csv")
    csv_mode = "a" if (resume and start_epoch > 0) else "w"
    with open(csv_path, csv_mode, newline="") as f:
        writer = csv.writer(f)
        if csv_mode == "w":
            writer.writerow(["epoch", "train_loss", "val_loss", "val_miou", "val_f1", "lr", "sec"])

        for epoch in range(start_epoch, cfg.epochs):
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
                scheduler.step()
                run_loss += loss.item()
                global_step += 1
                if global_step % cfg.log_interval == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"  epoch {epoch+1} step {global_step} loss {loss.item():.4f} lr {lr:.2e}",
                          flush=True)

            train_loss = run_loss / steps_per_epoch

            # 验证
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            val_f1 = 0.0
            nb = 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                        out = model(imgs)
                        branches = {k: out[k] for k in ("cnn", "vit", "ela", "color")}
                        loss = total_loss(out["final"], out["att"], branches, masks)
                    val_loss += loss.item()
                    iou, f1 = _binary_metrics(torch.sigmoid(out["final"]), masks)
                    val_iou += iou
                    val_f1 += f1
                    nb += 1
            val_loss /= nb
            val_iou /= nb
            val_f1 /= nb

            lr = optimizer.param_groups[0]["lr"]
            dt = time.time() - t0
            print(f"Epoch {epoch+1}/{cfg.epochs} | train {train_loss:.4f} val {val_loss:.4f} "
                  f"mIoU {val_iou:.4f} F1 {val_f1:.4f} | lr {lr:.2e} | {dt:.0f}s", flush=True)
            writer.writerow([epoch + 1, train_loss, val_loss, val_iou, val_f1, lr, dt])
            f.flush()

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(cfg.out_dir, "best.pth"))
                print(f"  保存最佳模型 (val loss {best_val:.4f})", flush=True)
            else:
                no_improve += 1

            # 每个 epoch 结束保存断点（可随时 --resume 续训）
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
            }, ckpt_path)

            if no_improve >= cfg.early_stop_patience:
                print(f"早停：连续 {cfg.early_stop_patience} epoch 未改善，停止于 epoch {epoch+1}", flush=True)
                break

    print(f"训练完成，最佳 epoch {best_epoch}，val loss {best_val:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--resume", action="store_true", help="从 checkpoint.pt 断点续训")
    args = ap.parse_args()

    cfg = Config()
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.epochs: cfg.epochs = args.epochs
    if args.lr: cfg.lr = args.lr
    if args.no_amp: cfg.use_amp = False
    if args.no_pretrained: cfg.pretrained = False
    train(cfg, resume=args.resume)
