# -*- coding: utf-8 -*-
"""联合训练：分割（BCE+Dice）+ 真伪（BCE）+ 篡改类型 4 分类（加权 CE）。

支持单卡与多卡 DDP：
  单卡：python -m fbfn.type_train --epochs 30 --batch_size 32 --num_workers 8
  双卡：torchrun --nproc_per_node=2 -m fbfn.type_train --epochs 30 --batch_size 32 --num_workers 8
（batch_size 为每张卡的大小）
"""
import argparse
import copy
import datetime
import math
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from .type_dataset import TypeManifestDataset
from .type_model import FBFNType

HERE = os.path.dirname(os.path.abspath(__file__))


def _root():
    return os.path.dirname(os.path.dirname(HERE))


def _dist_info():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
    return False, 0, 1, 0


def _subset_by_class(ds, limit):
    buckets = {}
    base = ds.dataset if isinstance(ds, Subset) else ds
    rows = base.rows
    for i, r in enumerate(rows):
        buckets.setdefault(r[2], []).append(i)
    idx = []
    for t, ids in buckets.items():
        idx.extend(ids[:limit])
    return Subset(base, idx)


def _stratified_subset(ds, total):
    """按类型分层取 total 个样本（每类均分），用于每轮快速评估。"""
    base = ds.dataset if isinstance(ds, Subset) else ds
    buckets = {}
    for i, r in enumerate(base.rows):
        buckets.setdefault(r[2], []).append(i)
    per = max(1, total // max(1, len(buckets)))
    idx = []
    for t, ids in buckets.items():
        idx.extend(ids[:per])
    return Subset(base, idx)


class ModelEma:
    """权重指数滑动平均，通常比原始权重更稳、更准。"""

    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.module.parameters(), model.parameters()):
            e.mul_(self.decay).add_(m.detach(), alpha=1.0 - self.decay)
        for e, m in zip(self.module.buffers(), model.buffers()):
            e.copy_(m)


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1 - num / den


def evaluate(model, loader, device):
    model.eval()
    ys, ps, yt, pt = [], [], [], []
    tp = fp = fn = 0.0
    with torch.no_grad():
        for x, ent, ela, y, t, mask, hm in loader:
            x, ent, ela = x.to(device), ent.to(device), ela.to(device)
            mask, hm = mask.to(device), hm.to(device)
            out = model(x, ela=ela, entropy=ent)
            ps.append(torch.sigmoid(out["final"]).cpu().numpy())
            pt.append(out["type"].argmax(1).cpu().numpy())
            ys.append(y.numpy()); yt.append(t.numpy())
            pm = (torch.sigmoid(out["final_map"]) > 0.5).float()
            sel = hm.bool()
            if sel.any():
                p_ = pm[sel]; m_ = mask[sel]
                tp += (p_ * m_).sum().item()
                fp += (p_ * (1 - m_)).sum().item()
                fn += ((1 - p_) * m_).sum().item()
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    yt, pt = np.concatenate(yt), np.concatenate(pt)
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    acc = (pt == yt).mean()
    f1 = f1_score(yt, pt, average="macro")
    cm = confusion_matrix(yt, pt, labels=[0, 1, 2, 3])
    miou = tp / (tp + fp + fn + 1e-7)
    sf1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
    return auc, acc, f1, cm, miou, sf1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "type_manifest_train.csv"))
    ap.add_argument("--val_manifest", default=os.path.join(HERE, "type_manifest_val.csv"))
    ap.add_argument("--cache_root", default=os.path.join(_root(), "data", "tigas", "cache256"))
    ap.add_argument("--cache_type_dir", default=os.path.join(_root(), "data", "cache_type"))
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--model", choices=["small", "large"], default="small")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=24)
    ap.add_argument("--eval_limit", type=int, default=1600,
                    help="每轮评估用的分层验证样本数（0=全量）")
    ap.add_argument("--eval_every", type=int, default=1,
                    help="每 N 轮评估一次；0=训练中不评估（DDP 下最稳）")
    ap.add_argument("--save_every", type=int, default=5,
                    help="每 N 轮保存一个快照 epN_raw.pth")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--w_seg", type=float, default=1.0)
    ap.add_argument("--w_bin", type=float, default=0.5)
    ap.add_argument("--w_type", type=float, default=0.5)
    ap.add_argument("--w_att", type=float, default=0.1)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--type_boost", default="",
                    help="类型类别权重倍率，如 2:1.5,1:1.2（类别:倍率，逗号分隔）")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--resume", default="", help="从指定 raw 权重续训")
    ap.add_argument("--resume_ema", default="", help="配套 EMA 权重（可选）")
    ap.add_argument("--start_epoch", type=int, default=0,
                    help="续训起始 epoch（已完成轮数，LR/epoch 编号按绝对轮次计算）")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "runs_type"))
    args = ap.parse_args()

    distributed, rank, world, local_rank = _dist_info()
    if distributed:
        dist.init_process_group(backend="nccl",
                                timeout=datetime.timedelta(hours=2))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"distributed={distributed} world={world} device={device}", flush=True)

    tr = TypeManifestDataset(args.manifest, args.img_size, True, args.cache_root, args.cache_type_dir)
    va = TypeManifestDataset(args.val_manifest, args.img_size, False, args.cache_root, args.cache_type_dir)
    if args.limit:
        tr = _subset_by_class(tr, args.limit)
        va = _subset_by_class(va, args.limit)
    if rank == 0:
        print(f"train {len(tr)} / val {len(va)}", flush=True)

    if distributed:
        tr_sampler = DistributedSampler(tr, num_replicas=world, rank=rank, shuffle=True)
        va_sampler = DistributedSampler(va, num_replicas=world, rank=rank, shuffle=False)
    else:
        tr_sampler = va_sampler = None

    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=(tr_sampler is None),
                    sampler=tr_sampler, num_workers=args.num_workers, pin_memory=True,
                    persistent_workers=args.num_workers > 0)
    # 验证集只在 rank0 用全量（不分片）
    va_eval = _stratified_subset(va, args.eval_limit) if args.eval_limit else va
    vl = DataLoader(va_eval, batch_size=args.eval_batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    base = tr.dataset if isinstance(tr, Subset) else tr
    types = [r[2] for r in base.rows]
    cnt = Counter(types)
    w = torch.tensor([len(types) / (4 * cnt.get(k, 1)) for k in range(4)],
                     dtype=torch.float32, device=device)
    if args.type_boost:
        for kv in args.type_boost.split(","):
            k, v = kv.split(":")
            w[int(k)] *= float(v)
    if rank == 0:
        print("type counts", dict(cnt), "weights", [round(v, 2) for v in w.tolist()], flush=True)

    if args.model == "large":
        from .type_model_large import FBFNTypeLarge
        model = FBFNTypeLarge(pretrained=not args.no_pretrained, img_size=args.img_size).to(device)
    else:
        model = FBFNType(pretrained=not args.no_pretrained, img_size=args.img_size).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smooth)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    raw_model = model.module if distributed else model
    if args.resume:
        raw_model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
        if rank == 0:
            print(f"resumed raw from {args.resume}", flush=True)
    ema = ModelEma(raw_model, args.ema_decay)
    if args.resume_ema:
        ema.module.load_state_dict(torch.load(args.resume_ema, map_location="cpu", weights_only=True))
        if rank == 0:
            print(f"resumed ema from {args.resume_ema}", flush=True)

    def set_lr(epoch):
        if epoch <= args.warmup_epochs:
            lr = args.lr * epoch / max(1, args.warmup_epochs)
        else:
            p = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * p))
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    best_f1, best_path = -1, os.path.join(args.out_dir, "best_type.pth")
    for ep in range(args.start_epoch + 1, args.epochs + 1):
        model.train()
        if distributed:
            tr_sampler.set_epoch(ep)
        cur_lr = set_lr(ep)
        run = 0.0
        for i, (x, ent, ela, y, t, mask, hm) in enumerate(tl):
            x, ent, ela = x.to(device), ent.to(device), ela.to(device)
            y, t, mask, hm = y.to(device), t.to(device), mask.to(device), hm.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(x, ela=ela, entropy=ent)
                loss = (args.w_bin * bce(out["final"], y)
                        + args.w_att * bce(out["att"], y)
                        + args.w_type * ce(out["type"], t))
                sel = hm.bool()
                if sel.any():
                    seg_bce = F.binary_cross_entropy_with_logits(
                        out["final_map"][sel], mask[sel], reduction="none").mean(dim=(1, 2, 3))
                    seg = seg_bce.mean() + dice_loss(out["final_map"][sel], mask[sel]).mean()
                else:
                    # 保证 final_map 分支在每步都参与计算图，避免 DDP 未用参数报错
                    seg = out["final_map"].sum() * 0.0
                loss = loss + args.w_seg * seg
            loss = loss / args.accum
            scaler.scale(loss).backward()
            run += loss.item() * args.accum
            if (i + 1) % args.accum == 0 or (i + 1) == len(tl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                ema.update(model.module if distributed else model)
            if rank == 0 and (i + 1) % 50 == 0:
                print(f"  ep{ep} it{i+1}/{len(tl)} loss={run/(i+1):.4f} lr={cur_lr:.2e}", flush=True)

        if rank == 0:
            # 每轮保存 last，防崩溃丢失
            torch.save(raw_model.state_dict(), os.path.join(args.out_dir, "last_raw.pth"))
            torch.save(ema.module.state_dict(), os.path.join(args.out_dir, "last_ema.pth"))
            if args.save_every and ep % args.save_every == 0:
                torch.save(raw_model.state_dict(),
                           os.path.join(args.out_dir, f"ep{ep}_raw.pth"))
                torch.save(ema.module.state_dict(),
                           os.path.join(args.out_dir, f"ep{ep}_ema.pth"))
                print(f"  snapshot ep{ep} saved", flush=True)

            if args.eval_every and ep % args.eval_every == 0:
                m_raw = evaluate(raw_model, vl, device)
                m_ema = evaluate(ema.module, vl, device)

                def fmt(m):
                    return (f"AUC={m[0]:.4f} typeAcc={m[1]:.4f} macroF1={m[2]:.4f} "
                            f"seg_mIoU={m[4]:.4f} seg_F1={m[5]:.4f}")

                print(f"[ep{ep}] loss={run/len(tl):.4f}", flush=True)
                print("  RAW " + fmt(m_raw), flush=True)
                print("  EMA " + fmt(m_ema), flush=True)
                use_ema = m_ema[2] >= m_raw[2]
                m = m_ema if use_ema else m_raw
                print(f"  confusion({'EMA' if use_ema else 'RAW'}):\n", m[3], flush=True)
                if m[2] > best_f1:
                    best_f1 = m[2]
                    state = ema.module.state_dict() if use_ema else raw_model.state_dict()
                    torch.save(state, best_path)
                    print(f"  saved {best_path} ({'ema' if use_ema else 'raw'})", flush=True)
            else:
                print(f"[ep{ep}] loss={run/len(tl):.4f} (no eval)", flush=True)

    if rank == 0:
        print("done, best macroF1", best_f1, flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
