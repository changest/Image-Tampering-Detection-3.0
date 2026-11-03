# -*- coding: utf-8 -*-
"""构建 4 分类（真实/拼接/复制移动/AI生成）+ 分割掩码的带标签清单。

只使用标签可靠的来源：
  real       : CASIA1 Au、CASIA2 Au、TIGAS train|val /real（真图用全零掩码）
  splice     : CASIA1 Tp_extracted/Tp/Sp、CASIA2 Tp_D（splicing）
  copy_move  : CASIA1 Tp_extracted/Tp/CM、CASIA2 Tp_S（copy-move）
  generated  : TIGAS train|val /fake（无掩码，跳过分割损失）

CASIA2 Tp 类型规则（官方且本副本 100% 自洽）：Tp_S=复制移动，Tp_D=拼接。

输出：type_manifest_train.csv / type_manifest_val.csv
列：path,label,type,has_cache,mask,has_mask
  label   0真/1假
  type    0真/1拼接/2复制/3生成
  has_mask 1=参与分割损失（真图为全零掩码）
"""
import os, random, csv
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
C1 = os.path.join(ROOT, "data", "CASIA", "CASIA", "CASIA 1.0 dataset")
C1GT = os.path.join(ROOT, "data", "CASIA", "CASIA", "CASIA 1.0 groundtruth")
C2 = os.path.join(ROOT, "data", "CASIA", "CASIA", "CASIA 2.0")
C2GT = os.path.join(ROOT, "data", "CASIA", "CASIA", "CASIA 2 Groundtruth")
TG = os.path.join(ROOT, "data", "tigas")

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
T_REAL, T_SPLICE, T_COPY, T_GEN = 0, 1, 2, 3


def imgs(d):
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(EXTS)]


def mask_of(p, gt_dir):
    cand = os.path.join(gt_dir, os.path.splitext(os.path.basename(p))[0] + "_gt.png")
    return cand if os.path.exists(cand) else ""


def main():
    rng = random.Random(42)
    rows = []  # (path, label, type, has_cache, mask, has_mask, split)

    # CASIA1 Au -> real（全零掩码）
    au = imgs(os.path.join(C1, "Au")); rng.shuffle(au)
    n = int(len(au) * 0.2)
    for p in au[n:]: rows.append((p, 0, T_REAL, 0, "", 1, "train"))
    for p in au[:n]: rows.append((p, 0, T_REAL, 0, "", 1, "val"))

    # CASIA1 CM / Sp
    for sub, typ, gtd in (("CM", T_COPY, "CM"), ("Sp", T_SPLICE, "Sp")):
        lst = imgs(os.path.join(C1, "Tp_extracted", "Tp", sub)); rng.shuffle(lst)
        k = max(1, int(len(lst) * 0.2))
        for split, part in (("train", lst[k:]), ("val", lst[:k])):
            for p in part:
                m = mask_of(p, os.path.join(C1GT, gtd))
                rows.append((p, 1, typ, 0, m, 1 if m else 0, split))

    # CASIA2 Au -> real（全零掩码）
    au2 = imgs(os.path.join(C2, "Au")); rng.shuffle(au2)
    n2 = int(len(au2) * 0.2)
    for p in au2[n2:]: rows.append((p, 0, T_REAL, 0, "", 1, "train"))
    for p in au2[:n2]: rows.append((p, 0, T_REAL, 0, "", 1, "val"))

    # CASIA2 Tp -> 类型（Tp_S=复制移动，Tp_D=拼接）
    tp2 = imgs(os.path.join(C2, "Tp")); rng.shuffle(tp2)
    k2 = int(len(tp2) * 0.2)
    for split, part in (("train", tp2[k2:]), ("val", tp2[:k2])):
        for p in part:
            tag = os.path.splitext(os.path.basename(p))[0].split("_")[1]
            typ = T_COPY if tag == "S" else T_SPLICE
            m = mask_of(p, C2GT)
            rows.append((p, 1, typ, 0, m, 1 if m else 0, split))

    # TIGAS：有缓存、无掩码
    for split in ("train", "val"):
        for cls, lab, typ in (("real", 0, T_REAL), ("fake", 1, T_GEN)):
            for p in imgs(os.path.join(TG, split, cls)):
                rows.append((p, lab, typ, 1, "", 0, split))

    # 合成篡改：直接扫描目录（避免依赖带绝对路径的 synth_manifest，跨机器可用）
    SYN = os.path.join(ROOT, "data", "synthetic")
    for split in ("train", "val"):
        for typ, name in ((T_COPY, "copy_move"), (T_SPLICE, "splice")):
            d_img = os.path.join(SYN, split, name)
            d_msk = os.path.join(SYN, split, name + "_mask")
            for p in imgs(d_img):
                stem = os.path.splitext(os.path.basename(p))[0]
                mp = os.path.join(d_msk, stem + "_gt.png")
                ok = os.path.exists(mp)
                rows.append((p, 1, typ, 0, mp if ok else "", 1 if ok else 0, split))

    for split in ("train", "val"):
        sel = [r for r in rows if r[6] == split]
        out = os.path.join(HERE, f"type_manifest_{split}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "type", "has_cache", "mask", "has_mask"])
            for p, lab, typ, has, m, hm, _ in sel:
                w.writerow([p, lab, typ, has, m, hm])
        c = Counter(r[2] for r in sel)
        seg = sum(1 for r in sel if r[5])
        print(f"{split}: 共 {len(sel)} | 真 {c[0]} 拼接 {c[1]} 复制 {c[2]} 生成 {c[3]} | 可分割 {seg} -> {out}")


if __name__ == "__main__":
    main()
