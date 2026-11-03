"""统一评测脚本：对任意方法的预测概率图计算像素级 mIoU / F1 / Precision / Recall / AUC。

用法示例：
    python compute_metrics.py --pred_dir <预测图目录> --gt_dir <真值掩码目录> [--th 0.5]

- 预测图：灰度 PNG（0-255）或 float32 npy（0-1）
- 真值掩码：灰度 PNG（0/255）或 npy（0/1）
- 通过文件名（去扩展名）配对 pred 与 gt；可用 --gt_suffix 指定掩码后缀，如 "_gt"
"""
import os
import argparse
import numpy as np
import cv2
from sklearn.metrics import jaccard_score, roc_auc_score, roc_curve


def load_float_map(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path).astype(np.float32)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"无法读取 {path}")
    return m.astype(np.float32) / 255.0


def load_mask(path):
    m = load_float_map(path)
    return (m > 0.5).astype(np.uint8)


def match_pairs(pred_dir, gt_dir, gt_suffix=''):
    gt_stems = {}
    for f in os.listdir(gt_dir):
        stem = os.path.splitext(f)[0]
        if gt_suffix and stem.endswith(gt_suffix):
            stem = stem[: -len(gt_suffix)]
        gt_stems[stem] = f

    pairs = []
    for f in sorted(os.listdir(pred_dir)):
        stem = os.path.splitext(f)[0]
        if stem in gt_stems:
            pairs.append(
                (os.path.join(pred_dir, f), os.path.join(gt_dir, gt_stems[stem]))
            )
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--gt_suffix', default='', help='真值掩码文件名相对图像多的后缀，如 _gt')
    ap.add_argument('--th', type=float, default=0.5, help='二值化阈值')
    args = ap.parse_args()

    pairs = match_pairs(args.pred_dir, args.gt_dir, args.gt_suffix)
    if not pairs:
        print("错误：没有匹配到任何 pred/gt 文件对，检查文件名或 --gt_suffix")
        return

    ious, f1s, precs, recs = [], [], [], []
    flat_pred_all, flat_gt_all = [], []
    # AUC 按像素抽样，避免全量拼接（数亿像素）撑爆内存
    rng = np.random.default_rng(0)
    per_img_budget = max(1, 10_000_000 // len(pairs))

    for pred_path, gt_path in pairs:
        pred = load_float_map(pred_path)
        gt = load_mask(gt_path)

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        pred_bin = (pred > args.th).astype(np.uint8)
        p = pred_bin.flatten().astype(np.uint8)
        g = gt.flatten().astype(np.uint8)

        p_auc = pred.flatten()
        g_auc = g
        if p_auc.size > per_img_budget:
            idx = rng.choice(p_auc.size, size=per_img_budget, replace=False)
            p_auc = p_auc[idx]
            g_auc = g_auc[idx]
        flat_pred_all.append(p_auc)
        flat_gt_all.append(g_auc)

        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        ious.append(inter / union if union else 0.0)

        tp = inter
        fp = np.logical_and(p, 1 - g).sum()
        fn = np.logical_and(1 - p, g).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    # 像素级 AUC（跨全部图像拼接后一次性计算）
    fp_all = np.concatenate(flat_pred_all)
    gt_all = np.concatenate(flat_gt_all)
    if gt_all.sum() == 0 or gt_all.sum() == len(gt_all):
        auc = float('nan')
    else:
        auc = roc_auc_score(gt_all, fp_all)

    n = len(pairs)
    print(f"\n评测完成：{n} 张图像，阈值 {args.th}")
    print("-" * 44)
    print(f"mIoU      : {np.mean(ious):.4f}")
    print(f"F1        : {np.mean(f1s):.4f}")
    print(f"Precision : {np.mean(precs):.4f}")
    print(f"Recall    : {np.mean(recs):.4f}")
    print(f"Pixel AUC : {auc:.4f}" if not np.isnan(auc) else "Pixel AUC : n/a")
    print("-" * 44)


if __name__ == '__main__':
    main()
