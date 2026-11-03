"""MVSS-Net 推理脚本（去掉官方代码里的 apex 依赖，纯 PyTorch）。

对目录内所有图像跑推理，输出每张图的篡改概率图（灰度 PNG，0-255），
命名与输入图像 basename 一致，供 compute_metrics.py 直接评测。

用法：
    python mvss_infer.py --model_path ../weights/mvss/mvssnetplus_casia.pt \
                         --data_root <图像目录> --save_dir <输出目录>
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch

# 把 MVSS-Net 仓库加入 sys.path，复用其模型定义
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'methods', 'MVSS-Net'))
sys.path.insert(0, REPO)
from models.mvssnet import get_mvss  # noqa: E402


def preprocess(img_bgr, size):
    img = cv2.resize(img_bgr, (size, size))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def collect_images(data_root, list_file):
    if list_file:
        with open(list_file) as f:
            return [l.strip() for l in f if l.strip()]
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    return [
        os.path.join(data_root, f)
        for f in sorted(os.listdir(data_root))
        if f.lower().endswith(exts)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', default='../weights/mvss/mvssnetplus_casia.pt')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--list_file', default=None)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    if not args.data_root and not args.list_file:
        ap.error('必须提供 --data_root 或 --list_file 之一')

    device = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu'
    print(f"设备: {device}")

    model = get_mvss(backbone='resnet50', pretrained_base=True,
                     nclass=1, sobel=True, constrain=True, n_input=3)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        for k in ('model_dict', 'state_dict'):
            if k in checkpoint:
                checkpoint = checkpoint[k]
                break
    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError:
        # 检查点含训练用的图像级分类头（clf_fc/clf_pool），定位推理用不到，忽略即可
        model.load_state_dict(checkpoint, strict=False)
        print("[提示] 权重含辅助分类头，已按 strict=False 加载（正常现象）")
    model.to(device)
    model.eval()

    imgs = collect_images(args.data_root, args.list_file)
    print(f"共 {len(imgs)} 张图像")

    os.makedirs(args.save_dir, exist_ok=True)
    with torch.no_grad():
        for i, path in enumerate(imgs):
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                print(f"[跳过] 无法读取 {path}")
                continue
            ori_h, ori_w = img_bgr.shape[:2]
            x = preprocess(img_bgr, args.size).to(device)
            _, seg = model(x)  # seg = x0 logits (1,1,H,W)
            prob = torch.sigmoid(seg)[0, 0].cpu().numpy()
            prob = cv2.resize(prob, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
            out = (prob * 255.0).astype(np.uint8)
            out_name = os.path.splitext(os.path.basename(path))[0] + '.png'
            cv2.imwrite(os.path.join(args.save_dir, out_name), out)
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(imgs)}")

    print(f"完成，预测图保存至 {args.save_dir}")


if __name__ == '__main__':
    main()
