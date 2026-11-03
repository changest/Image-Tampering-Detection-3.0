"""TruFor 推理脚本（复用官方 detconfcmx 模型，输出篡改概率图供 compute_metrics.py 评测）。

对目录内所有图像跑推理，输出灰度 PNG（0-255），命名与输入 basename 一致。

用法：
    python trufor_infer.py --model_path ../weights/trufor/extracted/weights/trufor.pth.tar \
                           --data_root <图像目录> --save_dir <输出目录>
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 把 TruFor_train_test 加入 sys.path，复用其 config / model 定义
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'methods', 'TruFor', 'TruFor_train_test'))
sys.path.insert(0, REPO)

from lib.config import config
from lib.utils import get_model


def build_config(model_path, gpu):
    config.defrost()
    config.merge_from_file(os.path.join(REPO, 'lib', 'config', 'trufor_ph3.yaml'))
    config.merge_from_list(['TEST.MODEL_FILE', model_path])
    config.freeze()
    return config


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
    ap.add_argument('--model_path',
                    default='../weights/trufor/extracted/weights/trufor.pth.tar')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--list_file', default=None)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    if not args.data_root and not args.list_file:
        ap.error('必须提供 --data_root 或 --list_file 之一')

    device = 'cuda:%d' % args.gpu if (torch.cuda.is_available() and not args.cpu) else 'cpu'
    print(f"设备: {device}")

    cfg = build_config(args.model_path, args.gpu)
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model = get_model(cfg)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    imgs = collect_images(args.data_root, args.list_file)
    print(f"共 {len(imgs)} 张图像")
    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        for i, path in enumerate(imgs):
            try:
                img = np.array(Image.open(path).convert('RGB'))
                ori_h, ori_w = img.shape[:2]
                # 限制最长边，避免大图原分辨率推理在 8GB 卡上显存峰值过高导致 OOM/segfault
                max_side = 1024
                m = max(ori_h, ori_w)
                if m > max_side:
                    scale = max_side / m
                    nw, nh = int(round(ori_w * scale)), int(round(ori_h * scale))
                    img = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
                # 与官方 TestDataset 一致：RGB -> [0,1]（/256），模型内部再做 imagenet 归一化
                x = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0) / 256.0
                x = x.to(device)

                out, conf, det, _ = model(x)
                pred = torch.squeeze(out, 0)
                pred = F.softmax(pred, dim=0)[1]  # 篡改类概率 (H,W)
                prob = pred.cpu().numpy()

                # 转回原始分辨率
                if prob.shape != (ori_h, ori_w):
                    prob = np.asarray(Image.fromarray(
                        (prob * 255).astype(np.uint8)).resize((ori_w, ori_h),
                                                              Image.BILINEAR), dtype=np.float32) / 255.0

                out_png = (prob * 255.0).astype(np.uint8)
                out_name = os.path.splitext(os.path.basename(path))[0] + '.png'
                Image.fromarray(out_png).save(os.path.join(args.save_dir, out_name))
            except Exception as e:
                print(f"[跳过] {path}: {e}")
                if device != 'cpu':
                    torch.cuda.empty_cache()
                continue
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(imgs)}")

    print(f"完成，预测图保存至 {args.save_dir}")


if __name__ == '__main__':
    main()
