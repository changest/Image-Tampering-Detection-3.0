"""CAT-Net (CAT_full) 推理脚本：输出篡改概率图供 compute_metrics.py 评测。

复用官方 CAT_Net 模型 + jpegio 提取 JPEG DCT 系数，逻辑对齐 AbstractDataset._create_tensor。
对非 JPEG 输入（tif/bmp 等）先转成 quality=100 的 JPEG 再提取 DCT 系数，与官方 arbitrary.get_tamp 一致。

用法：
    python catnet_infer.py --model_path <CAT_full_v2.pth.tar 路径> \
                           --data_root <图像目录> --save_dir <输出目录> [--list_file ...]
"""
import os
import sys
import argparse
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import jpegio

# 把 CAT-Net 仓库根目录加入 sys.path，复用其 config / model 定义
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'methods', 'CAT-Net'))
sys.path.insert(0, REPO)

from lib.config import config
from lib.config import update_config


def build_config(cfg_path, model_path):
    class _Args:
        cfg = cfg_path
        opts = ['TEST.MODEL_FILE', model_path]
    update_config(config, _Args())
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


MAX_SIDE = 1024


def to_jpeg(path):
    """确保拿到一个 JPEG 文件路径（非 JPEG 或超尺寸先转换+缩放），返回 (jpeg_path, 是否临时文件)。"""
    im = Image.open(path)
    if im.format == 'JPEG' and max(im.size) <= MAX_SIDE:
        return path, False
    fd, tmp = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    img = im.convert('RGB')
    m = max(img.size)
    if m > MAX_SIDE:
        scale = MAX_SIDE / m
        nw, nh = int(round(img.width * scale)), int(round(img.height * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
    img.save(tmp, quality=100, subsampling=0)
    return tmp, True


def preprocess(path):
    """复刻 AbstractDataset._create_tensor（blocks=('RGB','DCTvol','qtable'), DCT_channels=1）。"""
    ori_w, ori_h = Image.open(path).size  # 原始尺寸（输出还原用）
    jpg_path, is_tmp = to_jpeg(path)
    try:
        img = np.array(Image.open(jpg_path).convert('RGB'))
        h, w = img.shape[:2]

        # JPEG DCT 系数（仅 Y 通道）+ 量化表
        jpeg = jpegio.read(jpg_path)
        coef = jpeg.coef_arrays[0]            # (r, c) int32 量化 DCT 系数
        qt = jpeg.quant_tables[jpeg.comp_info[0].quant_tbl_no].astype(np.float32)

        crop_h = -(-h // 8) * 8
        crop_w = -(-w // 8) * 8

        # 尺寸补齐到 8 的倍数
        if h < crop_h or w < crop_w:
            pad_rgb = np.full((crop_h, crop_w, 3), 127.5, dtype=np.float32)
            pad_rgb[:h, :w, :] = img
            img = pad_rgb
            pad_coef = np.zeros((crop_h, crop_w), dtype=np.float32)
            pad_coef[:coef.shape[0], :coef.shape[1]] = coef
            coef = pad_coef
        else:
            img = img.astype(np.float32)
            coef = coef.astype(np.float32)

        # RGB -> [-1,1]
        t_rgb = (torch.tensor(img.transpose(2, 0, 1), dtype=torch.float) - 127.5) / 127.5

        # DCTvol 直方图（T=20, 21 通道）
        t_coef = torch.tensor(coef, dtype=torch.float)  # (crop_h, crop_w)
        T = 20
        t_dct_vol = torch.zeros((T + 1, crop_h, crop_w))
        t_dct_vol[0] = (t_coef == 0).float()
        for i in range(1, T):
            t_dct_vol[i] = (t_coef == i).float() + (t_coef == -i).float()
        t_dct_vol[T] = (t_coef >= T).float() + (t_coef <= -T).float()

        tensor = torch.cat([t_rgb, t_dct_vol], dim=0)  # (24, crop_h, crop_w)
        qtable = torch.tensor(qt, dtype=torch.float)   # (8, 8)
        return tensor.unsqueeze(0), qtable.unsqueeze(0), (ori_h, ori_w)
    finally:
        if is_tmp:
            os.remove(jpg_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=os.path.join(REPO, 'experiments', 'CAT_full.yaml'))
    ap.add_argument('--model_path', required=True, help='CAT_full_v2.pth.tar 路径')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--list_file', default=None)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    if not args.data_root and not args.list_file:
        ap.error('必须提供 --data_root 或 --list_file 之一')

    device = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu'
    print(f"设备: {device}")

    cfg = build_config(args.cfg, args.model_path)

    from lib import models
    model = eval('models.' + cfg.MODEL.NAME + '.get_seg_model')(cfg)
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
        print("[提示] 权重按 strict=False 加载（键不完全匹配，属正常现象）")
    model.to(device)
    model.eval()

    imgs = collect_images(args.data_root, args.list_file)
    print(f"共 {len(imgs)} 张图像")
    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        for i, path in enumerate(imgs):
            try:
                x, qtable, (ori_h, ori_w) = preprocess(path)
            except Exception as e:
                print(f"[跳过] {path}: {e}")
                continue
            x = x.to(device)
            qtable = qtable.to(device)
            out = model(x, qtable)
            pred = torch.squeeze(out, 0)
            pred = F.softmax(pred, dim=0)[1]
            prob = pred.cpu().numpy()
            if prob.shape != (ori_h, ori_w):
                prob = np.asarray(Image.fromarray(
                    (prob * 255).astype(np.uint8)).resize((ori_w, ori_h),
                                                          Image.BILINEAR), dtype=np.float32) / 255.0
            out_png = (prob * 255.0).astype(np.uint8)
            out_name = os.path.splitext(os.path.basename(path))[0] + '.png'
            Image.fromarray(out_png).save(os.path.join(args.save_dir, out_name))
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(imgs)}")

    print(f"完成，预测图保存至 {args.save_dir}")


if __name__ == '__main__':
    main()
