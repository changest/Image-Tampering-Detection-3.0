"""FBFNClassifier 效率表：各分支参数量 / FLOPs / 单图推理延迟。

口径：
- 输入尺寸默认 256（论文口径），可用 --size 覆盖（代码训练默认 384）。
- FLOPs 用 thop（含各分支）；延迟用 GPU torch.cuda.Event（50 次均值）。
- 为隔离网络本身开销，forward 传入预计算的 ela / entropy，跳过 CPU 端的
  ELA 编解码与滑窗熵（这两项属于数据预处理，不计入网络前向）。

用法：
    python -m fbfn.profiling [--size 256]
"""
import argparse
import sys
import torch

from .cls_model import FBFNClassifier

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _params(m):
    return sum(p.numel() for p in m.parameters())


def _trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def bench(model, x, ela, entropy, iters=50, warmup=10):
    for _ in range(warmup):
        model(x, ela=ela, entropy=entropy)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        model(x, ela=ela, entropy=entropy)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = args.size

    model = FBFNClassifier(pretrained=False, img_size=img_size).to(device).eval()
    x = torch.randn(1, 3, img_size, img_size).to(device)
    ela = torch.randn(1, 1, img_size, img_size).to(device)
    entropy = torch.randn(1, 3, 47, 47).to(device)

    print(f"输入尺寸: {img_size}×{img_size}  设备: {device}")
    print("\n=== 参数量 ===")
    total = 0
    for name in ["cnn_branch", "vit_branch", "noise_branch", "color_branch"]:
        if hasattr(model, name):
            p = _params(getattr(model, name))
            total += p
            print(f"  {name:14s} {p/1e6:8.2f}M")
    fp = _params(model.fusion)
    total += fp
    print(f"  {'fusion':14s} {fp/1e6:8.2f}M   ({fp/1e3:.1f}K)")
    print(f"  {'总计':14s} {total/1e6:8.2f}M   (可训练 {_trainable(model)/1e6:.2f}M)")
    print(f"  系统一(噪声+颜色) {(_params(model.noise_branch)+_params(model.color_branch))/1e6:.2f}M   "
          f"系统二(CNN+ViT) {(_params(model.cnn_branch)+_params(model.vit_branch))/1e6:.2f}M")

    print("\n=== FLOPs (thop) ===")
    try:
        from thop import profile, clever_format
        macs, params = profile(model, inputs=(x, ela, entropy), verbose=False)
        print(f"  总计: {clever_format([macs], '%.3f')}  (thop MACs)")
    except Exception as e:
        print(f"  thop 分析失败: {type(e).__name__}: {e}")

    if device.type != "cuda":
        print("\n[跳过延迟] 无 CUDA")
        return

    print("\n=== 单图推理延迟 (GPU, ms, 50 次均值) ===")
    ms = bench(model, x, ela, entropy)
    print(f"  全模型 (四分支+融合)        : {ms:8.3f} ms")

    sys1 = FBFNClassifier(pretrained=False, img_size=img_size,
                          drop_branches=("cnn", "vit")).to(device).eval()
    ms1 = bench(sys1, x, ela, entropy)
    print(f"  仅系统一 (噪声+颜色)        : {ms1:8.3f} ms")

    sys2 = FBFNClassifier(pretrained=False, img_size=img_size,
                          drop_branches=("noise", "color")).to(device).eval()
    ms2 = bench(sys2, x, ela, entropy)
    print(f"  仅系统二 (CNN+ViT)          : {ms2:8.3f} ms")

    # 级联平均延迟：系统一始终跑，仅 ρ 比例的样本升级到全模型
    print("\n=== 级联平均延迟预估 (系统一 + ρ×升级) ===")
    for rho in (0.0, 0.2, 0.5, 0.8, 1.0):
        avg = ms1 + rho * (ms - ms1)
        speedup = ms / avg if avg > 0 else float("inf")
        print(f"  升级比例 ρ={rho:.1f}  ->  平均 {avg:8.3f} ms   (相对全模型 {speedup:.2f}×)")


if __name__ == "__main__":
    main()
