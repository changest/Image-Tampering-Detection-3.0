# -*- coding: utf-8 -*-
"""推理速度基准：fp32 / bf16 / fp16 / channels_last，可选 torch.compile。

用法：
  python -m fbfn.bench_infer --ckpt fbfn/runs_type/last_ema.pth --img_size 448
"""
import argparse
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def make_inputs(b, size, dev, dtype):
    x = torch.randn(b, 3, size, size, device=dev, dtype=dtype)
    ent = torch.randn(b, 3, 31, 31, device=dev, dtype=dtype)
    ela = torch.randn(b, 1, size, size, device=dev, dtype=dtype)
    return x, ent, ela


def bench(model, b, size, dev, dtype, use_autocast=False, iters=30):
    x, ent, ela = make_inputs(b, size, dev, dtype)
    with torch.inference_mode():
        for _ in range(8):
            if use_autocast:
                with torch.autocast("cuda", dtype=dtype):
                    model(x, ela=ela, entropy=ent)
            else:
                model(x, ela=ela, entropy=ent)
        torch.cuda.synchronize()
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(iters):
            if use_autocast:
                with torch.autocast("cuda", dtype=dtype):
                    model(x, ela=ela, entropy=ent)
            else:
                model(x, ela=ela, entropy=ent)
        en.record(); torch.cuda.synchronize()
        ms = st.elapsed_time(en) / iters
    return ms / b, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_type", "last_ema.pth"))
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    from .type_model_large import FBFNTypeLarge
    model = FBFNTypeLarge(pretrained=False, img_size=args.img_size).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    model.eval()

    print(f"GPU={torch.cuda.get_device_name(0)} batch={args.batch} res={args.img_size}")
    print(f"{'mode':22s} {'ms/img':>8s} {'ms/batch':>9s}")
    for name, dtype, ac, cl in [
        ("fp32", torch.float32, False, False),
        ("bf16 autocast", torch.bfloat16, True, False),
        ("fp16 autocast", torch.float16, True, False),
        ("fp32 channels_last", torch.float32, False, True),
        ("bf16 channels_last", torch.bfloat16, True, True),
    ]:
        m = model
        if cl:
            m = model.to(memory_format=torch.channels_last)
        per, total = bench(m, args.batch, args.img_size, dev, dtype, ac)
        print(f"{name:22s} {per:8.2f} {total:9.2f}", flush=True)

    if args.compile:
        print("compiling (may take minutes)...", flush=True)
        cm = torch.compile(model, mode="max-autotune")
        for name, dtype, ac in [("compiled fp32", torch.float32, False),
                                ("compiled bf16", torch.bfloat16, True)]:
            per, total = bench(cm, args.batch, args.img_size, dev, dtype, ac, iters=20)
            print(f"{name:22s} {per:8.2f} {total:9.2f}", flush=True)


if __name__ == "__main__":
    main()
