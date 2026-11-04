# -*- coding: utf-8 -*-
"""本地端到端演示：上传图片 → 真伪判定 + 篡改类型 + 像素级热图。

默认使用 ep30_raw.pth（二分类+分割最佳），阈值 0.18、温度 1.88 为 CASIA 校准值。
运行：
  python -m fbfn.demo_app
然后在浏览器打开 http://127.0.0.1:7860
"""
import argparse
import os

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from .dataset import imread_unicode
from .precompute import compute_ela, compute_entropy

HERE = os.path.dirname(os.path.abspath(__file__))
TYPE_NAMES = {0: "真实", 1: "拼接", 2: "复制移动", 3: "AI 生成"}


def load_model(ckpt, img_size, device):
    from .type_model_large import FBFNTypeLarge
    model = FBFNTypeLarge(pretrained=False, img_size=img_size).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.inference_mode()
def infer(model, img_rgb, img_size, device, thr, temp):
    h0, w0 = img_rgb.shape[:2]
    img = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    ela = compute_ela(rgb)
    ent = compute_entropy(rgb)
    if ent.shape[-1] != 31:
        ent = np.stack([cv2.resize(ent[c], (31, 31), interpolation=cv2.INTER_LINEAR)
                        for c in range(ent.shape[0])])
    x = torch.from_numpy(rgb[None]).to(device)
    ent_t = torch.from_numpy(ent[None]).to(device)
    ela_t = torch.from_numpy(ela[None]).to(device)

    p_cnn = model.cnn_branch(x)
    p_vit = model.vit_branch(x)
    p_ela = model.noise_branch(x, ela=ela_t)
    p_col = model.color_branch(x, entropy=ent_t)
    final_map, att = model.fusion([p_cnn, p_vit, p_ela, p_col])
    feats = torch.cat([p_cnn, p_vit, p_ela, p_col], dim=1)
    tfeat = model.type_trunk(feats)
    type_logits = model.type_fc(tfeat)
    bin_logit = model.bin_fc(tfeat).squeeze(1)

    prob = float(torch.sigmoid(bin_logit / temp).item())
    type_prob = torch.softmax(type_logits, 1)[0].cpu().numpy()
    t_idx = int(type_prob.argmax())

    heat = torch.sigmoid(final_map)[0, 0].cpu().numpy()
    heat = cv2.resize(heat, (w0, h0), interpolation=cv2.INTER_LINEAR)
    heat_u8 = (heat * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.55, colored, 0.45, 0)

    maps = []
    for name, t in (("CNN", p_cnn), ("ViT", p_vit), ("ELA/SRM", p_ela), ("颜色/熵", p_col)):
        m = torch.sigmoid(t)[0, 0].cpu().numpy()
        m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_LINEAR)
        c = cv2.applyColorMap((m * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        maps.append((name, cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))

    is_fake = prob >= thr
    verdict = "疑似篡改" if is_fake else "疑似真实"
    text = (
        f"## 判定：{verdict}\n"
        f"- 篡改概率：**{prob * 100:.1f}%**（判定阈值 {thr:.2f}）\n"
        f"- 类型预测：**{TYPE_NAMES[t_idx]}**（{type_prob[t_idx] * 100:.1f}%）；"
        f"各类概率 " + "，".join(f"{TYPE_NAMES[i]} {p * 100:.1f}%" for i, p in enumerate(type_prob)) + "\n"
        f"- 说明：概率经温度缩放（T={temp}）校准，阈值针对 CASIA 场景；换数据域需重新校准。"
    )
    return overlay, maps, text


def build(ckpt, img_size, thr, temp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, img_size, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    def run(image):
        if image is None:
            return None, [], "请上传一张图片"
        img_rgb = np.array(image.convert("RGB")) if hasattr(image, "convert") else image
        overlay, maps, text = infer(model, img_rgb, img_size, device, thr, temp)
        gallery = [(m, f"{n} 证据图") for n, m in maps]
        return overlay, gallery, text

    with gr.Blocks(title="图像篡改检测 v3.0") as demo:
        gr.Markdown("# 图像篡改检测演示（FBFN v3.0）\n"
                    "上传图片 → 输出真伪判定、篡改类型与像素级热图。")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="上传图片")
                btn = gr.Button("开始检测", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="篡改区域热图（红=更可能被篡改）")
                out_txt = gr.Markdown()
        gal = gr.Gallery(label="四分支证据图", columns=4, height=220)
        btn.click(run, inputs=inp, outputs=[out_img, gal, out_txt])
        inp.change(run, inputs=inp, outputs=[out_img, gal, out_txt])
        gr.Markdown(f"模型：`{os.path.basename(ckpt)}` ｜ 输入 {img_size}×{img_size} ｜ "
                    "设备：" + str(device))
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_type", "ep30_raw.pth"))
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--thr", type=float, default=0.18)
    ap.add_argument("--temp", type=float, default=1.88)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--inbrowser", action="store_true")
    args = ap.parse_args()
    demo = build(args.ckpt, args.img_size, args.thr, args.temp)
    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=args.inbrowser)


if __name__ == "__main__":
    main()
