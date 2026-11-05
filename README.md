# 图像篡改检测与定位研究（FBFN v3.0）

四分支注意力融合网络 FBFN：CNN / ViT / 噪声(ELA+SRM) / 颜色(Lab 熵) 四路证据，
经通道-空间双重注意力融合，同时输出**真伪判定、篡改类型、像素级定位**。
v3.0 使用 DINOv2 ViT-L/14（448×448），并加入 JPEG/缩放/模糊鲁棒训练。

- 代码仓库：https://github.com/changest/Image-Tampering-Detection-3.0 （分支 `v3.0`）
- 最优权重：GitHub Release `v3.0-checkpoints`
- 在线演示（局域网）：http://172.19.71.5:7860 （A40 · bf16）

## 目录结构

```
图像篡改定位研究/
├── code/
│   ├── fbfn/                  # 主流程
│   │   ├── type_model_large.py    # v3 大骨干多任务模型
│   │   ├── type_train.py          # 多任务训练（支持 resume / 增强 / 类别权重）
│   │   ├── type_dataset.py        # 数据与在线扰动
│   │   ├── eval_real.py           # 纯真实 CASIA 评测（支持 --amp）
│   │   ├── calibrate.py           # 阈值/温度校准
│   │   ├── robust_eval.py         # 鲁棒性评测（JPEG/缩放/模糊/截图）
│   │   ├── eval_res_cascade.py    # 分辨率级联实验
│   │   ├── memory.py              # 案例记忆（Jev 借鉴，组件保留）
│   │   ├── demo_app.py            # Gradio 演示（支持 bf16/fp16、局域网）
│   │   ├── bench_infer.py         # 推理速度基准
│   │   ├── make_hard_copy_move.py # 困难复制移动合成
│   │   └── runs_type/             # 本地最优权重（ep20/ep30/last_ema）
│   ├── baselines/             # 基线代码与权重
│   └── experiments/           # 第三方方法（各自独立仓库）
├── data/                      # CASIA / TIGAS / 合成数据（不入库）
├── results/                   # 指标汇总、训练与评测日志
├── paper/                     # 论文（md/docx/tex/pdf）、答辩 PPTX、瑞士风 PDF
├── presentation/              # PPTX 生成脚本与导出
├── presentation_web/          # 瑞士风网页 PPT（index.html + PDF）
├── docs/                      # 参考论文
└── methods/                   # 第三方方法源码（不入库）
```

## 关键结果（纯真实 CASIA 验证集 2,866 张）

| 任务 | 权重 | 指标 |
|---|---|---|
| 真伪二分类 | `ep30_raw` / `last_ema` | AUC 0.950–0.959；阈值 0.18 时准确率 89.95% |
| 类型（4 类） | `ep20_ema` | 准确率 86.4%，macro-F1 0.849 |
| 像素级定位 | `ep30_raw` | mIoU 0.584，分割 F1 0.737 |
| 鲁棒性（JPEG q90） | `last_ema` | AUC 0.888（增强前 0.850） |
| 推理（A40） | `last_ema` · bf16 | 模型 34 ms，端到端 81 ms；batch8 摊销 21.7 ms/图 |

详见 `results/README.md`、`results/robustness.md`。

## 快速开始

```bash
cd code

# 纯真实 CASIA 评测（可选 --amp bf16）
python -m fbfn.eval_real --model large --img_size 448 \
  --ckpt fbfn/runs_type/last_ema.pth --amp bf16

# 演示（本机 / 服务器）
python -m fbfn.demo_app --ckpt fbfn/runs_type/last_ema.pth --dtype bf16 --host 0.0.0.0

# 推理速度基准
python -m fbfn.bench_infer --ckpt fbfn/runs_type/last_ema.pth --batch 8
```

训练：

```bash
torchrun --nproc_per_node=2 -m fbfn.type_train \
  --model large --img_size 448 --epochs 40 --batch_size 12 --accum 4 \
  --aug_perturb 0.5 --save_every 5
```

## 注意事项

1. **中文路径**：OpenCV 读图已用 `np.fromfile + cv2.imdecode` 兼容 Windows 中文路径。
2. **输入尺寸**：DINOv2 patch=14，尺寸需为 14 的倍数；CNN 分支要求 32 的倍数，
   因此可用 224 / 448（两者同时满足）。
3. **ELA/熵缓存**：`data/cache_type` 按路径 sha1 缓存；新增数据需先运行
   `python -m fbfn.precompute_manifest --manifests <csv>`。
4. **权重不入库**：`.gitignore` 排除所有 `*.pth/*.pt`，权重通过 Release 分发。

## 许可证与致谢

研究代码仅供本项目使用；第三方基线分别遵循其原始许可证。
