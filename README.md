# 图像篡改定位研究（FBFN）

四分支注意力融合网络（FBFN）：CNN / ViT / 噪声(ELA+SRM) / 颜色(Lab 熵) 四路证据，
经通道-空间双重注意力融合，输出像素级篡改定位图；分类版（`cls_model.py`）做整图真伪判定。

## 目录结构

```
图像篡改定位研究/
├── code/
│   ├── fbfn/            # 主流程：模型、训练、评估、消融
│   │   ├── model.py         # FBFN 分割版
│   │   ├── cls_model.py     # FBFN 分类版（整图真伪）
│   │   ├── branches.py      # 四个分支
│   │   ├── fusion.py        # 双重注意力融合
│   │   ├── dataset.py       # CASIA 分割数据集
│   │   ├── cls_dataset.py   # 分类数据集（TIGAS/GenImage）
│   │   ├── train.py         # 分割训练
│   │   ├── cls_train.py     # 分类训练
│   │   ├── evaluate.py      # CASIA 分割评估
│   │   ├── eval_cross.py    # 跨生成器评估
│   │   ├── precompute.py    # 熵图 + ELA 缓存预计算
│   │   └── runs_cls/best_cls.pth  # 当前架构分类权重（可用）
│   ├── baselines/       # 基线：CNNDetection / LOTA / ResNet50 / UnivFD
│   └── experiments/     # 基线方法的代码与权重（TruFor/CAT-Net/MVSS 等）
├── data/
│   ├── CASIA/           # CASIA 1.0 / 2.0 + Groundtruth（主流程用 2.0）
│   └── tigas/           # TIGAS：crossval 测试集、cache256 预计算缓存、joint 联合训练
├── methods/CAT-Net/     # CAT-Net 基线（代码 + 预训练权重）
├── paper/               # 中英文论文（md / tex + 图）
└── docs/                # 参考论文 PDF
```

## 环境

- Python 3.10+，PyTorch 2.x + CUDA
- 依赖：`timm torchvision opencv-python scikit-learn pandas numpy scipy`

## 复现跨生成器结果（已跑通）

```bash
cd code
python -m fbfn.eval_cross --gens sd15_1 sd15_2 Glide Midjourney ADM biggan VQDM wuk
```

参考结果（`runs_cls/best_cls.pth`）：SD 家族 >99%，biggan ≈54（接近随机），平均 ≈80。

## 训练

```bash
# 分割（CASIA 2.0）
cd code && python -m fbfn.train --epochs 100

# 分类（TIGAS / GenImage）
python -m fbfn.cls_train --real_dir <train/real> --fake_dir <train/fake> \
  --val_real_dir <val/real> --val_fake_dir <val/fake> --cache_dir ../data/tigas/cache256
```

## 注意事项

1. **中文路径**：OpenCV 在 Windows 读不了含中文的路径，已在 `dataset.py` / `cls_dataset.py`
   用 `np.fromfile + cv2.imdecode` 修复。
2. **权重版本**：`runs/best.pth`（旧 MobileNetV3 版 ColorBranch）与当前代码不兼容，已删除；
   可用的分类权重是 `runs_cls/best_cls.pth`。分割版需用当前代码重新训练。
3. **DINOv2 类骨干**：patch=14，图像尺寸需为 14 的倍数（如 392 / 448），不要用 384。

## 2026-11-01 清理记录

已删除（约 4.5G）：`data/tigas/cache`（旧缓存）、`weights/RGB_only_v2.tar`、
`weights/sdv4_scaling_patch32.pth`（均无代码引用）、`runs/best.pth`（旧架构）、
`runs_ablation/`、`code/experiments/datasets/`、临时脚本与 `__pycache__`。

保留：主流程代码、`runs_cls/best_cls.pth`、CASIA 2.0、TIGAS crossval/cache256/joint、
基线（baselines / experiments / CAT-Net）、论文 md/tex。
