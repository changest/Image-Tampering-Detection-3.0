# 同协议基线对比（TruFor）

## 设置

- 测试集：本项目 CASIA 验证集 2,866 张（真 1,658 / 假 1,208），与本文完全相同的图片列表。
- 基线：TruFor 官方权重（CVPR 2023），输入缩放到 512，官方 `test.py` 推理。
- 指标：图级 AUC、阈值 0.5 的准确率/F1；像素 mIoU（将预测图缩放到原掩码尺寸后扫描阈值）。

## 结果

| 方法 | 图级 AUC | 准确率 | F1 | mIoU (th=0.5) | mIoU (最优阈值) |
|---|---|---|---|---|---|
| TruFor | 0.9463 | 88.76% | 85.75% | 0.5854 | **0.6653**（th=0.75） |
| 本文（ep20+专用头 / ep30+TTA） | **0.9686** | **89.36%** | **87.05%** | 0.5836（单次） | 0.6029（TTA, th=0.40） |

- 图级检测：本文 AUC 高 2.2 点，F1 高 1.3 点。
- 像素定位：TruFor 最优阈值下 mIoU 0.665，高于本文 TTA 的 0.603；本文的优势仍是多任务统一与鲁棒性。

## 重要说明

1. **训练数据重叠**：TruFor 官方权重训练集包含 CASIA 系列，本文也在 CASIA 上训练，双方都可能见过该验证集的同源图像；该对比不是严格的跨数据集公平比较。
2. **阈值口径**：双方的最优 mIoU 都在同一验证集上扫描得到，属于乐观估计，仅用于同口径参考。
3. 跨数据集（Columbia / Coverage）由于数据集获取受限，暂未完成。

## 复现

```bash
python -m fbfn.prep_baseline_eval --out E:/trufor_eval
# TruFor 官方 test.py（见 code/experiments/methods/TruFor/TruFor_train_test）
python -m fbfn.eval_trufor --root E:/trufor_eval --pred E:/trufor_out_full
```
