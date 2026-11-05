# 冻结骨干的专用二分类头

## 方法

骨干（CNN + ViT + 噪声 + 颜色 + 融合）完全冻结，仅用每图 9 维特征训练 CPU 端分类器：
原二分类 logit、`final_map` 的 max / top-1% 均值 / 均值 / 标准差、4 类 type logits。
在 CASIA 训练集（11,469 张）拟合，CASIA 验证集（2,866 张）测试，验证集不参与训练。

## 结果

| 权重 | 方案 | AUC | 准确率 | 二分类 F1 |
|---|---|---|---|---|
| ep20_ema | 原二分类头 | 0.9634 | 88.17% | 85.34% |
| ep20_ema | 线性专用头 | 0.9668 | **89.46%** | **87.31%** |
| ep20_ema | MLP 专用头 | **0.9686** | 89.36% | 87.05% |
| last_ema | 原二分类头 | 0.9503 | 88.35% | **86.09%** |
| last_ema | MLP 专用头 | **0.9572** | **88.52%** | 85.99% |

## 结论

- 专用头把 ep20_ema 的 AUC 提升 **+0.0052**，F1 提升约 **+2 点**，且不增加 GPU 工作量（仅 9 维特征 + CPU 推理）。
- 对鲁棒版 last_ema，AUC 提升 **+0.0069**，准确率 +0.17 点，F1 基本持平。
- 部署选择：
  - 追求 clean 精度：`ep20_ema.pth` + `binhead.pkl`（MLP）；
  - 真实网络图：`last_ema.pth` + `binhead_last.pkl`（MLP）。
- 头部模型 41 KB，可随代码分发。

复现：

```bash
python -m fbfn.fit_binary_head --ckpt fbfn/runs_type/ep20_ema.pth --out fbfn/runs_type/binhead.pkl
python -m fbfn.eval_real --ckpt fbfn/runs_type/ep20_ema.pth --binhead fbfn/runs_type/binhead.pkl
```
