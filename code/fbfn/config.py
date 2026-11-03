"""FBFN 训练/评估配置。与论文表4 保持一致。"""
import os
from dataclasses import dataclass, field


# 项目根：<研究项目>/code/fbfn/config.py 上溯两级 → <研究项目>/
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))


@dataclass
class Config:
    # 数据（基于项目结构推导，搬动目录不会失效）
    casia_root: str = os.path.join(_PROJECT_ROOT, "data", "CASIA", "CASIA", "CASIA 2.0")
    gt_root: str = os.path.join(_PROJECT_ROOT, "data", "CASIA", "CASIA", "CASIA 2 Groundtruth")
    img_size: int = 384
    val_ratio: float = 0.2
    seed: int = 42

    # 训练
    batch_size: int = 4          # 实测 batch4+AMP 峰值 2.7GB，8GB 显存宽裕
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    early_stop_patience: int = 15
    num_workers: int = 2
    use_amp: bool = True

    # 模型
    pretrained: bool = True      # Swin-T / MobileNetV3 / ResNet34 用 ImageNet 初始化

    # 输出
    out_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    log_interval: int = 20       # 每 N 个 step 打印一次


DEFAULT_CONFIG = Config()
