Dear Editor-in-Chief,

I am pleased to submit our manuscript entitled **"Interpretable Four-Branch Attention Fusion Network for Image Tampering Detection, Type Classification, and Localization"** for consideration in the *Journal of Visual Communication and Image Representation*.

Real-world anti-fraud investigation requires more than a real/fake score: investigators need the manipulation type and the altered region, and the detector must survive JPEG compression and resizing introduced by social-media distribution. Our work makes three contributions:

1. **A unified, interpretable architecture.** Four physically interpretable evidence branches (CNN, DINOv2 ViT, ELA/SRM noise, Lab entropy) are fused by a channel-and-spatial dual attention module and optimized jointly for authenticity, four-class tampering type, and pixel-level localization. Each branch produces an inspectable evidence map.
2. **Strong and robust results on a real-only benchmark.** On 2,866 real CASIA validation images the model reaches binary AUC 0.9686 (accuracy 89.36%, F1 87.05%), tampering-type macro-F1 0.849, and localization mIoU 0.6029. Random JPEG/resize/blur augmentation raises the JPEG q90 AUC from 0.850 to 0.888 while leaving clean metrics unchanged.
3. **Practical deployment and an honest boundary analysis.** bf16 inference takes 34 ms per image on an NVIDIA A40 (81 ms end-to-end). Under an identical protocol, the method outperforms TruFor at image-level detection (AUC 0.9686 vs. 0.9463), while TruFor remains stronger at localization (mIoU 0.665). We also report a clear cross-dataset limitation on COVERAGE (zero-shot AUC 0.56–0.58) rather than overstating generalization.

We believe the combination of multi-task interpretability, robustness to compression, and deployment efficiency fits the scope of JVCIR. The manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests. Code and model weights are publicly available at https://github.com/changest/Image-Tampering-Detection-3.0.

We would be grateful for your consideration and look forward to the reviewers' comments.

Sincerely,

Jie Lu (卢杰)
Taiyuan University of Technology, Taiyuan, Shanxi, China
Email: lujie123456bhjcyx@qq.com
