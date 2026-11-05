"""生成中文版 Word 文档（manuscript_zh.docx）。

运行：python paper/export_docx_zh.py
输出：paper/manuscript_zh.docx
"""
import os
import re
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

_PAPER = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_PAPER, "manuscript_zh.docx")
FIG_ARCH = os.path.join(_PAPER, "figures", "architecture.png")
FIG_EVIDENCE = os.path.join(_PAPER, "figures", "evidence_map.png")

ABSTRACT = ("文本到图像生成模型的快速发展，使得 AI 生成图像越来越难以与真实照片区分；"
            "在一个生成器家族上表现准确的检测器，迁移到另一个家族时性能往往急剧下降。"
            "本文提出一种四分支注意力融合网络（FBFN），通过融合四种物理可解释的证据分支——"
            "CNN 语义分支、ViT 语义分支、基于误差水平分析（ELA）与隐写分析富模型（SRM）滤波器的噪声分支、"
            "以及基于 Lab 空间局部熵的颜色分支——并经过通道-空间双重注意力融合模块与全局池化，"
            "将图像分类为真实或生成。在 Stable Diffusion v1.4 子集上训练、并在 GenImage 协议下跨八个生成器评估，"
            "FBFN 在所有 Stable Diffusion 家族生成器上达到 99% 以上的 ROC 曲线下面积（AUC），"
            "在该家族上与一个更强的近期黑盒检测器相当，同时保留了像素级可解释证据。"
            "消融实验表明，噪声分支在训练分布内近乎冗余，但具体负责跨生成器泛化，"
            "移除后在远距离生成器上 AUC 下降 5–7 个点。多生成器联合训练进一步弥合了大部分跨家族差距。"
            "我们讨论了该方法仍然薄弱之处——尤其是 GAN 生成图像——并将其可解释性定位为区别于更强位平面检测器的主要优势。")

KEYWORDS = "AI 生成图像检测；伪造检测；可解释深度学习；跨生成器泛化；注意力融合。"

BODY = [
    ("h1", "1. 引言"),
    ("p", "大规模文本到图像模型现在可以低成本合成逼真图像，其输出越来越多地被用于虚假信息、身份欺诈和伪造证据。因此，检测这类图像已成为一个紧迫的实际问题，并催生了大量文献。核心难点不在于识别训练中见过的生成器生成的图像——这在很大程度上已经解决——而在于识别训练中未见过的新生成器生成的图像。在一个扩散模型上训练的检测器，面对来自不同家族的图像时往往失效，因为区分生成图像与真实图像的低级伪影在不同架构之间差异很大。"),
    ("p", "先前工作从两个主要方向攻克这一泛化难题。一条路线使用卷积或视觉 Transformer 骨干网络的通用语义特征 [1, 2]，泛化能力中等，但仍受限于训练分布的纹理统计。另一条路线利用手工设计或学习到的低级噪声线索——误差水平分析（ELA）[3]、隐写分析富模型（SRM）滤波器 [4]，以及最近的图像最低有效位平面 [5]——其假设是生成模型会在残差信号中留下特征性签名。这条路线的方法已被证明是迄今最强的跨生成器检测器，但它们通常是黑盒，只输出一个标量分数，不提供可供人类或下游系统检查的任何证据。"),
    ("p", "本文贡献了一种四分支架构，融合这两类证据，同时保持每个分支物理可解释。该网络我们称之为四分支注意力融合网络（FBFN），由 (i) CNN 分支和 (ii) ViT 分支捕获语义外观，(iii) 噪声分支捕获 ELA 和 SRM 残差伪影，(iv) 颜色分支捕获颜色空间和局部熵统计。四张像素级证据图通过通道-空间双重注意力融合模块组合，并池化为图像级分数。我们在 GenImage 协议下评估 FBFN——在 Stable Diffusion v1.4 上训练、跨八个生成器测试——并将其与三个代表性基线和一个新的更强检测器对比。我们进一步通过在训练分布内和分布外进行的留一法消融，剖析每个分支的贡献。"),
    ("p", "我们的结果说明了三点。第一，FBFN 在所有 Stable Diffusion 家族生成器上达到 99% 以上的 AUC，与我们测试的最强黑盒检测器相当，而通用 RGB 基线的退化要大得多。第二，噪声分支——尽管在训练分布内近乎冗余——是负责大部分跨生成器鲁棒性的组件；移除它在最远距离生成器上使 AUC 下降 5–7 个点。第三，多生成器联合训练为扩散模型弥合了大部分剩余跨家族差距，但对 GAN 图像无帮助，该方法在此与若干基线一样接近随机。我们讨论了这些发现的含义，以及使 FBFN 区别于更强但不透明检测器的可解释性优势。"),
    ("h1", "2. 相关工作"),
    ("h2", "语义特征检测器。"),
    ("p", "最早也最简单的方法，把生成图像检测当作使用标准图像骨干的二分类问题。GenImage 基准 [1] 建立了在单个生成器上训练的 ResNet-50 基线，并表明准确率跨生成器急剧下降。CNNDetection [2] 在训练时对 ResNet-50 加入模糊和 JPEG 损坏以诱导鲁棒性，但其零样本跨生成器准确率仍然有限。这些方法易于训练，但继承了骨干网络所学纹理统计的分布敏感性。"),
    ("h2", "低级伪影检测器。"),
    ("p", "一条并行路线针对生成管线留下的残差信号。误差水平分析 [3]（可视化 JPEG 压缩不一致）和隐写分析富模型高通滤波器 [4]（最初为隐写术设计）都暴露了真实与生成图像之间不同的像素级证据。该路线最强的近期方法 LOTA [5] 提取图像的最低有效位平面，按梯度幅值选择图像块，并用 ResNet-50 分类；其报告的跨生成器准确率显著高于语义骨干。这些基于伪影的检测器泛化良好，但通常只产生一个标量决策，可解释性有限。"),
    ("h2", "零样本与基础模型检测器。"),
    ("p", "第三条路线利用 CLIP [6] 等大型预训练视觉-语言模型的特征，假设生成图像落在自然图像流形之外。代表性方法 UnivFD [7] 仅在冻结的 CLIP 特征上训练一个线性头。这类检测器对某些生成器家族迁移良好，但对其他家族不敏感，反映了基础模型特征是为语义而非取证线索优化的这一事实。"),
    ("p", "FBFN 将前两条路线——语义证据与低级证据——结合在单一可解释架构中，与标量黑盒伪影检测器和通用骨干都不同：每个分支都产生一张可供人类检查的证据图。"),
    ("h1", "3. 方法"),
    ("h2", "3.1 任务定义"),
    ("p", "我们将问题视为二分类图像分类。给定输入 RGB 图像 x，网络输出一个标量 logit，经过 sigmoid 后估计 x 是 AI 生成的概率。模型在真实与生成图像对上以二值交叉熵监督训练。"),
    ("h2", "3.2 概述"),
    ("p", "FBFN 由四个分支组成，每个分支产生一张空间尺寸相同的像素级证据图，随后是双重注意力融合模块与全局平均池化。四个分支被选择来覆盖互补且物理可解释的证据来源："),
    ("bullet", "**CNN 分支**——以 ResNet-34 [8] 为编码器的编码器-解码器网络（UNet++ [11]），捕获多尺度语义外观。"),
    ("bullet", "**ViT 分支**——Swin Transformer [9]（Tiny）搭配 SegFormer [10] MLP 解码器，捕获长程与全局语义结构。"),
    ("bullet", "**噪声分支**——误差水平分析（JPEG 质量 90）与三个 5×5 零和 SRM 高通滤波器拼接，捕获压缩与残差噪声伪影。"),
    ("bullet", "**颜色分支**——Lab 颜色空间特征加滑窗局部熵，后接浅层 CNN，捕获颜色统计异常。"),
    ("figure1", None),
    ("h2", "3.3 证据分支"),
    ("p", "每个分支将输入映射为单通道证据图。CNN 与 ViT 分支直接作用于 RGB 输入，产出语义证据。噪声分支接收 ELA 与 SRM 滤波响应作为辅助输入，并将其聚合为噪声证据。颜色分支接收熵图作为辅助输入，聚合颜色空间与熵证据。使用显式、命名的证据流——而非单一融合表示——正是使模型决策可检查之处：每张证据图都可被可视化，以显示网络依赖*何处*以及*何种*伪影。"),
    ("figure2", None),
    ("h2", "3.4 双重注意力融合"),
    ("p", "四张证据图拼接后通过一个融合模块，该模块包含两个协同的注意力机制。通道注意力路径采用全局平均池化、两层 MLP 和 softmax，对四种证据类型的重要性重新加权。空间注意力路径采用一个卷积后接 sigmoid，对每个空间位置的重要性重新加权。两个注意力输出合并后卷积为单张融合证据图。全局平均池化随后将融合图压缩为图像级 logit，另有一个注意力池化 logit 在训练时作为辅助信号。"),
    ("h2", "3.5 训练"),
    ("p", "模型以组合二值交叉熵损失训练："),
    ("eq", "L = BCE(final) + 0.5·BCE(att) + 0.5·Σ BCE(branch)"),
    ("p", "其中分支项对保留的分支求和。我们使用 AdamW（学习率 1e-4，权重衰减 1e-4）、余弦学习率调度、混合精度与水平翻转增强训练。输入图像缩放到 256×256。完整模型含 54.68M 可训练参数。"),
    ("h1", "4. 实验"),
    ("h2", "4.1 设置"),
    ("p", "**数据集与协议。** 我们遵循 GenImage 协议：模型仅在 Stable Diffusion v1.4 子集上训练，并在八个生成器（Stable Diffusion v1.4 与 v1.5、Wukong、ADM、GLIDE、BigGAN、Midjourney 与 VQDM）的验证集上评估。图像取自 TIGAS 数据集 [12]（GenImage [1] 的 Hugging Face 镜像），该数据集将 Stable Diffusion v1.5 子集拆成两部分；因此表格将 Stable Diffusion v1.5 报告为两行。训练集由 Stable Diffusion v1.4 的 6,000 张图像（3,000 真实、3,000 生成）组成；同分布验证集含 1,000 张图像；八个跨生成器验证子集各含 400–1,200 张图像（BigGAN 最少，为 200 真实与 200 生成）。"),
    ("p", "**指标。** 我们以 ROC 曲线下面积（AUC）为主指标，因为它与阈值无关，因而对生成器子集间不同的类别平衡不太敏感。准确率与平均精度（AP）均已计算，但为简洁起见不在表格中列出。"),
    ("p", "**基线。** 我们与在相同 Stable Diffusion v1.4 数据上训练的 ResNet-50、CNNDetection（零样本）和 UnivFD（零样本，CLIP 特征）对比。所有方法都在相同的测试图像上评估。"),
    ("h2", "4.2 同分布与跨生成器结果"),
    ("p", "FBFN 在同分布 Stable Diffusion v1.4 验证集上达到 99.93% 的 AUC。表 1 报告了所有方法的跨生成器 AUC。"),
    ("table_caption", "表 1. 跨生成器 AUC（%），Stable Diffusion v1.4 训练。"),
    ("table1", None),
    ("p", "由此可得出三点观察。第一，FBFN 在 Stable Diffusion 家族和 Wukong 上近乎无损泛化（全部高于 99% AUC），在该家族上大幅超越两个零样本基线，并超过 ResNet-50 基线。第二，在更远的扩散生成器（ADM、GLIDE、Midjourney、VQDM）上，FBFN 退化到 66–77%，反映了语义分支未完全吸收的分布偏移。第三，在 BigGAN——一个 GAN 而非扩散模型——上，FBFN 接近随机（54.23%），与其噪声和颜色证据针对扩散专属伪影的假设一致。值得注意的是，基线呈现出镜像模式：UnivFD 和 CNNDetection 在 BigGAN 上（99.00% 与 88.11%）远强于在 Stable Diffusion 家族上，证实这些方法捕获的是与 FBFN 不同种类的证据。"),
    ("h2", "4.3 联合训练"),
    ("p", "为检验跨家族差距能否通过数据而非架构来缩小，我们在 Stable Diffusion v1.4、v1.5、VQDM 和 Midjourney 图像的组合池（共 9,000 张）上训练 FBFN，将 ADM、GLIDE 和 BigGAN 留作留出生成器。联合训练显著提升了留出扩散生成器——ADM 从 66.04% 升至 83.87%，GLIDE 从 68.42% 升至 94.48%——但没有提升 BigGAN（54.23% → 48.31%），后者仍停留在随机水平。这证实扩散模型的跨家族差距主要是数据覆盖差距，而 BigGAN 的失败反映了伪影类型上的更根本差异。"),
    ("h2", "4.4 消融研究"),
    ("p", "我们进行了留一法消融，训练四个变体，各移除一个分支，同时保持相同协议（15 epoch，Stable Diffusion v1.4）。由于两个语义分支变体在远距离生成器上对训练种子敏感，我们各用不同随机初始化重跑一次；表 2 报告第二次运行，方差在下方量化。"),
    ("table_caption", "表 2. 消融 AUC（%）。同分布（SD v1.4）与跨生成器。"),
    ("table2", None),
    ("p", "在训练分布内，所有变体都接近上限（AUC ≥ 99.4%），因此无法从这些饱和分数中识别出某个占主导的分支。跨生成器结果则隔离出唯一稳健的贡献者。移除噪声分支使 AUC 在三个与训练分布距离最远的生成器上下降——ADM（−6.44）、BigGAN（−7.04）和 VQDM（−5.19）——而 Stable Diffusion 家族几乎不变。这表明 ELA/SRM 噪声证据充当了一个可跨生成器家族迁移的扩散伪影检测器，而非冗余的分布内特征。"),
    ("p", "其余分支没有显示出可靠的跨生成器贡献。CNN 与颜色分支在 ADM、GLIDE 和 VQDM 上仅改变几个点，且方向不一致（颜色分支在 ADM 上移除后甚至提升）。语义分支在远距离生成器上还表现出很大的运行间方差：两次运行中，w/o-CNN 的 GLIDE AUC 从 39.27% 变为 77.93%，w/o-ViT 的 GLIDE AUC 从 92.91% 变为 57.05%，摆幅约 35 个点且方向相反。因此，我们不将跨生成器贡献归因于 CNN 或 ViT 分支，并将其精确的跨生成器值视为不可靠。相比之下，噪声分支的退化在 ADM、BigGAN 和 VQDM 上符号与幅度都稳定，这正是使其成为跨生成器鲁棒性的可解释来源的原因。"),
    ("h2", "4.5 计算效率"),
    ("p", "系统一/系统二的划分也体现为计算开销上的非对称分布。表 3 在 256×256 输入下统计各分支参数量；表 4 报告在 NVIDIA RTX 5060 Laptop 上测得的前向延迟（50 次均值，GPU 预热后）。测量时前向仅传入预计算的 ELA 与熵图，以隔离网络本身的开销——ELA 编解码与滑窗熵属数据预处理，不计入网络前向。"),
    ("table_caption", "表 3. 各分支参数量（256×256）。"),
    ("table3", None),
    ("p", "系统一分支合计 0.45M 参数（0.8%），系统二分支合计 54.23M（99.2%）；总 FLOPs 为 33.27 GFLOPs。"),
    ("table_caption", "表 4. 单图前向延迟（RTX 5060 Laptop，ms）。"),
    ("table4", None),
    ("p", "系统一分支仅占总参数不到 1%，单图延迟约 3 ms，而系统二分支承担了几乎全部计算开销，这直接印证了双过程类比中的成本假设：系统一证据几乎无成本，系统二证据昂贵。这一非对称性也自然引出一个级联推理方案——系统一先对全部图像快速筛选，仅当系统一无法给出高置信判断时才升级到全模型，期望平均延迟为 T1 + ρ·(T − T1)，其中 T1 = 3 ms 为系统一延迟，T = 30 ms 为全模型延迟，ρ 为升级比例。表 5 在同分布验证集上报告了该级联的权衡。"),
    ("table_caption", "表 5. 同分布（SD v1.4）级联 AUC 与平均延迟权衡。"),
    ("table5", None),
    ("p", "结果表明，系统一单独（噪声+颜色证据）在同分布上仅达 91.22% AUC，且在跨生成器上普遍更弱（各生成器 36–69% AUC）。级联要恢复到约 99% AUC 需升级约 63% 的样本，平均延迟约 20 ms——加速有限，且以明显精度损失为代价。这给出了一个与 4.4 节消融一致但更直接的结论：系统一证据虽然便宜，却不足以独立支撑高精度筛选；其价值在于与系统二证据融合以提供跨生成器鲁棒性，而非作为独立的快速通道。因此，除非训练一个更强的系统一头（例如为其补充轻量语义编码），否则级联在 FBFN 上的收益有限，我们将其留作未来工作。"),
    ("h2", "4.6 扩展：篡改类型分类与像素级定位（v3.0）"),
    ("p", "为适配反诈取证场景，我们在同一融合框架上扩展出多任务版本。(i) 将 ViT 分支替换为 DINOv2 ViT-L/14，输入 448×448（满足 14 的整除约束）；由于 1024 通道在高分辨率下直接插值会触发显存溢出，先以 1×1 卷积降至 128 通道再双线性上采样。(ii) 在四分支证据图之后接入共享的 128 维卷积分支，并设两个头：4 类篡改类型头与专用二分类头（真/假）。(iii) 训练时联合优化像素级分割（BCE+Dice）、图级真伪与类型分类。"),
    ("p", "**数据。** 训练集融合 CASIA v1/v2（真实、拼接、复制移动）、TIGAS 生成图与合成的复制移动/拼接样本，共 25,469 张（真 9,633、拼接 5,813、复制移动 7,023、AI 生成 3,000；其中 19,439 张带像素掩码）。为检验真实场景表现，评测仅使用纯真实 CASIA 验证集（2,866 张：真 1,658、拼接 466、复制移动 742），不混入 TIGAS 或合成数据。"),
    ("p", "**训练。** AdamW（学习率 1e-4）、余弦退火、EMA（衰减 0.999）、混合精度与 2×A40 DDP。总损失为 L = 1.0·L_seg + 0.5·L_bin + 0.5·L_type，其中 L_seg = BCE + Dice，L_type 为按类别频率加权的交叉熵。表 6 报告各检查点在真实 CASIA 验证集上的结果。"),
    ("table_caption", "表 6. 真实 CASIA 验证集结果。"),
    ("table6", None),
    ("p", "联合训练使类型 macro-F1 从 0.718 提升至 0.849，分割 mIoU 从 0.332 提升至 0.584，且分类与定位相互促进。与 256×256 输入的小骨干版本（Swin-T/ResNet-34：类型 F1 0.818、mIoU 0.490、AUC 0.977）相比，大骨干在类型分类与定位上均有提升，但单一真伪 AUC 略低（0.963 对 0.977），说明其收益主要来自细粒度类型判别与像素级定位，而非简单的图级二分类。继续训练至 40 轮后分类指标反而下降（AUC 0.9634→0.9473），呈现对混合训练分布的过拟合。"),
    ("p", "**阈值与温度校准。** 默认 0.5 判定阈值下，模型偏向保守。在验证集上搜索阈值后，ep30 检查点在阈值 0.18 时准确率由 88.14% 提升至 89.95%、二分类 F1 由 85.20% 提升至 88.47%；温度缩放（T≈1.88）将期望校准误差（ECE）由 0.081 降至 0.059。表 7 汇总三个代表性检查点。"),
    ("table_caption", "表 7. 二分类阈值与温度校准（真实 CASIA）。"),
    ("table7", None),
    ("p", "我们进一步测试了 ep20(EMA) 与 ep30 的 logits 融合：融合仅使 AUC 提高约 0.001、类型 F1 提高约 0.3 个点，而推理成本翻倍，因此最终部署不采用集成。综合权衡下，二分类与分割采用 ep30（阈值 0.18、温度 1.88），类型分类采用 ep20(EMA)。"),
    ("p", "**局限。** 复制移动类召回约 74.8%，常被误判为真实图像；拼接类表现最好（召回约 86%）。最优阈值是在 CASIA 上拟合的，部署到其他数据域时需要重新校准；且验证集仅覆盖 CASIA，跨数据集泛化仍有待评估。"),
    ("h2", "4.7 部署与推理加速"),
    ("p", "为验证工程可用性，我们在两张显卡上测试混合精度推理：服务器端 NVIDIA A40 与笔记本端 RTX 5060 Laptop。表 8 报告单图前向延迟（batch=1，预热后 30 次均值，输入 448×448）。"),
    ("table_caption", "表 8. 单图推理延迟（ms，模型前向，不含 ELA/熵预处理）。"),
    ("table8", None),
    ("p", "A40 上端到端单图耗时（含 ELA/熵计算与热图生成）由 fp32 的 128 ms 降至 bf16 的 81 ms；bf16 batch=8 的单图摊销延迟为 21.7 ms，约为 fp32 单图延迟的 3.6 倍加速。"),
    ("p", "精度方面，表 9 在完整真实 CASIA 验证集（2,866 张）上比较三种精度。bf16/fp16 与 fp32 的 AUC、类型准确率、macro-F1 与分割 mIoU 差异均不超过 0.0005，可视为无损。"),
    ("table_caption", "表 9. 混合精度推理的精度对比（last_ema，真实 CASIA）。"),
    ("table9", None),
    ("p", "基于该结果，我们将 bf16 设为默认推理精度，并把检测服务以局域网 HTTP 形式部署在 A40 服务器上：浏览器上传图片后返回真伪概率、类型分布与热图，单次请求的模型与预处理耗时约 81 ms。`torch.compile` 因自定义分支算子不受支持而无法使用；若需进一步压缩延迟，可尝试 ONNX Runtime 或 TensorRT 的算子融合与 INT8 量化，这属于后续工作。"),
    ("h1", "5. 讨论"),
    ("p", "结果对 FBFN 的优势来源与边界给出了一致的解释。在 Stable Diffusion 家族内及其间，FBFN 达到了我们所测试最强检测器的准确率，同时提供了标量检测器所缺乏的可解释证据结构。消融将噪声分支隔离为跨生成器鲁棒性的来源，联合训练则表明剩余的扩散家族差距是数据覆盖限制而非架构限制。"),
    ("p", "我们明确与 LOTA [5] 对比——这是一个近期的位平面检测器（ICCV 2025），在跨生成器 AUC 上强于 FBFN：LOTA 在 ADM 上达到 99.96%、GLIDE 上 99.99%、VQDM 上 100.00%，而 FBFN 仅 66–77%。我们不主张在这一轴上超越 LOTA。区别在于可解释性：LOTA 与其他伪影检测器一样，只输出标量分数，而 FBFN 的每个分支都产生一张像素级证据图，可检查是何种伪影、在哪个区域驱动了决策。对于决策必须可审计的应用——内容审核、取证分诊——即使原始跨生成器准确率较低，这种可审计性也是一项具体优势。"),
    ("p", "有两点局限值得强调。第一，该方法对 GAN 生成图像较弱，在 BigGAN 上退化到随机，且联合训练无法恢复；噪声与颜色证据似乎专属于扩散伪影。第二，我们的评估使用了 GenImage 的下采样子集（256×256，6,000 张训练图像），这可能低估了完整基准的挑战，并限制了与在完整数据集上报告的数字的直接比较。尽管如此，本文所有比较都在相同数据上进行，因此相对发现不受影响。"),
    ("h1", "6. 结论"),
    ("p", "我们提出了 FBFN，一个用于 AI 生成图像检测的四分支注意力融合网络，其分支对应四种物理可解释的证据流。在 GenImage 协议下，FBFN 在 Stable Diffusion 家族上近乎无损泛化，且其噪声分支被消融证明具体负责跨生成器鲁棒性。联合训练弥合了大部分扩散家族差距，而 GAN 图像仍是一个未解决的失败模式。v3.0 版本进一步将框架扩展至多任务篡改分析：在纯真实 CASIA 验证集上，篡改类型 macro-F1 达 0.849、像素级定位 mIoU 达 0.584，为反诈取证同时提供类型判别与区域证据。主要贡献不是跨生成器准确率的新最优——存在更强的黑盒检测器——而是一个在 Stable Diffusion 家族上保持竞争力、同时保留可解释证据的检测器。这种可解释性旨在支持标量分数不足时的可审计检测。"),
    ("h1", "致谢"),
    ("p", "本研究受 [基金来源与项目编号——待作者填写] 资助。"),
    ("h1", "利益冲突"),
    ("p", "作者声明不存在利益冲突。"),
    ("h1", "数据可用性声明"),
    ("p", "本研究使用的 GenImage 基准可通过 TIGAS 镜像公开获取：https://huggingface.co/datasets/H1merka/TIGAS_dataset 。v3.0 训练与评估代码开源于 https://github.com/changest/Image-Tampering-Detection-3.0 ；模型权重可向通讯作者合理索取。"),
    ("h1", "参考文献"),
]

TABLE1_HEADER = ["生成器", "FBFN (sd14)", "FBFN (联合)", "ResNet-50", "CNNDetection", "UnivFD"]
TABLE1_ROWS = [
    ["SD v1.4", "99.93", "—", "97.70", "56.96", "73.54"],
    ["SD v1.5 (a)", "99.74", "—", "98.28", "60.00", "72.59"],
    ["SD v1.5 (b)", "99.77", "—", "97.24", "59.89", "74.61"],
    ["Wukong", "99.43", "—", "96.55", "53.24", "80.55"],
    ["ADM", "66.04", "83.87", "37.34", "69.00", "94.30"],
    ["GLIDE", "68.42", "94.48", "52.86", "70.65", "86.39"],
    ["BigGAN", "54.23", "48.31", "36.62", "88.11", "99.00"],
    ["Midjourney", "76.03", "—", "77.62", "51.21", "51.35"],
    ["VQDM", "77.40", "—", "40.27", "72.13", "96.88"],
]

TABLE2_HEADER = ["配置", "SD v1.4", "ADM", "GLIDE", "BigGAN", "VQDM"]
TABLE2_ROWS = [
    ["Full", "99.93", "66.04", "68.42", "54.23", "77.40"],
    ["w/o CNN", "99.98", "79.44", "77.93", "48.95", "75.72"],
    ["w/o ViT", "99.40", "57.67", "57.05", "52.71", "70.48"],
    ["w/o Noise", "99.96", "59.60", "68.13", "47.19", "72.21"],
    ["w/o Color", "99.85", "74.81", "65.21", "49.76", "76.20"],
]

TABLE3_HEADER = ["分支", "参数量 (M)", "占比"]
TABLE3_ROWS = [
    ["噪声（系统一）", "0.04", "0.07%"],
    ["颜色（系统一）", "0.41", "0.75%"],
    ["CNN（系统二）", "26.08", "47.7%"],
    ["ViT（系统二）", "28.15", "51.5%"],
    ["总计", "54.68", "100%"],
]

TABLE4_HEADER = ["配置", "延迟 (ms)"]
TABLE4_ROWS = [
    ["全模型（四分支+融合）", "30"],
    ["仅系统一（噪声+颜色）", "3"],
    ["仅系统二（CNN+ViT）", "27"],
]

TABLE5_HEADER = ["升级比例 ρ", "级联 AUC (%)", "平均延迟 (ms)"]
TABLE5_ROWS = [
    ["0.00（仅系统一）", "91.22", "3"],
    ["0.38", "97.56", "13"],
    ["0.63", "99.06", "20"],
    ["0.85", "99.66", "26"],
    ["1.00（全模型）", "99.93", "30"],
]

TABLE6_HEADER = ["检查点", "真伪 AUC", "类型 macro-F1", "分割 mIoU"]
TABLE6_ROWS = [
    ["ep5", "0.9230", "0.7179", "0.3323"],
    ["ep10", "0.9555", "0.8019", "0.4830"],
    ["ep14 (EMA)", "0.9672", "0.8251", "—"],
    ["ep20 (EMA)", "0.9634", "0.8492", "0.5614"],
    ["ep30", "0.9587", "0.8394", "0.5836"],
    ["ep40 (EMA)", "0.9473", "0.8412", "0.5807"],
]

TABLE7_HEADER = ["检查点", "AUC", "原始准确率 / F1", "最优阈值", "校准后准确率 / F1", "T", "ECE(前→后)"]
TABLE7_ROWS = [
    ["ep20 (EMA)", "0.9634", "88.17% / 85.34%", "0.08", "89.64% / 88.27%", "1.60", "0.066→0.047"],
    ["ep30", "0.9587", "88.14% / 85.20%", "0.18", "89.95% / 88.47%", "1.88", "0.081→0.059"],
    ["ep40 (EMA)", "0.9473", "87.89% / 84.97%", "0.10", "89.64% / 88.37%", "2.01", "0.092→0.055"],
]

TABLE8_HEADER = ["精度", "A40 (ms)", "RTX 5060 (ms)", "加速比（A40 / 5060）"]
TABLE8_ROWS = [
    ["fp32", "78.8", "191.0", "1.00× / 1.00×"],
    ["bf16", "34.3", "80.9", "2.30× / 2.36×"],
    ["fp16", "31.9", "74.6", "2.47× / 2.56×"],
    ["bf16 + channels_last", "31.1", "78.6", "2.53× / 2.43×"],
    ["bf16, batch=8", "21.7", "—", "3.63× / —"],
]

TABLE9_HEADER = ["精度", "AUC", "类型准确率", "类型 macro-F1", "分割 mIoU"]
TABLE9_ROWS = [
    ["fp32", "0.9502", "0.8552", "0.8410", "0.5844"],
    ["bf16", "0.9507", "0.8555", "0.8414", "0.5840"],
    ["fp16", "0.9503", "0.8555", "0.8413", "0.5845"],
]

EXTRA_TABLES = {
    "table3": (TABLE3_HEADER, TABLE3_ROWS),
    "table4": (TABLE4_HEADER, TABLE4_ROWS),
    "table5": (TABLE5_HEADER, TABLE5_ROWS),
    "table6": (TABLE6_HEADER, TABLE6_ROWS),
    "table7": (TABLE7_HEADER, TABLE7_ROWS),
    "table8": (TABLE8_HEADER, TABLE8_ROWS),
    "table9": (TABLE9_HEADER, TABLE9_ROWS),
}

REFERENCES = [
    "Zhu, M., Chen, H., Yan, Q., et al.: “GenImage: A million-scale benchmark for detecting AI-generated image” Adv. Neural Inf. Process. Syst., 2023, 36.",
    "Wang, S.-Y., Wang, O., Zhang, R., Owens, A., Efros, A.A.: “CNN-generated images are surprisingly easy to spot... for now” Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 8695–8704.",
    "Krawetz, N.: “A picture’s worth: Digital image analysis and forensics” Hacker Factor Blog, 2007, http://www.hackerfactor.com/blog/, accessed August 2026.",
    "Fridrich, J., Kodovský, J.: “Rich models for steganalysis of digital images” IEEE Trans. Inf. Forensics Secur., 2012, 7, (3), pp. 868–882.",
    "Wang, H., Cheng, R., Zhang, Y., Han, C., Gui, J.: “LOTA: Bit-planes guided AI-generated image detection” Proc. IEEE/CVF Int. Conf. Comput. Vis., 2025.",
    "Radford, A., Kim, J.W., Hallacy, C., et al.: “Learning transferable visual models from natural language supervision” Proc. Int. Conf. Mach. Learn., 2021, 139, pp. 8748–8763.",
    "Ojha, U., Li, Y., Lee, Y.J.: “Towards universal fake image detectors that generalize across generative models” Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 24480–24489.",
    "He, K., Zhang, X., Ren, S., Sun, J.: “Deep residual learning for image recognition” Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2016, pp. 770–778.",
    "Liu, Z., Lin, Y., Cao, Y., et al.: “Swin transformer: Hierarchical vision transformer using shifted windows” Proc. IEEE/CVF Int. Conf. Comput. Vis., 2021, pp. 10012–10022.",
    "Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P.: “SegFormer: Simple and efficient design for semantic segmentation with transformers” Adv. Neural Inf. Process. Syst., 2021, 34, pp. 12077–12090.",
    "Zhou, Z., Rahman Siddiquee, M.M., Tajbakhsh, N., Liang, J.: “UNet++: A nested U-Net architecture for medical image segmentation” Deep Learn. Med. Image Anal. Multimodal Learn. Clin. Decis. Support, 2018, pp. 3–11.",
    "H1merka: “TIGAS dataset” Hugging Face, https://huggingface.co/datasets/H1merka/TIGAS_dataset, accessed August 2026.",
]


def add_md_runs(paragraph, text):
    token_re = re.compile(r"(\*\*.+?\*\*|\*.+?\*)")
    for part in token_re.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = paragraph.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = paragraph.add_run(part[1:-1])
            r.italic = True
        else:
            paragraph.add_run(part)


def add_table(doc, header, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htext in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.text = htext
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            table.rows[i + 1].cells[j].text = val
    return table


def _add_figure(doc, path, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(6.2))
    cap = doc.add_paragraph()
    r = cap.add_run(caption)
    r.font.size = Pt(9)


def main():
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = t.add_run("用于 AI 生成图像检测的可解释四分支注意力融合网络")
    tr.bold = True
    tr.font.size = Pt(14)

    doc.add_heading("摘要", level=1)
    doc.add_paragraph(ABSTRACT)

    kw = doc.add_paragraph()
    kw.add_run("关键词：").bold = True
    kw.add_run(KEYWORDS)

    for kind, text in BODY:
        if kind == "h1":
            doc.add_heading(text, level=1)
        elif kind == "h2":
            doc.add_heading(text, level=2)
        elif kind == "p":
            add_md_runs(doc.add_paragraph(), text)
        elif kind == "bullet":
            add_md_runs(doc.add_paragraph(style="List Bullet"), text)
        elif kind == "eq":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run(text).italic = True
        elif kind == "table_caption":
            p = doc.add_paragraph()
            p.add_run(text).bold = True
        elif kind == "table1":
            add_table(doc, TABLE1_HEADER, TABLE1_ROWS)
        elif kind == "table2":
            add_table(doc, TABLE2_HEADER, TABLE2_ROWS)
        elif kind in EXTRA_TABLES:
            header, rows = EXTRA_TABLES[kind]
            add_table(doc, header, rows)
        elif kind == "figure1":
            _add_figure(doc, FIG_ARCH,
                "图 1. 所提出的四分支注意力融合网络（FBFN）总体结构。CNN 与 ViT 分支捕获语义外观，噪声（ELA + SRM）与颜色（Lab + 熵）分支捕获物理可解释的低级伪影；每个分支产生一张像素级证据图。四张图由通道-空间双重注意力模块融合，并全局池化为图像级真/假分数。")
        elif kind == "figure2":
            _add_figure(doc, FIG_EVIDENCE,
                "图 2. 一张真实图像（上）和一张生成图像（下）的分支证据图。每个分支输出单通道伪造概率图（暖色 = 伪造概率更高）；融合图综合四者。噪声与颜色分支突出了 RGB 输入中不可见的残差与颜色统计伪影，使决策可检查。")

    for i, ref in enumerate(REFERENCES, 1):
        p = doc.add_paragraph()
        p.add_run(f"[{i}] ").bold = True
        p.add_run(ref)

    doc.save(OUT)
    print(f"已生成 {OUT}")


if __name__ == "__main__":
    main()
