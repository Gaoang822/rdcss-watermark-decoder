# R-DCSS: Partial Screen-Shooting Watermark Decoding

本项目用于《高级机器学习理论》课程报告实验，研究在局部屏幕拍摄模拟条件下，
如何利用预训练深度水印主干、轻量迁移学习对照与深度置信同步搜索提高不可见水印解码鲁棒性。

正式报告题为《基于迁移学习与深度置信同步搜索的局部屏摄不可见水印鲁棒解码研究》。
核心结论是：在 50% 局部观测测试中，`DCSS` 达到 `94.49%` Bit Accuracy，
显著超过 `Decoder-Tail-FT` 的 `84.91%`；门控版本 `R-DCSS` 面向额外退化提供
保守回退，但 JPEG 仍是明确的未解决边界。

## Method

- 深度水印基线：官方 `PIMoG` ScreenShooting 预训练编码器/解码器，消息长度为 30 bit。
- 攻击：官方透视、光照、莫尔纹与高斯噪声层；另外加入局部内容保留率
  `100% / 70% / 50%`、JPEG 和模糊扰动。
- 约束：局部裁剪后直接拉伸会破坏 PIMoG 的同步。本文增加“粗定位先验不超过
  2 像素误差”的设定，并围绕该位置构造 9 个空间重定位候选。
- `PMFA`：冻结 PIMoG 特征主干，仅训练注意力融合 MLP 和 30 bit 输出头；
  训练阶段利用模拟裁剪的已知位置增加候选对齐分类辅助损失，推理阶段不读取真值位置。
  为降低失败融合的退化风险，注意力汇聚结果通过一个初始化偏向粗定位首候选的
  可学习残差门控接入解码头。
- `DCSS`：改进的主方法。对 9 个空间候选分别运行预训练深度解码器，选择其
  30 bit 输出距离二值向量最近的候选，将深度网络自身的消息置信度作为同步评价函数。
- `R-DCSS`：仅使用 DIV2K 验证样本对 `DCSS` 的最小二值误差阈值与回退分支进行
  标定；低置信时回退到粗定位候选，缓解 JPEG 等分布外失真引发的错误同步。
- 对照：无定位 `PIMoG-Original`、`Coarse-Prior`、非学习平均融合、
  `Single-View-FT`、`Decoder-Tail-FT` 以及仅作上界分析的
  `Oracle-Alignment`。
  `Decoder-Tail-FT` 为耗时更高的诊断性对照，最多训练 10 epoch。

正式结果中，原计划的 PMFA 没有超过简单适配基线，因而论文将 `DCSS/R-DCSS`
作为改进方法，将 PMFA 作为如实报告的消融对照。该定位先验是方法适用条件，
并非无条件解决盲局部屏摄。

## Directory Layout

```text
pmfa_watermark_project/
  src/pmfa/                   # model wrapper, attacks, training and evaluation
  scripts/                    # data download, validation and experiment entrypoints
  third_party/PIMoG/          # official repository and pretrained weight
  data/raw/DIV2K_valid_HR/    # 100 DIV2K validation PNG files
  data/raw/kodak/             # kodim01.png ... kodim24.png
  outputs/full/               # formal experimental outputs
```

## Data Placement

正式实验需要：

1. `DIV2K_valid_HR` 验证集的 100 张 `.png` 图片放入
   `data/raw/DIV2K_valid_HR/`。
2. Kodak 图像集的 24 张图片 `kodim01.png` 到 `kodim24.png` 放入
   `data/raw/kodak/`。

也可尝试自动下载：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\download_data.py
```

## Environment

本机已验证的 Python 环境位于 `.venv`，主要包为 CPU 版 PyTorch、torchvision、
kornia、opencv-python-headless、scikit-image、pandas 和 matplotlib。
若需重建：

```powershell
python -m venv .\.venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

官方代码使用较早的 PyTorch/Kornia/NumPy 接口。本项目在
`src/pmfa/pimog.py` 中兼容 CPU 权重加载、Kornia 变换入口和向量化莫尔纹
生成，不修改第三方源码或模型权重。

## Run

先检查权重和数据：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\validate_setup.py
```

运行正式实验：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_experiment.py --output-dir .\outputs\full
```

重新训练轻量适配器但复用相同冻结特征与攻击样本：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_experiment.py --output-dir .\outputs\full --reuse-cache
```

开发新适配器时同时复用既有消融权重：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_experiment.py --output-dir .\outputs\full --reuse-cache --reuse-baseline-models
```

生成验证集稳健性缓存并只用验证集标定 `R-DCSS` 门控：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\build_validation_stress.py
.\.venv\Scripts\python.exe -X utf8 .\scripts\tune_dcss_gate.py
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_experiment.py --output-dir .\outputs\full --reuse-cache --reuse-baseline-models
```

输出包括：

- `tables/summary_results.csv`：各攻击条件的 Bit Accuracy、BER 和消息恢复率。
- `tables/decoding_runtime.csv`：从局部观测输入到消息输出的解码耗时。
- `tables/*_history.csv`：学习模型训练曲线数据。
- `tables/dcss_gate_tuning.csv` 与 `dcss_gate.json`：验证集选择的可靠性门控参数。
- `figures/validation_accuracy_curve.png` 和 `figures/crop_robustness_curve.png`。
- `figures/example_*.png`：原图、含水印图、无定位局部输入与粗定位候选案例。
- `run_info.json`：样本数量、参数量、训练耗时、PSNR 与 SSIM。

生成实验结果可视化图：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\make_paper_figures.py
```

`make_paper_figures.py` 直接读取正式 CSV 和 `run_info.json`，导出主结果、
泛化结果、门控曲线、耗时、不可见性分布与视觉案例图。

## Reproducibility Notes

- 默认随机种子为 `2026`。
- DIV2K 前 80 张用于训练，后 20 张用于验证；Kodak 仅用于测试。
- `R-DCSS` 阈值只由 DIV2K 验证条件 `crop50 / crop50_blur / crop50_jpeg`
  确定；Kodak 测试标签不参与选择。
- 正式输出只可由 `outputs/full/` 中的结果引用；`outputs/smoke_*` 为代码冒烟
  测试，使用重复示例图，不具有论文结论效力。
- 实验仅模拟屏摄失真，未进行手机真实拍摄验证。

## Acknowledgement

基线网络与预训练模型来源于 Fang 等人的官方 `PIMoG` 仓库：
<https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw>.
第三方代码许可见 `third_party/PIMoG/LICENSE`。

## License

本仓库包含 GPL-3.0 许可下的官方 `PIMoG` 源码及预训练权重，并以其作为实验运行时依赖。
为符合其分发要求，本仓库按 GNU GPL v3 发布；许可文本见根目录 `LICENSE` 及
`third_party/PIMoG/LICENSE`。本文新增代码位于 `src/pmfa/` 与 `scripts/`。
