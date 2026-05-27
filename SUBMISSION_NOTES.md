# 提交材料说明

## 包含内容

- `src/pmfa/`：本文实现的攻击、PIMoG 兼容封装、迁移学习对照、`DCSS/R-DCSS` 与评测流程。
- `scripts/`：环境检查、数据下载、实验运行、R-DCSS 标定和结果可视化入口。
- `tests/`：CPU 前向、候选同步、可靠性回退和 PMFA 张量接口单元测试。
- `third_party/PIMoG/`：官方 PIMoG 源码、预训练权重及许可证。
- `outputs/full/tables/`：正式实验数值输出。
- `outputs/full/figures/`：正式实验图像与曲线。
- `outputs/full/models/`：本研究训练的轻量模型权重。
- `outputs/full/run_info.json`：实验样本量、训练开销、图像质量及运行耗时。

## 未包含内容

- `.venv/`：本地 Python 虚拟环境，按 `README.md` 重建。
- `data/`：DIV2K 与 Kodak 公开数据集，请依据 README 获取后放到指定目录。
- `outputs/full/cache/`：冻结特征缓存，可由实验脚本重新生成。
- `outputs/smoke_*`：仅用于工程调试的示例输出，不作为论文依据。

## 一键验证

```powershell
python -m venv .\.venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -X utf8 .\scripts\validate_setup.py
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
.\.venv\Scripts\python.exe -X utf8 .\scripts\build_validation_stress.py
.\.venv\Scripts\python.exe -X utf8 .\scripts\tune_dcss_gate.py
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_experiment.py --output-dir .\outputs\full
```

正式报告引用的是 `outputs/full/tables/summary_results.csv` 与
`outputs/full/run_info.json`，门控选择依据为 `outputs/full/dcss_gate.json`，
不引用 smoke 测试输出。

实验可视化使用 `scripts/make_paper_figures.py` 生成的 `paper_*.png` 图组。
生成 Word 报告的脚本不属于本次 GitHub 实验代码发布范围，未包含在仓库中。
