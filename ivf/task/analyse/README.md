# 模型评估、SHAP 与特征消融

除特别说明外，命令均从**仓库根目录**执行，使用已安装项目依赖的 Python 环境。GPU 编号仅为示例，执行前确认设备空闲；模型 checkpoint 只加载可信的本地文件。

## 入口与输出

| 功能 | 入口 | 正式输出 |
|---|---|---|
| Benchmark 预测生成 | `benchmark_run.py` | `benchmark/` |
| Benchmark 表格与箱线图 | `benchmark_summary.py` | `benchmark_summary/` |
| SHAP、固定模型遮蔽与绘图 | `ablation.py` | `ablation_outputs/` |
| 单模型预测与解释 | `run.py` | 指定的 CSV / 本目录已有结果 |
| 交互分析 | `analyse.ipynb`、`distribution_analyse.ipynb` | notebook 中指定的图表 |

`ablation_manifest.json` 统一记录 seed 0–14 对应的 AnnData 与 Effristor 模型路径；移动数据或权重后需更新它。不要将原始数据、模型或 `benchmark/*.combined.csv` 当作临时文件删除：combined 预测仍用于提取训练集指标。

## 1. Benchmark

### 1.1 从已有预测生成表格与箱线图

```bash
python -m ivf.task.analyse.benchmark_summary
# 已有结果默认拒绝覆盖；确认后更新：
python -m ivf.task.analyse.benchmark_summary --overwrite
```

不重新拟合或推理。默认汇总 15 个划分、5 个模型：LR、HGB、RF、TabPFN、Effristor。

- LR 是 `LinearRegression(fit_intercept=False, positive=True)`，不是 Logistic Regression；HGB、RF、TabPFN 也是回归器，预测分数不一定是校准概率，不截断到 `[0, 1]`。
- 仅计算 ROC-AUC、Average Precision（AP，`sklearn.metrics.average_precision_score`），不是梯形 PR-AUC。正类 1 为异常受精；不计算 Accuracy/F1，不需要分类阈值。
- 表格分组：Training set、Validation set、2025 test set、Validation set − 2025 test set。差值在同一模型同一划分内计算后再汇总，报告 mean ± 样本 SD（ddof=1），不是两组标准差相减。
- 默认按 manifest 中原始 train 索引与样本 ID，从 `.combined.csv` 提取训练样本，并核对标签。Training set 是训练内拟合表现，不能作为独立泛化性能。

输出位于 `ivf/task/analyse/benchmark_summary/`：

| 文件 | 内容 |
|---|---|
| `metrics.csv` | 各划分各模型的指标及配对差值 |
| `summary.csv` | mean/std/count 数值汇总 |
| `performance_table.csv`、`performance_table.pdf` | 均值 ± SD 性能表 |
| `performance_boxplots.pdf` | 模型为横轴，上排 Validation/2025 test 并排着色，下排 Validation − 2025 test 配对差值；左 ROC-AUC、右 AP，上排 ROC-AUC 为 0.65–1，AP 按数据范围自适应；顶部统一图例，无大标题与底部说明 |

箱线图箱体为中位数与 IQR，须为 1.5 IQR，散点为各划分，白色菱形为均值；不展示 Training set，差值面板保留零参考线并按数据设置对称轴范围；表格及 CSV 仍保留训练集指标。可指定 `--source`、`--output`、`--seeds`（至少两个不同 seed）、`--training-manifest`。仅需合并集展示时使用 `--include-combined`，不与显式 `--training-manifest` 同用。

### 1.2 需要重新生成预测时

```bash
# 仅在确实需要重新拟合基线模型时执行；输出必须是不存在的新目录。
SCIPY_ARRAY_API=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m ivf.task.analyse.benchmark_run \
  --device cuda:3 --output /path/to/new_predictions

# 使用这次运行的全部预测单独汇总，不与历史拟合结果混用。
python -m ivf.task.analyse.benchmark_summary \
  --source /path/to/new_predictions --output /path/to/new_summary
```

每个划分只用训练集拟合 LR/HGB/RF/TabPFN，同一模型生成 validation、combined（train + validation）、2025 三组预测；Effristor 复用 manifest 中已有权重，不重新训练。缺失值均值填充器只在训练集拟合；TabPFN 使用原始含缺失值输入。

TabPFN 默认使用本地 `~/.cache/tabpfn/tabpfn-v2.6-regressor-v2.6_default.ckpt`，可通过 `--tabpfn-model` 指定；不自动下载。HGB/RF 的 random_state=42，TabPFN random_state=0。软件版本、TabPFN 权重 SHA256、运行设置和完成划分记录于 `provenance.json`。历史版本随机设置和权重可能不同，不能声称精确复现旧拟合。

正式预测目录含 225 份 CSV（5 模型 × 15 划分 × 3 组），包含分数、标签及样本索引；外部集索引为行号，不额外输出临床特征或患者 ID。脚本检查分数有限、两类标签及同一划分各模型标签序列一致。早期不含样本 ID 的历史预测，仅凭标签序列不能独立证明样本身份一致。

## 2. SHAP 与固定模型特征遮蔽

该流程不重新训练模型。输出根目录为 `ivf/task/analyse/ablation_outputs/`，已有正式结果位于 `shap/dataset2/`、`mask/dataset2/`、`plot/`；`smoke/` 仅用于重新执行小规模验证，历史冒烟结果已清理。

### 2.1 独立绘图

```bash
python -m ivf.task.analyse.ablation --stages plot
# 展示全部 0–44 档，另存新目录：
python -m ivf.task.analyse.ablation --stages plot --plot-counts all \
  --output ivf/task/analyse/ablation_outputs/plot/all_counts
```

只读取 `mask/dataset2/mask_all.csv`，无需 manifest、模型或 Excel。可用 `--mask-results` 指定其他结果，`--seeds` 筛选划分；要求选定实验网格完整且无重复、非有限指标。默认展示 0/1/3/5/10/15/20/30/40/44 档，使用 `--plot-counts` 调整。

输出仅为 `plot/mask_boxplots.pdf`，两行分别为 validate/external，两列分别为 ROC-AUC/AP，比较三种遮蔽顺序；白色圆点为均值，不画 delta，不额外输出 PNG/CSV/JSON。已有图需 `--overwrite` 才能覆盖。

### 2.2 生成共享 SHAP 排名

```bash
python -m ivf.task.analyse.ablation \
  --manifest ivf/task/analyse/ablation_manifest.json --seeds 2 --stages shap \
  --device cuda:4 --background-size 32 --explain-size 128 --shap-permutations 5 \
  --output ivf/task/analyse/ablation_outputs/shap/dataset2
```

使用 dataset_2 的基准模型：`2026-04-15_18-06-00_val_metric/epoch=123-step=19220-val_metric=0.09263389557600021`。排名保存为 `ablation_outputs/shap/dataset2/seed_2/ranking.csv`。默认背景样本 32、解释样本 128、5 次排列；背景与解释样本仅从该划分训练集抽取。

SHAP 使用 PermutationExplainer + Independent masker，解释 sigmoid 后的异常受精概率，按 mean(abs(SHAP)) 排名；NaN 与 NaN 在 masker 中视为相等，缺失输入仍交给原网络处理。

### 2.3 使用共享排名执行遮蔽

```bash
python -m ivf.task.analyse.ablation \
  --manifest ivf/task/analyse/ablation_manifest.json --stages mask \
  --ranking-file ivf/task/analyse/ablation_outputs/shap/dataset2/seed_2/ranking.csv \
  --device cuda:4 --output ivf/task/analyse/ablation_outputs/mask/dataset2
```

有 `--ranking-file` 时不重复计算 SHAP，15 个划分共用重要性降序、升序及随机对照顺序（默认 `--order-seed 2`、`--random-repeats 1`）；各划分独立用自己的训练数据计算替换值。默认遮蔽 0–44 个特征，至少保留一个，可通过 `--counts`、`--orders`、`--random-repeats` 调整。

遮蔽不改变列数、顺序或模型权重：连续变量用训练集中位数，类别用训练集众数，全缺失列保持 NaN。`--categorical-indices auto` 以训练集非缺失唯一值数 <10 推断类别，正式研究建议显式指定索引。`--mask-mode nan` 是另一类缺失特征实验，应使用新输出目录。

原始 AnnData 没有临床特征名，脚本按 Excel 表头和列顺序核查特征数；需确保原数据列序确实一致。共享排名可能让 dataset_2 的训练样本出现在其他划分验证集中，因此内部消融仅作探索性分析；2025 数据不参与排名。

### 2.4 小规模验证、覆盖规则与产物

```bash
python -m ivf.task.analyse.ablation \
  --manifest ivf/task/analyse/ablation_manifest.json --seeds 0 \
  --counts 0,1 --smoke-samples 32 \
  --background-size 2 --explain-size 2 --shap-permutations 1 \
  --device cuda:4 --output ivf/task/analyse/ablation_outputs/smoke/initial
```

冒烟参数仅验证流程，不用于科学结论。正式分析需检查 SHAP 排名稳定性，适当增加抽样及排列次数。

支持 `--stages shap`、`mask`、`plot`；默认执行 shap + mask，plot 必须独立执行。未指定共享排名时，可为各划分独立计算或读取其排名。配置与输入 SHA256 记录在 `config.json`，兼容旧配置中未使用的训练参数；同配置覆盖须 `--overwrite`，不同输入/配置使用新目录。GPU 设备允许变化，数据哈希必须一致。未传 `--output` 时单阶段使用对应目录、多阶段使用 `experiments/`。

- SHAP：`config.json`、`ranking.csv`、`shap.npz`、`shap_metadata.json`、SHAP 图。NPZ 含归因、样本、背景/解释索引和加性残差。
- 每划分 mask：`config.json`、排名副本、`feature_orders.json`、`mask_reference.csv`、`mask_metrics.csv` 及性能曲线。
- mask 根目录：`mask_all.csv`、`mask_summary.csv`，记录 ROC-AUC/AP 及相对全特征基准的 `delta_roc_auc`、`delta_average_precision`。

绘图拒绝含 `pr_auc` 的结果，避免将梯形 PR-AUC 误标为 AP；两者不能改列名互换，需要正确的 AP 结果或重新推理。正式图表、排名和指标可提交；仅忽略冒烟产物、权重 `*.pt` 及含样本输入的 `shap.npz`（以 `.gitignore` 为准）。

## 3. 单模型预测与解释分析

旧版逐个列出的 15 组模型路径已统一由 `ablation_manifest.json` 管理。正式多模型 benchmark 使用第 1 节入口，避免手工预测覆盖其中某一个模型的结果。

单模型临时导出示例（从仓库根目录）：

```bash
# 从 manifest 获取某一划分的路径；输出到临时目录，不覆盖正式 benchmark。
mkdir -p /tmp/effristor-predictions
python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

entry = json.loads(Path('ivf/task/analyse/ablation_manifest.json').read_text())['2']
command = [sys.executable, '-m', 'ivf.task.analyse.run', 'predict-and-evaluate',
           '--model_path', entry['model'], '--adata_path', entry['dataset']]
subprocess.run(command + ['--output_path', '/tmp/effristor-predictions/validation.csv'], check=True)
subprocess.run(command + ['--xlsx_path', 'ivf/task/data.2025.xlsx',
                          '--output_path', '/tmp/effristor-predictions/test2025.csv'], check=True)
PY
```

原有解释脚本使用相对路径，下面这组命令须切换到分析目录执行：

```bash
cd ivf/task/analyse
python run.py interpret \
  --model_path ../.ivf/2026-04-15_18-06-00_val_metric/epoch=123-step=19220-val_metric=0.09263389557600021 \
  --adata_path ../dataset/dataset_2
```

`analyse.ipynb` 包含历史 ROC-AUC 比较与解释分析，`distribution_analyse.ipynb` 用于特征分布分析；从分析目录启动 notebook 以匹配相对路径。当前 benchmark 的标准表格和箱线图以第 1 节脚本为准。

## 4. 结果解释边界

- 15 次结果来自重复随机划分，不是独立交叉验证折或 15 个独立临床队列。测试集相同且训练划分重叠，SD 是描述性离散度，不是置信区间；本汇总不自动新增显著性检验。
- 内部验证集曾用于模型选择，不是无偏的最终泛化性能估计；2025 数据不参与本流程选择，用于独立外部评估。训练集高分本身不代表模型更优，应结合验证及测试表现判断拟合与泛化差距。
- 遮蔽可能制造分布外输入，Independent masker 不保留特征条件依赖；SHAP 和消融不是因果效应或临床治疗建议。
