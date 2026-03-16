# 脚本总览

本目录包含数据字典扫描、质量审计、统一数据集构建、特征工程与基线训练的完整流水线。建议按下列顺序使用。

---

## 脚本列表（推荐执行顺序）

| 顺序 | 脚本 | 用途 |
|------|------|------|
| 0 | `_excel_scan_utils.py` | 内部共用：Excel 时间识别、列摘要、安全读取，供其他脚本复用 |
| 1 | `scan_excel_dictionary.py` | 扫描所有 Excel，生成字段级数据字典（CSV/MD） |
| 2 | `raw_data_quality_audit.py` | 原始数据质量审计：sheet/列统计、日期缺口、只读报告 |
| 3 | `build_dataset.py` | 从异构 Excel 构建统一频率时间序列 parquet + meta JSON |
| 4 | `audit_feature_missing.py` | 导出 V0 全量缺失统计（支持“连续零值>=阈值视缺失”） |
| 5 | `feature_engineering.py` | 基于 parquet 做滞后/滚动/日历特征，输出带 y 的特征表 |
| 6 | `train_baseline.py` | 在特征表上训练 LightGBM 基线，输出模型与 val/test 指标 |
| 7 | `notebook_env_smoke_test.py` | Notebook 环境快速自检（版本 + Parquet 读取） |

---

## 0. _excel_scan_utils.py（内部共用）

- **用途**：Excel 扫描共用工具（时间列推断、列摘要、安全读取），被 `scan_excel_dictionary.py`、`build_dataset.py` 等引用。
- **输入/输出**：无独立 CLI，仅作为模块导入。
- **依赖**：pandas、numpy。

---

## 1. scan_excel_dictionary.py

- **用途**：递归扫描 `--data-root` 下所有 Excel（.xls/.xlsx/.xlsm/.xlsb），生成字段级数据字典；单文件异常不阻断整体。
- **输入**：`--data-root`（工程根目录）、`--output-dir`（输出目录）。
- **输出**：`data_dictionary.csv`、`data_dictionary.md`（写入 output-dir）。
- **依赖**：pandas、`_excel_scan_utils`。

**示例：**

```bash
python scripts/scan_excel_dictionary.py \
  --data-root "/path/to/jiangsu_prj" \
  --output-dir "/path/to/jiangsu_prj/report"
```

---

## 2. raw_data_quality_audit.py

- **用途**：对江苏/ 与 总表/ 下 Excel 做只读质量审计，统计时间列解析率、96 点/日完整性、列缺失率、疑似废弃列、按 dataset_group 的日期缺口等。
- **输入**：`--data-root`、`--output-dir`；可选 `--sample-files-per-group N`（0=全量，>0=每组只取前 N 个文件）。
- **输出**：`raw_data_quality_summary.csv`、`raw_data_quality_details.csv`、`raw_data_gap_report.csv`、`raw_data_quality_report.json`、`raw_data_quality_report.md`。
- **依赖**：pandas、numpy；可选 tabulate（Markdown 表格美化）。

**示例：**

```bash
python scripts/raw_data_quality_audit.py \
  --data-root "/path/to/jiangsu_prj" \
  --output-dir "/path/to/jiangsu_prj/report" \
  --sample-files-per-group 0
```

---

## 3. build_dataset.py

- **用途**：从异构 Excel 按统一频率（`--freq`）构建时间序列 parquet，支持 full/preview 模式、可选规则文件（`--rules`）；输出 parquet + meta JSON。
- **输入**：`--data-root`、`--output-parquet`、`--output-meta`、`--mode`（full/preview）、`--freq`（如 15min/1H/1D）；可选 `--rules <path>`。
- **输出**：指定路径的 parquet 与 meta JSON（列说明、来源统计等）；meta 中每列含 `time_source`（`column` 或 `filename`）便于追溯。
- **无时间列 sheet**：若 sheet 内无可用时间/日期列但**文件名含日期**（如 `2025-06-01.xlsx`），则按“数据按时间排序”的假设，用文件名日期 + 行数推断时间轴（1 行→当日 0 点，24 行→整点，96 行→15min×96 等），仍可抽取江南/江北价格等数值列；与“同类文件”按日期归档思路一致。
- **依赖**：pandas、numpy、`_excel_scan_utils`。

**示例：**

```bash
python scripts/build_dataset.py \
  --data-root "/path/to/jiangsu_prj" \
  --output-parquet "/path/to/jiangsu_prj/processed_data/jiangsu_dataset.parquet" \
  --output-meta "/path/to/jiangsu_prj/processed_data/jiangsu_dataset_meta.json" \
  --mode full \
  --freq 1D
```

**使用规则文件：**

```bash
python scripts/build_dataset.py ... --rules scripts/dataset_rules.example.json
```

规则文件说明见本文档末尾「规则文件 dataset_rules.example.json」。

---

## 4. audit_feature_missing.py

- **用途**：从 `warehouse/feature_ready/V0/power_market_feature_ready_wide.parquet` 导出全量缺失统计，并支持把“连续零值（如 >=48 / >=96）”计为缺失。
- **输入**：`--plan scripts/warehouse_plan.json`（读取输出路径和阈值配置）。
- **输出**：
  - `feature_missing_stats_full_v0.csv`
  - `feature_missing_stats_full_v0_zero_run_ge{threshold}.csv`
- **依赖**：pandas、pyarrow。

**示例：**

```bash
python scripts/audit_feature_missing.py --plan scripts/warehouse_plan.json
```

---

## 5. feature_engineering.py

- **用途**：基于统一频率 parquet 做特征工程，生成标签 y = target(t + horizon) 以及滞后、滚动、日历特征；必须提供 `--freq`（与 build_dataset 一致）。
- **输入**：`--input-parquet`、`--output-parquet`、`--freq`、`--horizon`（步数或时间如 15min/1H）；可选 `--target-col`、`--include-regex`、`--exclude-regex`、`--max-missing-rate`、`--output-meta`。
- **输出**：特征 parquet；可选 meta JSON。
- **依赖**：pandas、numpy。

**示例：**

```bash
python scripts/feature_engineering.py \
  --input-parquet "/path/to/jiangsu_prj/processed_data/jiangsu_dataset.parquet" \
  --output-parquet "/path/to/jiangsu_prj/feature_table/features.parquet" \
  --freq 1D \
  --horizon 1
```

---

## 6. train_baseline.py

- **用途**：读取特征 parquet，按时间划分 train/val/test，训练 LightGBM 回归，在 val 上早停，在 test 上报告指标，并与 naive_last 基线对比。
- **输入**：`--features-parquet`、`--model-dir`；可选 `--target-col`（默认 y）。
- **输出**：`model_dir/baseline_lgbm.joblib`、`model_dir/metrics.json`。
- **依赖**：pandas、numpy、lightgbm、scikit-learn、joblib。

**示例：**

```bash
python scripts/train_baseline.py \
  --features-parquet "/path/to/jiangsu_prj/feature_table/features.parquet" \
  --model-dir "/path/to/jiangsu_prj/models"
```

---

## 规则文件 dataset_rules.example.json

`build_dataset.py` 通过 `--rules <path>` 接入规则文件，将「抽取哪些 sheet/列、如何聚合、如何命名」从代码中抽离，便于审计与迭代。

**结构概览：**

- **version**：规则版本号。
- **notes**：说明文字。
- **file_rules**：数组，每项针对一类文件路径：
  - **match_path_regex**：匹配相对 `data-root` 的文件路径的正则。
  - **mode**：如 `extract_numeric` 表示抽取数值列。
  - **time_column_regex**：识别时间列的正则。
  - **exclude_columns_regex**：排除列的正则。
  - **max_metrics_per_sheet**：每 sheet 最多保留的指标数。
  - **aggregations**：聚合方式；`default` 为默认（如 mean），也可为列名或正则指定 price/weight 加权等。

未传 `--rules` 时，`build_dataset.py` 使用内置默认逻辑；传入后，匹配到的文件按对应规则抽取，便于不同目录使用不同规则。

**列排除两层**：`build_dataset.py` 会**始终**排除与质量审计一致的模板列（`Unnamed:*`、序号、index、编号）；规则中的 `exclude_columns_regex` 在此基础上再排除（如节点、ID 等），避免噪声入模。

---

## 数据质量审计启示与处理约定

基于 `raw_data_quality_audit.py` 全量扫描结论，建议遵守以下约定：

- **日期缺口**：部分 dataset_group（如「实时出清结果统合」「日前出清结果统合」）存在较多缺失天，产出 parquet 的日期范围以实际覆盖为准，不做“连续日历”假设；缺口明细见 `report/raw_data_gap_report.csv`。
- **列排除**：在规则或默认逻辑中已排除 `Unnamed:*` 与常见模板列；高缺失列可在 `feature_engineering.py` 用 `--max-missing-rate` 控制，或在规则中通过列名正则排除。
- **多节点/多列**：日前价格等多节点表建议先做区域聚合或价差特征再入模，避免全节点展开导致维度过高与过拟合。
- **时序类型标签**：若需区分 actual/forecast/plan，建议在 meta 或特征表中打上时序类型，训练时避免用“未来/计划”信息预测“实际”，防止泄露。

---

## 交付与规范

- 所有脚本均包含模块级 docstring 与主要函数/类的用途说明（中文）。
- 建议在正式实验中：`build_dataset.py` 使用 `--freq` 固定频率，`feature_engineering.py` 使用相同 `--freq`；数据集构建使用 `--mode full`，preview 仅用于调试。

更多使用说明见项目根目录 `README.md`。

---

## Notebook 环境与自检

- 建议使用项目根目录 `environment.notebook.yml` 创建独立环境，避免 `numpy/pandas/pyarrow` ABI 冲突。
- 启动 Jupyter 前可执行：

```bash
python scripts/notebook_env_smoke_test.py
```

若输出 `read_parquet` 成功，则 Notebook 可视化工具通常可直接使用。

---

## 修改记录（备注）

以下为基于**数据质量审计报告**对数据处理与文档的更新，便于交付与后续维护追溯。

| 日期 | 文件 | 修改要点 |
|------|------|----------|
| 2025-03（基于质量审计报告） | `scripts/build_dataset.py` | 新增 `DEFAULT_EXCLUDE_COL_RE`，与质量审计 TRASH_COL_RE 一致；抽取时始终排除 Unnamed、序号、index、编号；支持规则中 `exclude_columns_regex` 第二层排除。模块 docstring 与抽取逻辑处已加备注。 |
| 2025-03（基于质量审计报告） | `scripts/dataset_rules.example.json` | `notes` 说明默认排除由 build_dataset 统一处理、规则可追加排除；`exclude_columns_regex` 示例中增加 `Unnamed`；建议价格日表与汇总总表分条规则。 |
| 2025-03（基于质量审计报告） | `scripts/README.md` | 规则文件小节补充「列排除两层」说明；新增「数据质量审计启示与处理约定」小节（日期缺口、列排除、多节点聚合、时序类型标签）；新增本修改记录表。 |
| 2025-03（无时间列可归档） | `scripts/build_dataset.py` | 无时间列时用文件名日期+行数推断时间轴：新增 `DATE_IN_FILENAME_RE`、`infer_time_column_from_filename()`，行数 1/24/96/48/288 分别按 1D/1H/15min/30min/5min 生成时间；meta 增加 `time_source`（column/filename）。 |
| 2025-03（无时间列可归档） | `scripts/README.md` | build_dataset 小节补充「无时间列 sheet」「time_source」说明；修改记录表增加本行。 |
