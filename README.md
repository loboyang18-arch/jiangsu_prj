# 江苏项目 (jiangsu_prj)

项目说明与数据目录。数据以 15 分钟粒度（96 点/日）为主，入库与建模方案见 [report/总表数据入库方案.md](report/总表数据入库方案.md)。

## 当前工程结构（整理后）

| 目录 | 说明 |
|------|------|
| **总表/** | 总表数据（事前/事后合并总表） |
| **江苏/** | 江苏电价与电网数据 |
| **scripts/** | 抽取、入库、审计、训练脚本（详见 [scripts/README.md](scripts/README.md)） |
| **warehouse/** | **主库产出**：Stage/DWD/dim/audit（Parquet），见 [warehouse/README.md](warehouse/README.md) |
| **report/** | 方案与报告：入库方案、数据分析报告、子文件夹梳理 01～09、审计报告 |
| **processed_data/** | 可选：build_dataset 直出 parquet（非 warehouse 流程时使用） |
| **feature_table/** | 可选：特征工程产出 |
| **models/** | 可选：训练产物（.joblib、评估结果） |

## 使用说明

### 安装依赖

```bash
pip install -r requirements.txt
```

### 推荐主流程：总表入库（warehouse）

按 [report/总表数据入库方案.md](report/总表数据入库方案.md) 执行，默认主窗 **2025-09-01～2025-12-31**：

```bash
# 仅总表：Stage → DWD / dim / audit
python scripts/build_parquet_warehouse.py --plan scripts/warehouse_plan.json --mode full --total-only
```

产出：`warehouse/stage/`、`warehouse/dwd/`、`warehouse/dim/`、`warehouse/audit/`。

### ① 生成字段字典（可选，可复现）

运行后生成 `report/data_dictionary.md`、`report/data_dictionary.csv`。

```bash
python scripts/scan_excel_dictionary.py \
  --data-root "/Users/tim/pythonwork/jiangsu_prj" \
  --output-dir "/Users/tim/pythonwork/jiangsu_prj/report"
```

### ①’ 原始数据质量审计（可选，可复现）

只读扫描 Excel，输出 sheet/列级统计与日期缺口；需时再跑即可复现报告：

```bash
python scripts/raw_data_quality_audit.py \
  --data-root "/Users/tim/pythonwork/jiangsu_prj" \
  --output-dir "/Users/tim/pythonwork/jiangsu_prj/report" \
  --sample-files-per-group 0
```

输出：`raw_data_quality_summary.csv`、`raw_data_quality_details.csv`、`raw_data_gap_report.csv`、`raw_data_quality_report.json`、`raw_data_quality_report.md`。可选依赖 `tabulate` 以美化 Markdown 表格。

### ② 构建统一数据集（parquet，build_dataset 直出）

```bash
python scripts/build_dataset.py \
  --data-root "/Users/tim/pythonwork/jiangsu_prj" \
  --output-parquet "/Users/tim/pythonwork/jiangsu_prj/processed_data/jiangsu_dataset.parquet" \
  --output-meta "/Users/tim/pythonwork/jiangsu_prj/processed_data/jiangsu_dataset_meta.json" \
  --mode full \
  --freq 1D
```

### ③ 特征工程 + 训练 + 评估（baseline）

```bash
python scripts/feature_engineering.py \
  --input-parquet "/Users/tim/pythonwork/jiangsu_prj/processed_data/jiangsu_dataset.parquet" \
  --output-parquet "/Users/tim/pythonwork/jiangsu_prj/feature_table/features.parquet" \
  --freq 1D \
  --horizon 1

python scripts/train_baseline.py \
  --features-parquet "/Users/tim/pythonwork/jiangsu_prj/feature_table/features.parquet" \
  --model-dir "/Users/tim/pythonwork/jiangsu_prj/models"
```

### 重要说明

- **主库优先**：正式入库与建模以 **warehouse** + [总表数据入库方案](report/总表数据入库方案.md) 为准；`build_dataset` 直出可用于单次探索或非 15min 频率。
- **统一频率**：`build_dataset.py` 与 `feature_engineering.py` 需一致 `--freq`（如 `15min`/`1D`）。
- **规则配置**：参考 `scripts/dataset_rules.example.json`，`build_dataset.py --rules <path>` 接入。

## 整理说明（已删除的临时/冗余文件）

以下已删除，便于后续只维护必要文件；需要时可由脚本重新生成：

- **report/**：`数据汇总报告_*.md`、`总表与江苏数据梳理.md`、`data_folder_structure.md`、`raw_data_inventory_and_modeling.md`、`file_index.csv`、`merged_tables_time_range.csv`、`raw_data_quality_*`、`data_dictionary.*`（由 `scan_excel_dictionary.py` / `raw_data_quality_audit.py` 复现）。
- **processed_data/**：`jiangsu_dataset_meta.json`（由 `build_dataset.py` 复现）。
- **feature_table/**：`features_meta.json`（由 `feature_engineering.py` 复现）。
- **models/**：`metrics.json`（由 `train_baseline.py` 复现）。

**保留的 report 核心文档**：`总表数据入库方案.md`、`数据分析报告.md`、`总表入库审计报告.md`、`子文件夹梳理/01～09`。

## 推送至 GitHub 私有仓库

1. 在 [GitHub](https://github.com/new) 新建一个 **私有** 仓库（不要勾选 “Add a README”）。
2. 在本地执行（将 `YOUR_USERNAME` 和 `YOUR_REPO` 换成你的用户名和仓库名）：

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

若使用 SSH：

```bash
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```
