# 江苏项目 (jiangsu_prj)

项目说明与数据目录。

## 目录结构

- **总表/** - 总表相关（事前 / 事后）
- **江苏/** - 江苏电价与电网数据
- **scripts/** - 数据字典 / 数据集构建 / 训练评估脚本
- **processed_data/** - 统一数据集产出（parquet 等）
- **feature_table/** - 特征表产出（parquet 等）
- **models/** - 训练产物（模型文件、评估结果）
- **report/** - 自动生成的统计与汇报文档

## 使用说明

克隆后按需使用各目录下的数据与脚本。

### 安装依赖

```bash
pip install -r requirements.txt
```

### ① 生成字段字典（data_dictionary）

```bash
python scripts/scan_excel_dictionary.py \
  --data-root "/Users/tim/pythonwork/jiangsu_prj" \
  --output-dir "/Users/tim/pythonwork/jiangsu_prj/report"
```

输出：
- `report/data_dictionary.md`
- `report/data_dictionary.csv`

### ② 构建统一数据集（parquet）

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

### 重要说明（严肃实验建议遵守）

- **统一频率**：务必在 `build_dataset.py` 使用 `--freq` 固化输出频率（如 `15min/1H/1D`），并在 `feature_engineering.py` 使用相同 `--freq`。
- **preview vs full**：`build_dataset.py` 的 `--mode preview` 仅用于调试抽样；正式实验使用 `--mode full`。
- **规则配置化**：如需把“抽取哪些 sheet/列、如何聚合”固化到可审计规则，参考 `scripts/dataset_rules.example.json` 并通过 `build_dataset.py --rules <path>` 接入。

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
