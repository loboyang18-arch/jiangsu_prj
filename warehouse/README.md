# Parquet 建库（warehouse）

本目录为“建库落地”产物输出目录，**以总表为主源**，江苏仅用于扩展段与特有指标补齐，并可作为校验来源。

## 目录结构（约定）

- `warehouse/stage/`：从 Excel 抽取出的**宽表** parquet（中间产物）
  - `total_wide_15m.parquet`：从 `总表/` 抽取（主源）
  - `jiangsu_supplement_wide_15m.parquet`：从 `江苏/电价数据/` 抽取（补充/校验）
- `warehouse/dwd/`：明细层（长表）
  - `dwd_timeseries_15m.parquet`
- `warehouse/dim/`：维表
  - `dim_metric.parquet`
  - `dim_source_field_map.parquet`
- `warehouse/audit/`：审计输出
  - `coverage_report.csv`
  - `feature_missing_stats_full_v0.csv`
  - `feature_missing_stats_full_v0_zero_run_ge48.csv`
  - `feature_missing_stats_full_v0_zero_run_ge96.csv`

## 一键构建

```bash
python scripts/build_parquet_warehouse.py \
  --plan scripts/warehouse_plan.json \
  --mode full
```

## 缺失统计（含连续零值规则）

```bash
python scripts/audit_feature_missing.py --plan scripts/warehouse_plan.json
```

## 说明

- 宽表抽取复用 `scripts/build_dataset.py`（统一频率 `--freq 15min`）。
- 长表入库只收“已明确映射”的主干指标（见 `scripts/build_parquet_warehouse.py` 中 `_canonicalize_from_meta()`），避免把未知/高维列直接入库。
- `江苏/电价数据/26年日前电价.xlsx` 为宽表（列头为 Excel 日期序列号），当前脚本不直接纳入 DWD（建议单独解析为长表后以显式 metric_id 入库）。

