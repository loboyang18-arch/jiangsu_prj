"""
基于总表数据入库方案，落地 Parquet 建库（L1 原始标准化层 + L2 建模可用层）。

产出：
- warehouse/stage/*_wide_15m.parquet：抽取得到的宽表（timestamp + 多指标列）
- warehouse/dwd/dwd_timeseries_15m.parquet：L1 长表（ts, trade_date, hh_index, metric_id, value, quality_flag, fill_method, source_*）
- warehouse/dim/dim_metric.parquet：字段注册表（含 valid_start_date、feature_name_cn 等）
- warehouse/dim/dim_source_field_map.parquet：源字段映射
- warehouse/feature_ready/V0/power_market_feature_ready_wide.parquet：L2 V0 主窗宽表（2025-09-01～12-31，带 _qflag）
- warehouse/audit/*.csv：覆盖与审计

使用：
python scripts/build_parquet_warehouse.py --plan scripts/warehouse_plan.json --mode full --total-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

# 质量标记（与方案六一致）
QUALITY_FLAG_ORIGINAL_VALID = 0
QUALITY_FLAG_STRUCTURALLY_INVALID = 4
QUALITY_FLAG_NOT_STARTED_YET = 5
VALID_START_STORAGE = "2025-12-10"
VALID_START_COAL = "2025-12-23"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")


def _load_plan(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as r:
        return json.load(r)


def _load_feature_registry(plan: Dict) -> Dict[str, Dict[str, Any]]:
    """从 plan 或默认路径加载 feature_registry，返回 feature_code -> { feature_name_cn, valid_start_date, ... }"""
    reg_path = plan.get("paths", {}).get("feature_registry")
    if not reg_path or not os.path.isfile(reg_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reg_path = os.path.join(script_dir, "feature_registry.json")
    if not os.path.isfile(reg_path):
        return {}
    with open(reg_path, "r", encoding="utf-8") as r:
        data = json.load(r)
    return data.get("features", {})


def _build_wide(data_root: str, out_parquet: str, out_meta: str, mode: str, freq: str) -> None:
    _ensure_dir(os.path.dirname(out_parquet))
    _ensure_dir(os.path.dirname(out_meta))
    cmd = [
        os.path.join(os.path.dirname(__file__), "build_dataset.py"),
        "--data-root",
        data_root,
        "--output-parquet",
        out_parquet,
        "--output-meta",
        out_meta,
        "--mode",
        mode,
        "--freq",
        freq,
        "--resample-agg",
        "mean",
    ]
    # 用 python 运行脚本，避免 shebang/权限问题
    cmd = ["python", *cmd]
    _run(cmd)


@dataclass(frozen=True)
class CanonicalField:
    metric_id: str
    region_id: str
    unit: str
    phase: str


def _canonicalize_from_meta(metric_name: str, mm: Dict) -> Optional[CanonicalField]:
    """
    将 build_dataset 输出的 metric_meta（含 source_file/sheet/source_column/phase/region）映射为规范 metric_id。
    优先用 meta 中的 source_column（Excel 原始列名）匹配，无则用 metric_name（宽表列名）。
    """
    rel = mm.get("source_file", "")
    sheet = mm.get("sheet", "")
    phase = mm.get("phase", "")
    region = mm.get("region", "") or "汇总"
    # 优先用原始列名匹配，便于总表入库可追溯
    col_raw = mm.get("source_column", "") or metric_name

    # rel 为相对 data_root 的路径（如 事前/事前_汇总总表合集/受电计划_合并总表.xlsx）
    # 价格（总表事后公开出清）
    # 说明：
    # - 日前：build_dataset 已对「江南/江北分区价格（元/MWh）」做 mean 聚合，作为分区价。
    # - 实时：我们区分「终发布分区价格」与「节点边际电价均价」，四列均入库，但分配不同 metric_id。

    if "事后_汇总总表合集/日前出清结果（公开）_合并总表.xlsx" in rel:
        # 先匹配节点均价，避免「江南分区价格」误匹配到「江南分区节点边际电价均价」
        if "江南分区节点边际电价均价" in col_raw and "元/MWh" in col_raw:
            return CanonicalField("price_dayahead_jn_node", "江南", "元/MWh", "事后")
        if "江北分区节点边际电价均价" in col_raw and "元/MWh" in col_raw:
            return CanonicalField("price_dayahead_jb_node", "江北", "元/MWh", "事后")
        if "江南分区价格" in col_raw and "节点" not in col_raw and "元/MWh" in col_raw:
            return CanonicalField("price_dayahead_jn", "江南", "元/MWh", "事后")
        if "江北分区价格" in col_raw and "节点" not in col_raw and "元/MWh" in col_raw:
            return CanonicalField("price_dayahead_jb", "江北", "元/MWh", "事后")

    if "事后_汇总总表合集/实时出清结果（公开）_合并总表.xlsx" in rel:
        # 先匹配节点均价
        if "江南分区节点边际电价均价" in col_raw and "元/MWh" in col_raw:
            return CanonicalField("price_realtime_jn_node", "江南", "元/MWh", "事后")
        if "江北分区节点边际电价均价" in col_raw and "元/MWh" in col_raw:
            return CanonicalField("price_realtime_jb_node", "江北", "元/MWh", "事后")
        # 分区价（终发布口径；列名可能含 (终发布) 或仅「江南/江北分区价格（元/MWh）」）
        if "江南分区价格" in col_raw and "节点" not in col_raw and "元/MWh" in col_raw:
            return CanonicalField("price_realtime_jn_final", "江南", "元/MWh", "事后")
        if "江北分区价格" in col_raw and "节点" not in col_raw and "元/MWh" in col_raw:
            return CanonicalField("price_realtime_jb_final", "江北", "元/MWh", "事后")

    # 实际受电/负荷（总表事后汇总）
    if "事后_汇总总表合集/实际受电情况_合并总表.xlsx" in rel:
        if "实际发布电力（MW）" in col_raw:
            return CanonicalField("receive_actual_huadong", "华东", "MW", "事后")
    if "事后_汇总总表合集/实际系统负荷_合并总表.xlsx" in rel:
        if "实际发布电力（MW）" in col_raw:
            return CanonicalField("load_actual_total", "汇总", "MW", "事后")

    # 事前：受电计划、短期负荷预测、备用（总表事前汇总）
    if "事前_汇总总表合集/受电计划_合并总表.xlsx" in rel:
        if "边界信息发布电力（MW）" in col_raw:
            return CanonicalField("receive_plan_boundary", "汇总", "MW", "事前")
        if "出清发布电力（MW）" in col_raw:
            return CanonicalField("receive_plan_clearing", "汇总", "MW", "事前")
    if "事前_汇总总表合集/短期系统负荷预测_合并总表.xlsx" in rel:
        if "边界信息发布电力（MW）" in col_raw:
            return CanonicalField("load_forecast_boundary", "汇总", "MW", "事前")
        if "出清发布电力（MW）" in col_raw:
            return CanonicalField("load_forecast_clearing", "汇总", "MW", "事前")
    if "事前_汇总总表合集/正负备用空间_合并总表.xlsx" in rel:
        if "正备用(MW)" in col_raw:
            return CanonicalField("reserve_positive", "汇总", "MW", "事前")
        if "负备用(MW)" in col_raw:
            return CanonicalField("reserve_negative", "汇总", "MW", "事前")

    # 事后：江北/江南分区（总表事后江北/江南）
    if "事后_江北总表合集/" in rel or "事后_江南总表合集/" in rel:
        if "实际储能固定出力总值" in rel and "实际数据（MW）" in col_raw:
            return CanonicalField("storage_actual", region, "MW", "事后")
        if "实际煤电固定出力总值" in rel and "实际数据（MW）" in col_raw:
            return CanonicalField("coal_actual", region, "MW", "事后")
        if "实际统调风光情况_光伏" in rel and "实际发布电力（MW）" in col_raw:
            return CanonicalField("pv_actual", region, "MW", "事后")
        if ("实际统调风光情况_风力" in rel or "实际统调风光情况_江北分区" in rel) and "实际发布电力（MW）" in col_raw:
            return CanonicalField("wind_actual", region, "MW", "事后")
        if "实际电网运行情况" in rel and "实际总值（MW）" in col_raw:
            return CanonicalField("gas_actual", region, "MW", "事后")

    # 江苏补齐（仅总表入库时可注释或保留，不影响总表）
    if "江苏/电价数据/26年实时价格/" in rel and "实时市场加权均价（元/MWh）" in col_raw:
        return CanonicalField("price_realtime_weighted_public", "汇总", "元/MWh", "事后")
    if "江苏/电价数据/26年日前电价.xlsx" in rel:
        return None
    if "江苏/电价数据/26年电网实际边界" in rel:
        if "实际受电情况-华东" in rel and "实际发布电力（MW）" in col_raw:
            return CanonicalField("receive_actual_huadong", "华东", "MW", "事后")
        if "实际系统负荷" in rel and "实际发布电力（MW）" in col_raw:
            return CanonicalField("load_actual_total", "汇总", "MW", "事后")
        if "实际燃机固定出力总值-汇总" in rel and ("实际总值（MW）" in col_raw or "实际总值" in col_raw):
            return CanonicalField("gas_actual", "汇总", "MW", "事后")
        if "储能发电计划实际数据-汇总" in rel and "实际数据（MW）" in col_raw:
            return CanonicalField("storage_actual", "汇总", "MW", "事后")
        if "煤电发电计划实际数据-汇总" in rel and "实际数据（MW）" in col_raw:
            return CanonicalField("coal_actual", "汇总", "MW", "事后")
        if "实际统调风光情况-光伏" in rel and ("汇总实际发布电力（MW）" in col_raw or "实际发布电力（MW）" in col_raw):
            return CanonicalField("pv_actual", "汇总", "MW", "事后")
        if "实际统调风光情况-风力" in rel and ("汇总实际发布电力（MW）" in col_raw or "实际发布电力（MW）" in col_raw):
            return CanonicalField("wind_actual", "汇总", "MW", "事后")

    return None


def _wide_to_dwd(
    wide_parquet: str,
    meta_json: str,
    source_system: str,
    feature_registry: Dict[str, Dict[str, Any]],
    etl_batch_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    宽表 + meta → L1 长表（可识别的 canonical 字段，保留缺失行并打 quality_flag）。
    返回：(dwd, dim_metric, dim_source_field_map)
    """
    df = pd.read_parquet(wide_parquet)
    with open(meta_json, "r", encoding="utf-8") as r:
        meta = json.load(r)
    metric_meta: Dict[str, Dict] = meta["metric_meta"]

    rows: List[pd.DataFrame] = []
    metric_rows: Dict[str, Dict] = {}
    map_rows: List[Dict] = []

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"])
    # 统一时点口径：交易日 1~96 点对应 00:15~24:00。
    # 即 timestamp=次日 00:00 视为上一交易日第 96 点（24:00），
    # 避免把 00:00 当作当日首点。
    ts_series = df["timestamp"]
    is_midnight = (ts_series.dt.hour == 0) & (ts_series.dt.minute == 0)
    trade_dates = ts_series.dt.normalize() - pd.to_timedelta(is_midnight.astype(int), unit="D")
    hh_index = np.where(
        is_midnight,
        96,
        ts_series.dt.hour * 4 + ts_series.dt.minute // 15,
    ).astype("int64")

    for col in df.columns:
        if col == "timestamp":
            continue
        mm = metric_meta.get(col)
        if not mm:
            continue
        cf = _canonicalize_from_meta(col, mm)
        if not cf:
            continue

        metric_id = cf.metric_id
        region_id = cf.region_id
        unit = cf.unit
        phase = cf.phase

        reg = feature_registry.get(metric_id, {})
        metric_rows.setdefault(
            metric_id,
            {
                "metric_id": metric_id,
                "feature_name_cn": reg.get("feature_name_cn", metric_id),
                "unit": unit,
                "phase": phase,
                "valid_start_date": reg.get("valid_start_date"),
                "valid_end_date": reg.get("valid_end_date"),
                "allow_fill": reg.get("allow_fill", "Y"),
                "fill_strategy": reg.get("fill_strategy", ""),
                "is_model_default_enabled": reg.get("is_model_default_enabled", False),
            },
        )

        source_col_display = mm.get("source_column", "") or col
        map_rows.append(
            {
                "metric_id": metric_id,
                "region_id": region_id,
                "source_system": source_system,
                "source_file": mm.get("source_file", ""),
                "source_sheet": mm.get("sheet", ""),
                "source_column": source_col_display,
                "time_source": mm.get("time_source", ""),
                "phase": mm.get("phase", ""),
            }
        )

        s = pd.to_numeric(df[col], errors="coerce")
        # 保留所有行（含 value 为 NaN），做缺失与质量标记
        is_missing_value = s.isna()
        valid_start = reg.get("valid_start_date")
        is_before_valid = pd.Series(False, index=s.index)
        if valid_start and metric_id == "storage_actual":
            is_before_valid = trade_dates < pd.Timestamp(valid_start)
        elif valid_start and metric_id == "coal_actual":
            is_before_valid = trade_dates < pd.Timestamp(valid_start)
        quality_flag = pd.Series(QUALITY_FLAG_ORIGINAL_VALID, index=s.index, dtype="int64")
        quality_flag = quality_flag.mask(is_missing_value & is_before_valid, QUALITY_FLAG_NOT_STARTED_YET)
        quality_flag = quality_flag.mask(is_missing_value & ~is_before_valid, QUALITY_FLAG_STRUCTURALLY_INVALID)

        tmp = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "trade_date": trade_dates,
                "hh_index": hh_index,
                "metric_id": metric_id,
                "region_id": region_id,
                "value": s,
                "unit": unit,
                "phase": phase,
                "source_system": source_system,
                "source_file": mm.get("source_file", ""),
                "source_sheet": mm.get("sheet", ""),
                "source_column": source_col_display,
                "time_source": mm.get("time_source", ""),
                "is_missing_row": False,
                "is_missing_value": is_missing_value,
                "is_filled": False,
                "fill_method": "",
                "quality_flag": quality_flag,
                "etl_batch_id": etl_batch_id,
            }
        )
        rows.append(tmp)

    if rows:
        dwd = pd.concat(rows, ignore_index=True)
    else:
        dwd = pd.DataFrame(
            columns=[
                "timestamp",
                "trade_date",
                "hh_index",
                "metric_id",
                "region_id",
                "value",
                "unit",
                "phase",
                "source_system",
                "source_file",
                "source_sheet",
                "source_column",
                "time_source",
                "is_missing_row",
                "is_missing_value",
                "is_filled",
                "fill_method",
                "quality_flag",
                "etl_batch_id",
            ]
        )

    if metric_rows:
        dim_metric = pd.DataFrame(list(metric_rows.values())).sort_values(["metric_id"]).reset_index(drop=True)
    else:
        dim_metric = pd.DataFrame(
            columns=[
                "metric_id",
                "feature_name_cn",
                "unit",
                "phase",
                "valid_start_date",
                "valid_end_date",
                "allow_fill",
                "fill_strategy",
                "is_model_default_enabled",
            ]
        )
    if map_rows:
        dim_map = pd.DataFrame(map_rows).sort_values(
            ["metric_id", "region_id", "source_system", "source_file", "source_sheet", "source_column"]
        )
    else:
        dim_map = pd.DataFrame(
            columns=[
                "metric_id",
                "region_id",
                "source_system",
                "source_file",
                "source_sheet",
                "source_column",
                "time_source",
                "phase",
            ]
        )
    return dwd, dim_metric, dim_map


def _apply_primary_policy(dwd_total: pd.DataFrame, dwd_js: pd.DataFrame, cutoff_inclusive: str) -> pd.DataFrame:
    """
    主源=总表：覆盖到 cutoff_inclusive（含）；江苏仅用于 cutoff 之后补齐（或总表缺失时可补齐）。
    """
    cutoff = pd.Timestamp(f"{cutoff_inclusive} 23:59:59")
    # 总表全量保留
    keep_total = dwd_total.copy()
    # 江苏：仅保留 timestamp > cutoff，且不与总表同键重复覆盖
    js = dwd_js[dwd_js["timestamp"] > cutoff].copy()
    # 同键去重（若总表缺失该键，江苏可补；此处只需要防止扩展段之外的覆盖）
    return pd.concat([keep_total, js], ignore_index=True).sort_values(["timestamp", "metric_id", "region_id"]).reset_index(drop=True)


def _build_feature_ready_wide_v0(
    dwd: pd.DataFrame,
    out_path: str,
    window_start: str = "2025-09-01",
    window_end: str = "2025-12-31",
    drop_feature_if_empty_days_gt: int = 10,
    drop_log_csv: str = "",
) -> None:
    """
    L2 建模宽表 V0：主窗 2025-09-01～2025-12-31，仅 quality_flag < 4，宽表 + 每列 _qflag。
    """
    if dwd.empty or "trade_date" not in dwd.columns or "quality_flag" not in dwd.columns:
        return
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)
    sub = dwd[(dwd["trade_date"] >= start) & (dwd["trade_date"] <= end)].copy()
    sub = sub[sub["quality_flag"] < QUALITY_FLAG_STRUCTURALLY_INVALID]
    if sub.empty:
        return
    # 宽表：索引 (ts, trade_date, hh_index)，列 = 各 (metric_id, region_id) 的值列 + 对应 _qflag 列
    sub["metric_region"] = sub["metric_id"].astype(str) + "_" + sub["region_id"].astype(str)
    value_pivot = sub.pivot_table(
        index=["timestamp", "trade_date", "hh_index"],
        columns="metric_region",
        values="value",
        aggfunc="first",
    )
    qflag_pivot = sub.pivot_table(
        index=["timestamp", "trade_date", "hh_index"],
        columns="metric_region",
        values="quality_flag",
        aggfunc="first",
    )
    # 若某特征在窗口内被 quality_flag 过滤后整列无可用值，则不纳入 V0。
    keep_cols = [c for c in value_pivot.columns if value_pivot[c].notna().any()]
    dropped_rows: List[Dict[str, Any]] = []
    # 新规则：在要求时段中，按日统计“整天全空”，超过阈值则剔除该特征。
    if drop_feature_if_empty_days_gt is not None and int(drop_feature_if_empty_days_gt) >= 0:
        threshold = int(drop_feature_if_empty_days_gt)
        for c in list(keep_cols):
            # 对每个 trade_date，若该特征 96 点（或该交易日全部点）全空，则记为 1 天全空
            day_all_nan = value_pivot[c].groupby(level="trade_date").apply(lambda x: x.isna().all())
            empty_days = int(day_all_nan.sum())
            if empty_days > threshold:
                keep_cols.remove(c)
                dropped_rows.append(
                    {
                        "metric_region": c,
                        "empty_days_in_window": empty_days,
                        "threshold": threshold,
                        "window_start": str(window_start),
                        "window_end": str(window_end),
                        "drop_reason": "empty_days_exceed_threshold",
                    }
                )
    if dropped_rows:
        print(f"[build_parquet] V0 特征剔除：{len(dropped_rows)} 列（按日全空天数>{int(drop_feature_if_empty_days_gt)}）")
        for row in dropped_rows:
            print(f"[build_parquet]   - {row['metric_region']} | empty_days={row['empty_days_in_window']}")
        if drop_log_csv:
            os.makedirs(os.path.dirname(drop_log_csv), exist_ok=True)
            pd.DataFrame(dropped_rows).sort_values(["empty_days_in_window", "metric_region"], ascending=[False, True]).to_csv(
                drop_log_csv, index=False
            )
    if not keep_cols:
        return
    value_pivot = value_pivot[keep_cols]
    qflag_pivot = qflag_pivot[keep_cols].add_suffix("_qflag")
    wide = value_pivot.join(qflag_pivot)
    wide = wide.reset_index()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wide.to_parquet(out_path, index=False)


def _coverage_report(dwd: pd.DataFrame) -> pd.DataFrame:
    if dwd.empty:
        return pd.DataFrame(
            columns=[
                "metric_id",
                "region_id",
                "start_ts",
                "end_ts",
                "days",
                "points",
                "missing_points_ratio",
                "missing_value_ratio",
            ]
        )
    g = dwd.groupby(["metric_id", "region_id"], dropna=False)
    out = []
    for (m, r), dfm in g:
        ts = pd.to_datetime(dfm["timestamp"])
        start, end = ts.min(), ts.max()
        points = len(dfm)
        # 以 trade_date 为准，遵循 00:15~24:00 的交易日定义
        td = pd.to_datetime(dfm["trade_date"], errors="coerce")
        days = td.dt.normalize().nunique()
        # 粗略缺点率：以 days*96 为参照（适用于 15min 网格）
        denom = max(days * 96, 1)
        out.append(
            {
                "metric_id": m,
                "region_id": r,
                "start_ts": start,
                "end_ts": end,
                "days": int(days),
                "points": int(points),
                "missing_points_ratio": float(max(0.0, 1.0 - points / denom)),
                "missing_value_ratio": float(pd.to_numeric(dfm["value"], errors="coerce").isna().mean()),
            }
        )
    return pd.DataFrame(out).sort_values(["metric_id", "region_id"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build parquet warehouse (total as primary, jiangsu as supplement).")
    ap.add_argument("--plan", required=True, help="Path to scripts/warehouse_plan.json")
    ap.add_argument("--mode", choices=["full", "preview"], default="full")
    ap.add_argument("--total-only", dest="total_only", action="store_true", help="仅总表入库：不跑江苏 data_root，DWD/dim 仅来自总表")
    args = ap.parse_args()
    total_only = args.total_only

    plan = _load_plan(args.plan)
    print(f"[build_parquet] 使用计划文件: {args.plan}")
    print(f"[build_parquet] 模式: {args.mode} | 仅总表: {total_only}")

    out_dir = plan["paths"]["out_dir"]
    _ensure_dir(out_dir)
    _ensure_dir(os.path.join(out_dir, "stage"))
    _ensure_dir(os.path.join(out_dir, "dwd"))
    _ensure_dir(os.path.join(out_dir, "dim"))
    _ensure_dir(os.path.join(out_dir, "audit"))

    freq = plan["stage"]["freq"]
    total = plan["stage"]["total"]
    js_list = plan["stage"].get("jiangsu_supplement", [])
    feature_registry = _load_feature_registry(plan)
    etl_batch_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    # 1) 总表 wide
    print(f"[build_parquet] 1/4 构建总表宽表: data_root={total['data_root']}")
    _build_wide(total["data_root"], total["wide_parquet"], total["meta_json"], args.mode, freq)
    print(f"[build_parquet] 总表宽表完成: {total['wide_parquet']}")

    dwd_js_parts: List[pd.DataFrame] = []
    dim_metric_js_parts: List[pd.DataFrame] = []
    dim_map_js_parts: List[pd.DataFrame] = []
    if not total_only:
        print("[build_parquet] 1b/4 构建江苏补齐宽表...")
        for item in js_list:
            print(f"[build_parquet]  - Jiangsu wide: name={item.get('name', '')} root={item['data_root']}")
            _build_wide(item["data_root"], item["wide_parquet"], item["meta_json"], args.mode, freq)
            dwd_js_i, dim_metric_js_i, dim_map_js_i = _wide_to_dwd(
                item["wide_parquet"], item["meta_json"], "江苏", feature_registry, etl_batch_id
            )
            dwd_js_parts.append(dwd_js_i)
            dim_metric_js_parts.append(dim_metric_js_i)
            dim_map_js_parts.append(dim_map_js_i)
        print("[build_parquet] 江苏补齐宽表 & DWD 转换完成。")

    # 2) wide -> L1 dwd long（含 trade_date / hh_index / quality_flag）
    print("[build_parquet] 2/4 总表宽表展开为 L1 DWD 长表...")
    dwd_total, dim_metric_total, dim_map_total = _wide_to_dwd(
        total["wide_parquet"], total["meta_json"], "总表", feature_registry, etl_batch_id
    )
    print(f"[build_parquet] 总表 DWD: rows={len(dwd_total)}, dim_metric={len(dim_metric_total)}, dim_map={len(dim_map_total)}")

    dwd_js = pd.concat(dwd_js_parts, ignore_index=True) if dwd_js_parts else pd.DataFrame()
    _dim_metric_cols = [
        "metric_id", "feature_name_cn", "unit", "phase",
        "valid_start_date", "valid_end_date", "allow_fill", "fill_strategy", "is_model_default_enabled",
    ]
    dim_metric_js = (
        pd.concat(dim_metric_js_parts, ignore_index=True).drop_duplicates(["metric_id"]).reset_index(drop=True)
        if dim_metric_js_parts
        else pd.DataFrame(columns=_dim_metric_cols)
    )
    dim_map_js = pd.concat(dim_map_js_parts, ignore_index=True) if dim_map_js_parts else pd.DataFrame()

    # 3) 仅总表则直接用 total；否则按主源策略合并
    if total_only:
        print("[build_parquet] 3/4 仅总表模式: 不使用江苏补齐数据。")
        dwd = dwd_total
    else:
        print("[build_parquet] 3/4 合并总表 + 江苏补齐（总表为主源）...")
        cutoff = plan["build_policy"]["total_coverage_end_date_inclusive"]
        dwd = _apply_primary_policy(dwd_total, dwd_js, cutoff_inclusive=cutoff)
        print(f"[build_parquet] 合并后 DWD: rows={len(dwd)}")

    # 4) write outputs
    print("[build_parquet] 4/4 写出 DWD 与维表...")
    dwd_out = plan["outputs"]["dwd_timeseries_15m_parquet"]
    dim_metric_out = plan["outputs"]["dim_metric_parquet"]
    dim_map_out = plan["outputs"]["dim_source_field_map_parquet"]

    _ensure_dir(os.path.dirname(dwd_out))
    _ensure_dir(os.path.dirname(dim_metric_out))
    _ensure_dir(os.path.dirname(dim_map_out))

    dwd.to_parquet(dwd_out, index=False)
    dim_metric = (
        dim_metric_total
        if total_only
        else pd.concat([dim_metric_total, dim_metric_js], ignore_index=True).drop_duplicates(["metric_id"]).sort_values(["metric_id"])
    )
    dim_map = dim_map_total if total_only else pd.concat([dim_map_total, dim_map_js], ignore_index=True)
    dim_metric.to_parquet(dim_metric_out, index=False)
    dim_map.to_parquet(dim_map_out, index=False)

    # 5) L2 V0 主窗宽表（2025-09-01～2025-12-31）
    v0_out = plan.get("outputs", {}).get("feature_ready_v0_parquet")
    if v0_out:
        window = plan.get("sample_windows", {}).get("v0", {})
        ws = window.get("start", "2025-09-01")
        we = window.get("end", "2025-12-31")
        drop_if_empty_days_gt = int(window.get("drop_feature_if_empty_days_gt", 10))
        drop_log_csv = plan.get("outputs", {}).get(
            "feature_drop_v0_log_csv",
            os.path.join(out_dir, "audit", "feature_drop_v0_log.csv"),
        )
        print(f"[build_parquet] 5/5 生成 L2 V0 主窗宽表: {ws}～{we}")
        _build_feature_ready_wide_v0(
            dwd,
            v0_out,
            window_start=ws,
            window_end=we,
            drop_feature_if_empty_days_gt=drop_if_empty_days_gt,
            drop_log_csv=drop_log_csv,
        )
        if os.path.isfile(v0_out):
            print(f"Wrote feature_ready V0: {v0_out}")

    # 6) audits
    cov = _coverage_report(dwd)
    cov.to_csv(plan["outputs"]["coverage_report_csv"], index=False)

    print(f"Wrote DWD: {dwd_out} (rows={len(dwd)})")
    print(f"Wrote dim_metric: {dim_metric_out} (rows={len(dim_metric)})")
    print(f"Wrote dim_source_field_map: {dim_map_out} (rows={len(dim_map)})")
    print(f"Wrote coverage: {plan['outputs']['coverage_report_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

