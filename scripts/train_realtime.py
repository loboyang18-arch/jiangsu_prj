"""
实时价格预测 — 独立实现

任务定义:
  预测 D 日 96 个 15 分钟时段的实时出清价格。

四种预测模式:
  Mode A0: D-1 09:30 决策，D 日日前价不可用，D-1 实时价仅 hh 1~34 可用
  Mode A1: D-1 17:00 决策，D-1 实时价仅 hh 1~64 可用
  Mode A2: D-1 24:00 决策，D-1 实时价全部 96 时段可用
  Mode B:  D 日逐小时滚动预测，利用已出清 D 日实时价

可用信息:
  - D 日日前出清价 (D-1 ~17:00 发布; Mode A0 不可用)
  - D 日 boundary 曲线 (D-1 09:00 发布)
  - D-1/D-2 历史日前+实时价 (受 cutoff_hh 截断保护)
  - 日历特征: dow, month, is_weekend, hh_index
  - (Mode B) D 日已出清实时价 hh 1~t

建模策略:
  - 每个样本 = (trade_date, hh_index) 对
  - 统一模型: hh_index 作为特征，单模型覆盖 96 个时段
  - Expanding window CV 按 trade_date 切分
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(mean_squared_error(y_true, y_pred, squared=False)),
        mape=_mape(y_true, y_pred),
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOUNDARY_COLS = [
    "load_forecast_boundary_汇总",
    "receive_plan_boundary_汇总",
    "gas_plan_boundary_江北",
    "gas_plan_boundary_江南",
    "pv_forecast_boundary_江北",
    "pv_forecast_boundary_江南",
    "wind_forecast_boundary_江北",
    "wind_forecast_boundary_江南",
    "reserve_positive_汇总",
    "reserve_negative_汇总",
]

REALTIME_TARGETS = {
    "jn": "price_realtime_jn_final_江南",
    "jb": "price_realtime_jb_final_江北",
}

DAYAHEAD_PRICE_COLS = [
    "price_dayahead_jn_江南",
    "price_dayahead_jb_江北",
    "price_dayahead_jn_node_江南",
    "price_dayahead_jb_node_江北",
]

REALTIME_PRICE_COLS = [
    "price_realtime_jn_final_江南",
    "price_realtime_jb_final_江北",
    "price_realtime_jn_node_江南",
    "price_realtime_jb_node_江北",
]

ACTUAL_COLS = {
    "load_actual_total_汇总": "load_actual",
    "wind_actual_江北": "wind_actual_jb",
    "wind_actual_江南": "wind_actual_jn",
    "pv_actual_江北": "pv_actual_jb",
    "pv_actual_江南": "pv_actual_jn",
    "gas_actual_江北": "gas_actual_jb",
    "gas_actual_江南": "gas_actual_jn",
    "receive_actual_huadong_华东": "receive_actual",
}

FLOOR_PRICE = 50.0
FLOOR_PRED_VALUE = 30.0
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_wide(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["trade_date", "hh_index"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature building — Mode A (D-1 全天预测)
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    region: str,
    cutoff_hh: int = 96,
    include_d_dayahead: bool = True,
) -> pd.DataFrame:
    """
    构建实时价格预测特征表。

    cutoff_hh: D-1 实时价可用的最大 hh_index。
      - Mode A0 (D-1 09:30): cutoff_hh=34
      - Mode A1 (D-1 17:00): cutoff_hh=64
      - Mode A2 (D-1 24:00): cutoff_hh=96

    include_d_dayahead: 是否包含 D 日日前出清价特征。
      - Mode A0: False (D日日前价在09:30尚未发布)
      - Mode A1/A2/B: True
    """
    target_col = REALTIME_TARGETS[region]
    da_target = f"price_dayahead_{region}_江南" if region == "jn" else f"price_dayahead_{region}_江北"
    available_boundary = [c for c in BOUNDARY_COLS if c in df.columns]

    records = []
    dates = sorted(df["trade_date"].unique())

    for i, d in enumerate(dates):
        day_df = df[df["trade_date"] == d].sort_values("hh_index")
        if len(day_df) != 96:
            continue

        row = {}
        row["trade_date"] = day_df["trade_date"].values
        row["hh_index"] = day_df["hh_index"].values
        row["y"] = day_df[target_col].values

        # === D 日日前出清价特征 (核心锚点, Mode A0 时跳过) ===
        if include_d_dayahead and da_target in day_df.columns:
            da_vals = day_df[da_target].values
            row["da_same_hh"] = da_vals
            row["da_day_mean"] = np.nanmean(da_vals)
            row["da_day_std"] = np.nanstd(da_vals)
            row["da_day_min"] = np.nanmin(da_vals)
            row["da_day_max"] = np.nanmax(da_vals)
            row["da_rank"] = pd.Series(da_vals).rank(pct=True).values
            row["da_diff"] = np.diff(da_vals, prepend=da_vals[0])
            row["da_floor_rate"] = (da_vals <= FLOOR_PRICE).mean()
            row["da_spike_rate"] = (da_vals >= 400).mean()
            row["da_day_range"] = np.nanmax(da_vals) - np.nanmin(da_vals)
            row["da_day_median"] = np.nanmedian(da_vals)

            row["da_is_floor_hh"] = (da_vals <= FLOOR_PRICE).astype(int)
            row["da_is_spike_hh"] = (da_vals >= 400).astype(int)
            row["da_dev_from_mean_hh"] = da_vals - np.nanmean(da_vals)

            for da_node in DAYAHEAD_PRICE_COLS:
                if da_node == da_target or da_node not in day_df.columns:
                    continue
                short = da_node.replace("price_dayahead_", "da_").replace("_江南", "_jn").replace("_江北", "_jb").replace("_node", "_nd")
                node_vals = day_df[da_node].values
                row[f"{short}_same_hh"] = node_vals
                row[f"{short}_day_mean"] = np.nanmean(node_vals)

        # === Mode A0: 用 D-1 日前价作为替代锚点 ===
        if not include_d_dayahead and i >= 1:
            prev_date = dates[i - 1]
            prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
            if len(prev_df) == 96 and da_target in prev_df.columns:
                da_vals_dm1 = prev_df[da_target].values
                hh_vals = row["hh_index"]
                row["da_dm1_anchor_same_hh"] = da_vals_dm1[hh_vals - 1]
                row["da_dm1_anchor_day_mean"] = np.nanmean(da_vals_dm1)
                row["da_dm1_anchor_day_std"] = np.nanstd(da_vals_dm1)
                row["da_dm1_anchor_day_min"] = np.nanmin(da_vals_dm1)
                row["da_dm1_anchor_day_max"] = np.nanmax(da_vals_dm1)
                row["da_dm1_anchor_rank"] = pd.Series(da_vals_dm1[hh_vals - 1]).rank(pct=True).values if len(da_vals_dm1) > 0 else np.nan
                row["da_dm1_anchor_diff"] = np.diff(da_vals_dm1, prepend=da_vals_dm1[0])[hh_vals - 1]
                row["da_dm1_anchor_floor_rate"] = (da_vals_dm1 <= FLOOR_PRICE).mean()
                row["da_dm1_anchor_spike_rate"] = (da_vals_dm1 >= 400).mean()
                row["da_dm1_anchor_range"] = np.nanmax(da_vals_dm1) - np.nanmin(da_vals_dm1)
                row["da_dm1_anchor_median"] = np.nanmedian(da_vals_dm1)
                row["da_dm1_anchor_is_floor_hh"] = (da_vals_dm1[hh_vals - 1] <= FLOOR_PRICE).astype(int)
                row["da_dm1_anchor_is_spike_hh"] = (da_vals_dm1[hh_vals - 1] >= 400).astype(int)
                row["da_dm1_anchor_dev_from_mean_hh"] = da_vals_dm1[hh_vals - 1] - np.nanmean(da_vals_dm1)
                for da_node in DAYAHEAD_PRICE_COLS:
                    if da_node == da_target or da_node not in prev_df.columns:
                        continue
                    short = da_node.replace("price_dayahead_", "da_dm1_").replace("_江南", "_jn").replace("_江北", "_jb").replace("_node", "_nd")
                    node_vals = prev_df[da_node].values
                    row[f"{short}_same_hh"] = node_vals[hh_vals - 1]
                    row[f"{short}_day_mean"] = np.nanmean(node_vals)

        # === D 日 boundary 曲线 ===
        for col in available_boundary:
            short = col.replace("_汇总", "").replace("_江北", "_jb").replace("_江南", "_jn")
            vals = day_df[col].values
            row[f"bnd_{short}"] = vals
            row[f"bnd_{short}_diff"] = np.diff(vals, prepend=vals[0])
            row[f"bnd_{short}_day_mean"] = np.nanmean(vals)
            row[f"bnd_{short}_day_std"] = np.nanstd(vals)
            row[f"bnd_{short}_rank"] = pd.Series(vals).rank(pct=True).values

        # 净负荷与新能源渗透率
        load_col = "load_forecast_boundary_汇总"
        wind_jb, wind_jn = "wind_forecast_boundary_江北", "wind_forecast_boundary_江南"
        pv_jb, pv_jn = "pv_forecast_boundary_江北", "pv_forecast_boundary_江南"
        if all(c in day_df.columns for c in [load_col, wind_jb, wind_jn, pv_jb, pv_jn]):
            load_v = day_df[load_col].values
            re_total = (day_df[wind_jb].values + day_df[wind_jn].values
                        + day_df[pv_jb].values + day_df[pv_jn].values)
            net_load = load_v - re_total
            row["net_load"] = net_load
            row["net_load_day_mean"] = np.nanmean(net_load)
            row["net_load_day_std"] = np.nanstd(net_load)
            row["net_load_rank"] = pd.Series(net_load).rank(pct=True).values
            row["re_penetration"] = np.where(load_v > 0, re_total / load_v, 0.0)
            row["re_total"] = re_total
            row["re_total_day_mean"] = np.nanmean(re_total)


        # === D-1 / D-2 日前价特征 ===
        for lag_d, lag_label in [(1, "dm1"), (2, "dm2"), (7, "dm7")]:
            if i >= lag_d:
                prev_date = dates[i - lag_d]
                prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
                if len(prev_df) == 96:
                    for pcol in DAYAHEAD_PRICE_COLS:
                        if pcol not in prev_df.columns:
                            continue
                        short = pcol.replace("price_dayahead_", "p_").replace("_江南", "_jn").replace("_江北", "_jb").replace("_node", "_nd")
                        vals = prev_df[pcol].values
                        hh_vals = row["hh_index"]
                        row[f"{short}_{lag_label}_mean"] = np.nanmean(vals)
                        row[f"{short}_{lag_label}_std"] = np.nanstd(vals)
                        row[f"{short}_{lag_label}_min"] = np.nanmin(vals)
                        row[f"{short}_{lag_label}_max"] = np.nanmax(vals)
                        row[f"{short}_{lag_label}_same_hh"] = vals[hh_vals - 1]
                        row[f"{short}_{lag_label}_same_hh_diff"] = vals[hh_vals - 1] - np.nanmean(vals)

        # === D-1 / D-2 实时价特征 (受 cutoff_hh 截断保护) ===
        for lag_d, lag_label in [(1, "dm1"), (2, "dm2")]:
            if i >= lag_d:
                prev_date = dates[i - lag_d]
                prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
                if len(prev_df) == 96:
                    # D-1 受截断, D-2 及更早用全量
                    eff_cutoff = cutoff_hh if lag_d == 1 else 96
                    for pcol in REALTIME_PRICE_COLS:
                        if pcol not in prev_df.columns:
                            continue
                        short = pcol.replace("price_realtime_", "rt_").replace("_final", "").replace("_江南", "_jn").replace("_江北", "_jb").replace("_node", "_nd")
                        all_vals = prev_df[pcol].values
                        vals = all_vals[:eff_cutoff]
                        hh_vals = row["hh_index"]

                        row[f"{short}_{lag_label}_mean"] = np.nanmean(vals)
                        row[f"{short}_{lag_label}_std"] = np.nanstd(vals)
                        row[f"{short}_{lag_label}_min"] = np.nanmin(vals)
                        row[f"{short}_{lag_label}_max"] = np.nanmax(vals)
                        row[f"{short}_{lag_label}_floor_rate"] = (vals <= FLOOR_PRICE).mean()
                        row[f"{short}_{lag_label}_spike_rate"] = (vals >= 400).mean()
                        # same_hh: 仅当该 hh 在截断范围内时可用
                        safe_hh = np.where(hh_vals <= eff_cutoff, hh_vals - 1, 0)
                        same_hh_vals = all_vals[safe_hh]
                        same_hh_vals = np.where(hh_vals <= eff_cutoff, same_hh_vals, np.nan)
                        row[f"{short}_{lag_label}_same_hh"] = same_hh_vals

        # === D-1 日前-实时价差 (受 cutoff_hh 截断) ===
        if i >= 1:
            prev_date = dates[i - 1]
            prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
            if len(prev_df) == 96:
                for da_col, rt_col, label in [
                    ("price_dayahead_jn_江南", "price_realtime_jn_final_江南", "jn"),
                    ("price_dayahead_jb_江北", "price_realtime_jb_final_江北", "jb"),
                ]:
                    if da_col in prev_df.columns and rt_col in prev_df.columns:
                        da_v = prev_df[da_col].values[:cutoff_hh]
                        rt_v = prev_df[rt_col].values[:cutoff_hh]
                        spread = da_v - rt_v
                        row[f"dm1_spread_{label}_mean"] = np.mean(spread)
                        row[f"dm1_spread_{label}_std"] = np.std(spread)
                        row[f"dm1_spread_{label}_abs_mean"] = np.mean(np.abs(spread))
                        # same_hh spread
                        hh_vals = row["hh_index"]
                        all_spread = prev_df[da_col].values - prev_df[rt_col].values
                        safe_hh = np.where(hh_vals <= cutoff_hh, hh_vals - 1, 0)
                        sh_spread = all_spread[safe_hh]
                        sh_spread = np.where(hh_vals <= cutoff_hh, sh_spread, np.nan)
                        row[f"dm1_spread_{label}_same_hh"] = sh_spread

        # === D-1 actual 外生变量特征 (受 cutoff_hh 截断) ===
        if i >= 1:
            prev_date = dates[i - 1]
            prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
            if len(prev_df) == 96:
                for orig_col, short in ACTUAL_COLS.items():
                    if orig_col not in prev_df.columns:
                        continue
                    vals = prev_df[orig_col].values[:cutoff_hh]
                    row[f"{short}_dm1_mean"] = np.nanmean(vals)
                    row[f"{short}_dm1_std"] = np.nanstd(vals)
                    hh_vals = row["hh_index"]
                    all_vals = prev_df[orig_col].values
                    safe_hh = np.where(hh_vals <= cutoff_hh, hh_vals - 1, 0)
                    same_hh_vals = all_vals[safe_hh]
                    row[f"{short}_dm1_same_hh"] = np.where(hh_vals <= cutoff_hh, same_hh_vals, np.nan)

                re_actual = (prev_df.get("wind_actual_江北", pd.Series(0)).values[:cutoff_hh]
                             + prev_df.get("wind_actual_江南", pd.Series(0)).values[:cutoff_hh]
                             + prev_df.get("pv_actual_江北", pd.Series(0)).values[:cutoff_hh]
                             + prev_df.get("pv_actual_江南", pd.Series(0)).values[:cutoff_hh])
                load_actual = prev_df.get("load_actual_total_汇总", pd.Series(0)).values[:cutoff_hh]
                net_load_actual = load_actual - re_actual
                row["net_load_actual_dm1_mean"] = np.nanmean(net_load_actual)
                row["net_load_actual_dm1_min"] = np.nanmin(net_load_actual)
                hh_vals = row["hh_index"]
                all_nl = prev_df.get("load_actual_total_汇总", pd.Series(0)).values - (
                    prev_df.get("wind_actual_江北", pd.Series(0)).values
                    + prev_df.get("wind_actual_江南", pd.Series(0)).values
                    + prev_df.get("pv_actual_江北", pd.Series(0)).values
                    + prev_df.get("pv_actual_江南", pd.Series(0)).values
                )
                safe_hh = np.where(hh_vals <= cutoff_hh, hh_vals - 1, 0)
                row["net_load_actual_dm1_same_hh"] = np.where(
                    hh_vals <= cutoff_hh, all_nl[safe_hh], np.nan
                )
                row["re_penetration_actual_dm1"] = np.where(
                    load_actual > 0, re_actual / load_actual, 0.0
                ).mean()

                load_bnd = prev_df.get("load_forecast_boundary_汇总", pd.Series(0)).values[:cutoff_hh]
                if "load_forecast_boundary_汇总" in prev_df.columns:
                    row["load_fcst_err_dm1_mean"] = np.mean(load_bnd - load_actual)
                re_bnd = (prev_df.get("wind_forecast_boundary_江北", pd.Series(0)).values[:cutoff_hh]
                          + prev_df.get("wind_forecast_boundary_江南", pd.Series(0)).values[:cutoff_hh]
                          + prev_df.get("pv_forecast_boundary_江北", pd.Series(0)).values[:cutoff_hh]
                          + prev_df.get("pv_forecast_boundary_江南", pd.Series(0)).values[:cutoff_hh])
                row["re_fcst_err_dm1_mean"] = np.mean(re_bnd - re_actual)
                row["re_fcst_err_dm1_std"] = np.std(re_bnd - re_actual)

        # === D-1 实时价目标列地板价统计 ===
        if i >= 1:
            prev_date = dates[i - 1]
            prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
            if len(prev_df) == 96 and target_col in prev_df.columns:
                prev_rt = prev_df[target_col].values[:cutoff_hh]
                row["dm1_rt_floor_rate"] = (prev_rt <= FLOOR_PRICE).mean()
                row["dm1_rt_floor_count"] = (prev_rt <= FLOOR_PRICE).sum()
                row["dm1_rt_spike_rate"] = (prev_rt >= 400).mean()

        # === D-1 日前目标列地板价统计 ===
        if i >= 1:
            prev_date = dates[i - 1]
            prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
            da_prev_col = f"price_dayahead_{region}_江南" if region == "jn" else f"price_dayahead_{region}_江北"
            if len(prev_df) == 96 and da_prev_col in prev_df.columns:
                prev_da = prev_df[da_prev_col].values
                row["dm1_da_floor_rate"] = (prev_da <= FLOOR_PRICE).mean()
                row["dm1_da_spike_rate"] = (prev_da >= 400).mean()

        # === 近 3 日价格趋势 ===
        if i >= 3:
            for pcol in DAYAHEAD_PRICE_COLS:
                if pcol not in df.columns:
                    continue
                short = pcol.replace("price_dayahead_", "p_").replace("_江南", "_jn").replace("_江北", "_jb").replace("_node", "_nd")
                recent_dates = dates[max(0, i - 3):i]
                recent_df = df[df["trade_date"].isin(recent_dates)]
                if len(recent_df) > 0:
                    daily_means = recent_df.groupby("trade_date")[pcol].mean()
                    row[f"{short}_3d_avg"] = daily_means.mean()
                    row[f"{short}_3d_trend"] = daily_means.iloc[-1] - daily_means.iloc[0] if len(daily_means) > 1 else 0.0

            # 实时价近3日趋势 (D-2及更早全量可用, D-1受截断)
            for pcol in [target_col]:
                if pcol not in df.columns:
                    continue
                recent_dates = dates[max(0, i - 3):i]
                daily_rt_means = []
                for rd in recent_dates:
                    rdf = df[df["trade_date"] == rd]
                    if len(rdf) == 96 and pcol in rdf.columns:
                        eff_c = cutoff_hh if rd == dates[i - 1] else 96
                        daily_rt_means.append(rdf[pcol].values[:eff_c].mean())
                if len(daily_rt_means) >= 2:
                    row["rt_target_3d_avg"] = np.mean(daily_rt_means)
                    row["rt_target_3d_trend"] = daily_rt_means[-1] - daily_rt_means[0]

        # === 近 7 天价格趋势和波动率 ===
        if i >= 7:
            for pcol, label in [(da_target, f"da_{region}"), (target_col, f"rt_{region}")]:
                if pcol not in df.columns:
                    continue
                week_means = []
                for j in range(7):
                    dj = dates[i - 7 + j]
                    djdf = df[df["trade_date"] == dj]
                    if len(djdf) == 96 and pcol in djdf.columns:
                        # 对实时价: D-1用截断, 更早用全量
                        if pcol in REALTIME_PRICE_COLS and dj == dates[i - 1]:
                            week_means.append(djdf[pcol].values[:cutoff_hh].mean())
                        else:
                            week_means.append(djdf[pcol].mean())
                if len(week_means) == 7:
                    x = np.arange(7, dtype=float)
                    slope = np.polyfit(x, week_means, 1)[0]
                    row[f"trend_7d_{label}"] = slope
                    row[f"vol_7d_{label}"] = np.std(week_means)

        # === 近 7 天同时段地板价频率 ===
        if i >= 7:
            # 实时价
            for rt_col, label in [(target_col, "rt_target"), ("price_realtime_jn_final_江南", "rt_jn"), ("price_realtime_jb_final_江北", "rt_jb")]:
                if rt_col not in df.columns:
                    continue
                hist_dates = dates[max(0, i - 7):i]
                hist_vals_by_hh = {hh: [] for hh in range(1, 97)}
                for hd in hist_dates:
                    hdf = df[df["trade_date"] == hd].sort_values("hh_index")
                    if len(hdf) == 96 and rt_col in hdf.columns:
                        eff_c = cutoff_hh if hd == dates[i - 1] else 96
                        for hh_idx in range(1, eff_c + 1):
                            hist_vals_by_hh[hh_idx].append(hdf[rt_col].values[hh_idx - 1])
                hh_floor_rates = {}
                for hh_idx in range(1, 97):
                    vv = hist_vals_by_hh[hh_idx]
                    hh_floor_rates[hh_idx] = np.mean([v <= FLOOR_PRICE for v in vv]) if vv else np.nan
                hh_idx_arr = row["hh_index"]
                row[f"hh_{label}_floor_rate_7d"] = np.array([hh_floor_rates.get(h, np.nan) for h in hh_idx_arr])

            # 日前价
            if da_target in df.columns:
                hist_dates = dates[max(0, i - 7):i]
                hist_df = df[df["trade_date"].isin(hist_dates)]
                if len(hist_df) > 0:
                    hh_da_floor = hist_df.groupby("hh_index")[da_target].apply(lambda x: (x <= FLOOR_PRICE).mean())
                    hh_idx_arr = row["hh_index"]
                    row["hh_da_floor_rate_7d"] = pd.Series(hh_idx_arr).map(hh_da_floor).values

        # === 日历特征 ===
        dt = pd.Timestamp(d)
        row["dow"] = dt.dayofweek
        row["month"] = dt.month
        row["is_weekend"] = int(dt.dayofweek >= 5)

        records.append(pd.DataFrame(row))

    feat = pd.concat(records, ignore_index=True)
    feat["trade_date"] = pd.to_datetime(feat["trade_date"])
    return feat


# ---------------------------------------------------------------------------
# Feature building — Mode B (滚动增量特征)
# ---------------------------------------------------------------------------

def augment_rolling_features(
    feat: pd.DataFrame,
    df: pd.DataFrame,
    region: str,
    current_hh: int,
) -> pd.DataFrame:
    """
    为 Mode B 添加 D 日已出清实时价的滚动特征。

    current_hh: 当前预测时刻，表示 hh 1~current_hh 的实时价已出清。
    只预测 hh > current_hh 的时段。
    """
    target_col = REALTIME_TARGETS[region]
    da_target = f"price_dayahead_{region}_江南" if region == "jn" else f"price_dayahead_{region}_江北"
    feat = feat.copy()

    # 只保留待预测时段
    feat = feat[feat["hh_index"] > current_hh].copy()

    if current_hh <= 0 or len(feat) == 0:
        feat["hours_ahead"] = np.nan
        feat["rt_d_available"] = 0
        return feat

    feat["hours_ahead"] = (feat["hh_index"].values - current_hh) * 0.25

    dates = sorted(feat["trade_date"].unique())
    for d in dates:
        day_df = df[df["trade_date"] == d].sort_values("hh_index")
        if len(day_df) != 96 or target_col not in day_df.columns:
            continue

        rt_cleared = day_df[target_col].values[:current_hh]
        mask = feat["trade_date"] == d

        feat.loc[mask, "rt_d_latest_hh"] = rt_cleared[-1]
        recent_n = min(4 * 4, current_hh)  # 最近4个小时 = 16个hh
        recent_vals = rt_cleared[-recent_n:]
        feat.loc[mask, "rt_d_recent_mean"] = np.nanmean(recent_vals)
        feat.loc[mask, "rt_d_recent_std"] = np.nanstd(recent_vals)
        feat.loc[mask, "rt_d_so_far_mean"] = np.nanmean(rt_cleared)
        feat.loc[mask, "rt_d_so_far_min"] = np.nanmin(rt_cleared)
        feat.loc[mask, "rt_d_so_far_max"] = np.nanmax(rt_cleared)
        feat.loc[mask, "rt_d_so_far_floor_rate"] = (rt_cleared <= FLOOR_PRICE).mean()
        feat.loc[mask, "rt_d_so_far_spike_rate"] = (rt_cleared >= 400).mean()
        feat.loc[mask, "rt_d_available"] = current_hh

        if da_target in day_df.columns:
            da_cleared = day_df[da_target].values[:current_hh]
            spread_recent = da_cleared[-recent_n:] - rt_cleared[-recent_n:]
            feat.loc[mask, "rt_d_da_spread_recent_mean"] = np.nanmean(spread_recent)
            feat.loc[mask, "rt_d_da_spread_recent_std"] = np.nanstd(spread_recent)
            feat.loc[mask, "rt_d_da_spread_latest"] = da_cleared[-1] - rt_cleared[-1]

        if current_hh >= 4:
            tail4 = rt_cleared[-4:]
            x4 = np.arange(len(tail4), dtype=float)
            slope = np.polyfit(x4, tail4, 1)[0] if len(tail4) > 1 else 0.0
            feat.loc[mask, "rt_d_trend_slope_1h"] = slope
            feat.loc[mask, "rt_d_momentum_1h"] = tail4[-1] - tail4[0]

        if current_hh >= 8:
            tail8 = rt_cleared[-8:]
            x8 = np.arange(len(tail8), dtype=float)
            feat.loc[mask, "rt_d_trend_slope_2h"] = np.polyfit(x8, tail8, 1)[0]

        n_transitions = 0
        if len(rt_cleared) >= 2:
            is_floor = rt_cleared <= FLOOR_PRICE
            n_transitions = int(np.sum(np.abs(np.diff(is_floor.astype(int)))))
        feat.loc[mask, "rt_d_floor_transitions"] = n_transitions

        feat.loc[mask, "rt_d_range"] = np.nanmax(rt_cleared) - np.nanmin(rt_cleared)
        feat.loc[mask, "rt_d_cv"] = (
            np.nanstd(rt_cleared) / max(np.nanmean(rt_cleared), 1e-6)
        )

    return feat


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _fit_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict,
    sample_weight: Optional[np.ndarray] = None,
) -> LGBMRegressor:
    model = LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[early_stopping(stopping_rounds=80), log_evaluation(period=100)],
    )
    return model


def _fit_quantile_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict,
    alpha: float,
    sample_weight: Optional[np.ndarray] = None,
) -> LGBMRegressor:
    q_params = {k: v for k, v in params.items() if k != "objective"}
    q_params["objective"] = "quantile"
    q_params["alpha"] = alpha
    model = LGBMRegressor(**q_params)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=[early_stopping(stopping_rounds=80), log_evaluation(period=200)],
    )
    return model


def _quantile_combine(
    q_preds: Dict[float, np.ndarray],
    naive_pred: Optional[np.ndarray],
    floor_price: float = 50.0,
    floor_q10_threshold: float = 80.0,
    floor_blend_weight: float = 0.6,
    uncertainty_threshold: float = 200.0,
    naive_weight: float = 0.3,
    da_anchor: Optional[np.ndarray] = None,
) -> np.ndarray:
    q10 = q_preds[0.10]
    q25 = q_preds[0.25]
    q50 = q_preds[0.50]
    q90 = q_preds[0.90]

    pred = q50.copy()

    if da_anchor is not None:
        floor_mask = (da_anchor + q10) <= floor_q10_threshold
    else:
        floor_mask = q10 <= floor_q10_threshold
    pred[floor_mask] = (
        floor_blend_weight * q10[floor_mask]
        + (1 - floor_blend_weight) * q25[floor_mask]
    )

    if naive_pred is not None:
        spread = q90 - q10
        uncertain_mask = spread > uncertainty_threshold
        valid = uncertain_mask & ~floor_mask & ~np.isnan(naive_pred)
        pred[valid] = (1 - naive_weight) * q50[valid] + naive_weight * naive_pred[valid]

    return pred


def _tune_quantile_params(
    q_preds: Dict[float, np.ndarray],
    naive_pred: Optional[np.ndarray],
    y_true: np.ndarray,
    da_anchor: Optional[np.ndarray] = None,
) -> Dict:
    best_mae = float("inf")
    best_params: Dict = {
        "floor_q10_threshold": 80.0,
        "floor_blend_weight": 0.6,
        "uncertainty_threshold": 200.0,
        "naive_weight": 0.0,
    }

    for fq10 in [50.0, 80.0, 100.0, 130.0, 160.0]:
        for fbw in [0.4, 0.6, 0.8, 1.0]:
            for unc_thr in [150, 200, 250, 300, 400]:
                for nw in [0.0, 0.2, 0.4]:
                    pred = _quantile_combine(
                        q_preds, naive_pred,
                        floor_q10_threshold=fq10,
                        floor_blend_weight=fbw,
                        uncertainty_threshold=float(unc_thr),
                        naive_weight=nw,
                        da_anchor=da_anchor,
                    )
                    mae = float(mean_absolute_error(y_true, pred))
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {
                            "floor_q10_threshold": fq10,
                            "floor_blend_weight": fbw,
                            "uncertainty_threshold": float(unc_thr),
                            "naive_weight": nw,
                        }

    return best_params


# ---------------------------------------------------------------------------
# Two-stage (floor classifier)
# ---------------------------------------------------------------------------

def _fit_floor_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict,
    sample_weight: Optional[np.ndarray] = None,
) -> LGBMClassifier:
    y_cls_train = (y_train <= FLOOR_PRICE).astype(int)
    y_cls_val = (y_val <= FLOOR_PRICE).astype(int)

    pos_rate = y_cls_train.mean()
    scale = max(1.0, (1 - pos_rate) / max(pos_rate, 0.01))

    cls_params = {
        "n_estimators": 600,
        "learning_rate": params.get("learning_rate", 0.03),
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "scale_pos_weight": scale,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = LGBMClassifier(**cls_params)
    model.fit(
        X_train, y_cls_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_cls_val)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=200)],
    )
    return model


def _find_best_threshold(
    floor_prob: np.ndarray,
    reg_pred: np.ndarray,
    y_true: np.ndarray,
    floor_pred_value: float = FLOOR_PRED_VALUE,
) -> Tuple[float, float]:
    reg_mae = float(mean_absolute_error(y_true, reg_pred))
    best_t, best_mae = 1.1, reg_mae

    for t in np.arange(0.40, 0.85, 0.05):
        pred_floor = floor_prob >= t
        if pred_floor.sum() == 0:
            continue
        actual_floor = y_true <= FLOOR_PRICE
        tp = (pred_floor & actual_floor).sum()
        fp = (pred_floor & ~actual_floor).sum()
        precision = tp / max(tp + fp, 1)
        if precision < 0.60:
            continue
        combined = np.where(pred_floor, floor_pred_value, reg_pred)
        mae = float(mean_absolute_error(y_true, combined))
        if mae < best_mae:
            best_mae = mae
            best_t = float(t)

    return best_t, best_mae


def _two_stage_predict(
    clf: LGBMClassifier,
    reg: LGBMRegressor,
    X: pd.DataFrame,
    threshold: float,
    floor_pred_value: float = FLOOR_PRED_VALUE,
) -> np.ndarray:
    floor_prob = clf.predict_proba(X)[:, 1]
    reg_pred = reg.predict(X)
    return np.where(floor_prob >= threshold, floor_pred_value, reg_pred)


# ---------------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------------

def _compute_time_decay_weights(
    trade_dates: pd.Series,
    half_life_days: int = 30,
) -> np.ndarray:
    date_nums = pd.to_datetime(trade_dates).astype(np.int64) // 10**9
    max_t = date_nums.max()
    diff_days = (date_nums - max_t) / 86400.0
    weights = np.power(2.0, diff_days / half_life_days)
    return weights


def _tune_adaptive_naive_blend(
    pred: np.ndarray,
    naive: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[float, float]:
    valid_mask = ~np.isnan(naive)
    if valid_mask.sum() == 0:
        return 0.0, float(mean_absolute_error(y_true, pred))

    best_alpha = 0.0
    best_mae = float(mean_absolute_error(y_true, pred))

    for alpha in np.arange(0.00, 0.55, 0.05):
        blended = pred.copy()
        blended[valid_mask] = (1 - alpha) * pred[valid_mask] + alpha * naive[valid_mask]
        mae = float(mean_absolute_error(y_true, blended))
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)

    return best_alpha, best_mae


def _apply_adaptive_naive_blend(
    pred: np.ndarray,
    naive: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if alpha <= 0.0:
        return pred
    result = pred.copy()
    valid = ~np.isnan(naive)
    result[valid] = (1 - alpha) * pred[valid] + alpha * naive[valid]
    return result


def _compute_dynamic_floor_value(y_train: np.ndarray) -> float:
    floor_vals = y_train[y_train <= FLOOR_PRICE]
    if len(floor_vals) > 0:
        return float(np.median(floor_vals))
    return FLOOR_PRED_VALUE


def _compute_residual_bias(
    pred_val: np.ndarray,
    y_val: np.ndarray,
    hh_val: np.ndarray,
) -> Dict[int, float]:
    residuals = y_val - pred_val
    bias: Dict[int, float] = {}
    for hh in np.unique(hh_val):
        mask = hh_val == hh
        if mask.sum() > 0:
            bias[int(hh)] = float(np.mean(residuals[mask]))
    return bias


def _tune_residual_gamma(
    pred_val: np.ndarray,
    y_val: np.ndarray,
    hh_val: np.ndarray,
) -> Tuple[float, Dict[int, float]]:
    bias_full = _compute_residual_bias(pred_val, y_val, hh_val)
    best_gamma = 0.0
    best_mae = float(mean_absolute_error(y_val, pred_val))

    for gamma in np.arange(0.0, 1.05, 0.1):
        corrected = pred_val.copy()
        for hh, b in bias_full.items():
            mask = hh_val == hh
            corrected[mask] += gamma * b
        mae = float(mean_absolute_error(y_val, corrected))
        if mae < best_mae:
            best_mae = mae
            best_gamma = float(gamma)

    return best_gamma, bias_full


def _apply_residual_correction(
    pred: np.ndarray,
    hh_arr: np.ndarray,
    bias: Dict[int, float],
    gamma: float,
) -> np.ndarray:
    if gamma <= 0.0:
        return pred
    result = pred.copy()
    for hh, b in bias.items():
        mask = hh_arr == hh
        result[mask] += gamma * b
    return result


# ---------------------------------------------------------------------------
# Expanding window CV — Mode A
# ---------------------------------------------------------------------------

def expanding_window_cv(
    feat: pd.DataFrame,
    df_raw: pd.DataFrame,
    region: str,
    min_train_days: int = 60,
    val_days: int = 7,
    test_days: int = 7,
    step_days: int = 7,
    params: Optional[Dict] = None,
    feature_select_top_k: int = 0,
    two_stage: bool = False,
    quantile_mode: bool = False,
    quantile_two_stage: bool = False,
    time_decay_half_life: int = 0,
    adaptive_naive_blend: bool = False,
    dynamic_floor_value: bool = False,
    residual_correction: bool = False,
    naive_use_dm1: bool = False,
    sharpening: float = 1.0,
    spread_mode: bool = False,
) -> Tuple[List[Dict], Optional[Dict], Optional[pd.DataFrame]]:
    """Expanding window CV for Mode A (full-day prediction).

    naive_use_dm1: if True, naive baseline = D-1 dayahead price (for Mode A0).
    spread_mode: if True, target is DA-RT spread; naive baseline = 0.
    """
    if params is None:
        params = {}
    default_params = {
        "n_estimators": 1500,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    default_params.update(params)

    da_target = f"price_dayahead_{region}_江南" if region == "jn" else f"price_dayahead_{region}_江北"

    dates = sorted(feat["trade_date"].unique())
    n_dates = len(dates)

    _meta_cols = {"trade_date", "y", "y_abs", "da_anchor"}
    x_cols = [c for c in feat.columns if c not in _meta_cols]

    if feature_select_top_k > 0 and feature_select_top_k < len(x_cols):
        pre_train_dates = dates[:min_train_days]
        pre_val_dates = dates[min_train_days: min_train_days + val_days]
        pre_train_mask = feat["trade_date"].isin(pre_train_dates)
        pre_val_mask = feat["trade_date"].isin(pre_val_dates)
        X_pre = feat.loc[pre_train_mask, x_cols]
        y_pre = feat.loc[pre_train_mask, "y"].values
        X_pre_val = feat.loc[pre_val_mask, x_cols]
        y_pre_val = feat.loc[pre_val_mask, "y"].values
        pre_model = _fit_lgbm(X_pre, y_pre, X_pre_val, y_pre_val,
                              {**default_params, "n_estimators": 300})
        imp = pre_model.feature_importances_
        sorted_idx = np.argsort(imp)[::-1]
        x_cols = [x_cols[i] for i in sorted_idx[:feature_select_top_k] if imp[i] > 0]
        print(f"  [FEATURE SELECT] Kept {len(x_cols)}/{len(feat.columns) - 2} features (top-{feature_select_top_k})")

    folds = []
    last_payload = None
    last_test_df: Optional[pd.DataFrame] = None

    fold_idx = 0
    start = min_train_days

    while start + val_days + test_days <= n_dates:
        train_dates = dates[:start]
        val_dates_list = dates[start: start + val_days]
        test_dates = dates[start + val_days: start + val_days + test_days]

        train_mask = feat["trade_date"].isin(train_dates)
        val_mask = feat["trade_date"].isin(val_dates_list)
        test_mask = feat["trade_date"].isin(test_dates)

        X_train = feat.loc[train_mask, x_cols]
        y_train = feat.loc[train_mask, "y"].values
        X_val = feat.loc[val_mask, x_cols]
        y_val = feat.loc[val_mask, "y"].values
        X_test = feat.loc[test_mask, x_cols]
        y_test = feat.loc[test_mask, "y"].values

        if len(X_train) < 96 or len(X_test) < 96:
            start += step_days
            continue

        sw = None
        if time_decay_half_life > 0:
            sw = _compute_time_decay_weights(
                feat.loc[train_mask, "trade_date"], time_decay_half_life
            )

        fold_floor_value = (
            _compute_dynamic_floor_value(y_train)
            if dynamic_floor_value
            else FLOOR_PRED_VALUE
        )

        def _get_naive_dayahead(target_dates):
            """Naive baseline: spread模式=0, 否则=D日日前价(A1/A2)或D-1日前价(A0)."""
            all_dates_sorted = sorted(df_raw["trade_date"].unique())
            date_to_idx = {d: i for i, d in enumerate(all_dates_sorted)}
            naive_vals = []
            for td in target_dates:
                cur = feat.loc[feat["trade_date"] == td, ["hh_index", "y"]].copy()
                if spread_mode:
                    cur["naive_da"] = 0.0
                elif naive_use_dm1:
                    idx = date_to_idx.get(td)
                    if idx is not None and idx >= 1:
                        dm1_date = all_dates_sorted[idx - 1]
                        dm1_df = df_raw[df_raw["trade_date"] == dm1_date].sort_values("hh_index")
                        if len(dm1_df) == 96 and da_target in dm1_df.columns:
                            da_map = dict(zip(dm1_df["hh_index"].values, dm1_df[da_target].values))
                            cur["naive_da"] = cur["hh_index"].map(da_map)
                        else:
                            cur["naive_da"] = np.nan
                    else:
                        cur["naive_da"] = np.nan
                else:
                    td_df = df_raw[df_raw["trade_date"] == td].sort_values("hh_index")
                    if len(td_df) == 96 and da_target in td_df.columns:
                        da_map = dict(zip(td_df["hh_index"].values, td_df[da_target].values))
                        cur["naive_da"] = cur["hh_index"].map(da_map)
                    else:
                        cur["naive_da"] = np.nan
                naive_vals.append(cur)
            if naive_vals:
                return pd.concat(naive_vals)
            return pd.DataFrame(columns=["hh_index", "y", "naive_da"])

        naive_test_df = _get_naive_dayahead(test_dates)
        naive_test_arr = naive_test_df["naive_da"].values

        da_anchor_val = feat.loc[val_mask, "da_anchor"].values if spread_mode and "da_anchor" in feat.columns else None
        da_anchor_test = feat.loc[test_mask, "da_anchor"].values if spread_mode and "da_anchor" in feat.columns else None

        if quantile_mode:
            q_models = {}
            for alpha in QUANTILES:
                q_models[alpha] = _fit_quantile_lgbm(
                    X_train, y_train, X_val, y_val, default_params, alpha,
                    sample_weight=sw,
                )
            model = q_models[0.50]

            naive_val_df = _get_naive_dayahead(val_dates_list)
            naive_val_arr = naive_val_df["naive_da"].values
            q_val_preds = {a: m.predict(X_val) for a, m in q_models.items()}
            qc_params = _tune_quantile_params(q_val_preds, naive_val_arr, y_val,
                                              da_anchor=da_anchor_val)

            q_test_preds = {a: m.predict(X_test) for a, m in q_models.items()}
            pred_test = _quantile_combine(q_test_preds, naive_test_arr, **qc_params,
                                         da_anchor=da_anchor_test)

            if quantile_two_stage:
                clf = _fit_floor_classifier(
                    X_train, y_train, X_val, y_val, default_params, sample_weight=sw,
                )
                val_qc_pred = _quantile_combine(q_val_preds, naive_val_arr, **qc_params,
                                                da_anchor=da_anchor_val)
                val_floor_prob = clf.predict_proba(X_val)[:, 1]
                threshold, _ = _find_best_threshold(
                    val_floor_prob, val_qc_pred, y_val,
                    floor_pred_value=fold_floor_value,
                )
                if threshold < 1.0:
                    test_floor_prob = clf.predict_proba(X_test)[:, 1]
                    pred_test = np.where(
                        test_floor_prob >= threshold, fold_floor_value, pred_test
                    )
            else:
                clf = None
                threshold = 0.0

        elif two_stage:
            clf = _fit_floor_classifier(
                X_train, y_train, X_val, y_val, default_params, sample_weight=sw,
            )
            model = _fit_lgbm(X_train, y_train, X_val, y_val, default_params,
                              sample_weight=sw)
            val_floor_prob = clf.predict_proba(X_val)[:, 1]
            val_reg_pred = model.predict(X_val)
            threshold, _ = _find_best_threshold(
                val_floor_prob, val_reg_pred, y_val,
                floor_pred_value=fold_floor_value,
            )
            pred_test = _two_stage_predict(clf, model, X_test, threshold,
                                           floor_pred_value=fold_floor_value)
            q_models = None
            q_test_preds = None
            qc_params = {}

        else:
            model = _fit_lgbm(X_train, y_train, X_val, y_val, default_params,
                              sample_weight=sw)
            pred_test = model.predict(X_test)
            clf = None
            threshold = 0.0
            q_models = None
            q_test_preds = None
            qc_params = {}

        if sharpening != 1.0:
            gm = np.median(pred_test)
            pred_test = gm + sharpening * (pred_test - gm)
            if not spread_mode:
                pred_test = np.clip(pred_test, 0, 1500)

        # Val predictions for post-processing
        need_val_pred = adaptive_naive_blend or residual_correction
        pred_val = None
        if need_val_pred:
            naive_val_df_blend = _get_naive_dayahead(val_dates_list)
            naive_val_arr_blend = naive_val_df_blend["naive_da"].values
            if quantile_mode:
                pred_val = _quantile_combine(
                    {a: q_models[a].predict(X_val) for a in QUANTILES},
                    naive_val_arr_blend, **qc_params,
                    da_anchor=da_anchor_val,
                )
                if quantile_two_stage and clf is not None and threshold < 1.0:
                    vfp = clf.predict_proba(X_val)[:, 1]
                    pred_val = np.where(vfp >= threshold, fold_floor_value, pred_val)
            elif two_stage and clf is not None and threshold < 1.0:
                pred_val = _two_stage_predict(
                    clf, model, X_val, threshold,
                    floor_pred_value=fold_floor_value,
                )
            else:
                pred_val = model.predict(X_val)

        # Adaptive naive blending
        naive_blend_alpha = 0.0
        if adaptive_naive_blend and pred_val is not None:
            naive_blend_alpha, _ = _tune_adaptive_naive_blend(
                pred_val, naive_val_arr_blend, y_val
            )
            if naive_blend_alpha > 0:
                pred_val = _apply_adaptive_naive_blend(
                    pred_val, naive_val_arr_blend, naive_blend_alpha
                )
                pred_test = _apply_adaptive_naive_blend(
                    pred_test, naive_test_arr, naive_blend_alpha
                )

        # Residual correction
        residual_gamma = 0.0
        residual_bias: Dict[int, float] = {}
        if residual_correction and pred_val is not None:
            hh_val = feat.loc[val_mask, "hh_index"].values
            hh_test = feat.loc[test_mask, "hh_index"].values
            residual_gamma, residual_bias = _tune_residual_gamma(
                pred_val, y_val, hh_val
            )
            if residual_gamma > 0:
                pred_test = _apply_residual_correction(
                    pred_test, hh_test, residual_bias, residual_gamma
                )

        test_metrics = asdict(evaluate(y_test, pred_test))

        test_df = feat.loc[test_mask, ["trade_date", "hh_index", "y"]].copy()
        test_df["pred"] = pred_test
        if spread_mode and "y_abs" in feat.columns:
            test_df["y_abs"] = feat.loc[test_mask, "y_abs"].values
            test_df["da_anchor"] = feat.loc[test_mask, "da_anchor"].values
            test_df["pred_rt"] = pred_test + test_df["da_anchor"].values
        if quantile_mode and q_test_preds is not None:
            for alpha in QUANTILES:
                test_df[f"q{int(alpha*100):02d}"] = q_test_preds[alpha]

        naive_valid = naive_test_df.dropna(subset=["naive_da"])
        if len(naive_valid) > 0:
            naive_metrics = asdict(evaluate(
                naive_valid["y"].values, naive_valid["naive_da"].values
            ))
        else:
            naive_metrics = None

        rt_restored_metrics = None
        if spread_mode and "y_abs" in test_df.columns:
            rt_restored_metrics = asdict(evaluate(
                test_df["y_abs"].values, test_df["pred_rt"].values
            ))

        fold_info = {
            "fold": fold_idx,
            "train_days": len(train_dates),
            "val_days": len(val_dates_list),
            "test_days": len(test_dates),
            "test_date_start": str(test_dates[0])[:10],
            "test_date_end": str(test_dates[-1])[:10],
            "test": test_metrics,
            "naive_dayahead": naive_metrics,
            "rt_restored": rt_restored_metrics,
        }
        if two_stage or quantile_two_stage:
            fold_info["floor_threshold"] = threshold
            fold_info["floor_actual"] = int((y_test <= FLOOR_PRICE).sum())
            fold_info["floor_pred"] = int((pred_test <= FLOOR_PRICE + 1).sum())
        if quantile_mode:
            fold_info["quantile_params"] = qc_params
        if adaptive_naive_blend:
            fold_info["naive_blend_alpha"] = naive_blend_alpha
        if dynamic_floor_value:
            fold_info["floor_pred_value"] = fold_floor_value
        if time_decay_half_life > 0:
            fold_info["time_decay_half_life"] = time_decay_half_life
        if residual_correction:
            fold_info["residual_gamma"] = residual_gamma

        extra_info = []
        if time_decay_half_life > 0:
            extra_info.append(f"decay_hl={time_decay_half_life}")
        if dynamic_floor_value:
            extra_info.append(f"floor_val={fold_floor_value:.1f}")
        if adaptive_naive_blend and naive_blend_alpha > 0:
            extra_info.append(f"naive_alpha={naive_blend_alpha:.2f}")
        if residual_correction and residual_gamma > 0:
            extra_info.append(f"res_gamma={residual_gamma:.1f}")
        if rt_restored_metrics is not None:
            extra_info.append(f"RT_MAE={rt_restored_metrics['mae']:.2f}")
        extra_str = f" | {', '.join(extra_info)}" if extra_info else ""
        metric_label = "Spread_MAE" if spread_mode else "MAE"
        print(f"  Fold {fold_idx}: {metric_label}={test_metrics['mae']:.2f}  "
              f"({test_dates[0]} ~ {test_dates[-1]}){extra_str}")

        folds.append(fold_info)

        payload = {
            "model": model,
            "feature_columns": list(x_cols),
            "params": default_params,
        }
        if two_stage and clf is not None:
            payload["classifier"] = clf
            payload["threshold"] = threshold
        if quantile_mode and q_models is not None:
            payload["quantile_models"] = q_models
            payload["quantile_params"] = qc_params
        if quantile_two_stage and clf is not None:
            payload["classifier"] = clf
            payload["threshold"] = threshold
        last_payload = payload
        last_test_df = test_df.copy()
        fold_idx += 1
        start += step_days

    return folds, last_payload, last_test_df


# ---------------------------------------------------------------------------
# Expanding window CV — Mode B (rolling)
# ---------------------------------------------------------------------------

def expanding_window_cv_rolling(
    feat_base: pd.DataFrame,
    df_raw: pd.DataFrame,
    region: str,
    step_hours: int = 1,
    min_train_days: int = 60,
    val_days: int = 7,
    test_days: int = 7,
    step_days: int = 7,
    params: Optional[Dict] = None,
    feature_select_top_k: int = 0,
    two_stage: bool = False,
    quantile_mode: bool = False,
    quantile_two_stage: bool = False,
    dynamic_floor_value: bool = False,
    time_decay_half_life: int = 0,
) -> Tuple[List[Dict], Optional[Dict], Optional[pd.DataFrame]]:
    """
    Expanding window CV for Mode B: simulate rolling prediction.

    For each test day, iterate over rolling_hh = [step_hours*4, 2*step_hours*4, ...],
    augment features with cleared realtime price, then predict remaining hh.

    Returns (fold_results, last_model_payload, last_fold_test_df).
    """
    if params is None:
        params = {}
    default_params = {
        "n_estimators": 1500,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    default_params.update(params)

    dates = sorted(feat_base["trade_date"].unique())
    n_dates = len(dates)

    step_hh = step_hours * 4
    rolling_points = list(range(step_hh, 96, step_hh))

    folds = []
    last_payload = None
    last_test_df: Optional[pd.DataFrame] = None
    fold_idx = 0
    start = min_train_days

    while start + val_days + test_days <= n_dates:
        train_dates = dates[:start]
        val_dates_list = dates[start: start + val_days]
        test_dates = dates[start + val_days: start + val_days + test_days]

        # Train using augmented features at multiple rolling points
        train_records = []
        val_records = []
        for current_hh in [0] + rolling_points:
            feat_aug = augment_rolling_features(feat_base, df_raw, region, current_hh)

            train_sub = feat_aug[feat_aug["trade_date"].isin(train_dates)]
            val_sub = feat_aug[feat_aug["trade_date"].isin(val_dates_list)]
            if len(train_sub) > 0:
                train_records.append(train_sub)
            if len(val_sub) > 0:
                val_records.append(val_sub)

        if not train_records or not val_records:
            start += step_days
            continue

        train_all = pd.concat(train_records, ignore_index=True)
        val_all = pd.concat(val_records, ignore_index=True)

        x_cols = [c for c in train_all.columns if c not in ("trade_date", "y")]

        X_train = train_all[x_cols]
        y_train = train_all["y"].values
        X_val = val_all[x_cols]
        y_val = val_all["y"].values

        if len(X_train) < 96 or len(X_val) < 96:
            start += step_days
            continue

        if feature_select_top_k > 0 and feature_select_top_k < len(x_cols):
            pre_model = _fit_lgbm(
                X_train, y_train, X_val, y_val,
                {**default_params, "n_estimators": 300},
            )
            imp = pre_model.feature_importances_
            sorted_idx = np.argsort(imp)[::-1]
            x_cols = [x_cols[i] for i in sorted_idx[:feature_select_top_k] if imp[i] > 0]
            X_train = train_all[x_cols]
            X_val = val_all[x_cols]
            print(f"  [FEATURE SELECT] Kept {len(x_cols)}/{len(train_all.columns)-2} features (top-{feature_select_top_k})")

        sw = None
        if time_decay_half_life > 0:
            sw = _compute_time_decay_weights(train_all["trade_date"], time_decay_half_life)

        fold_floor_value = (
            _compute_dynamic_floor_value(y_train)
            if dynamic_floor_value
            else FLOOR_PRED_VALUE
        )

        # Mode B naive: D日滚动优先用最新已出清实时价；若不可用则退化为D日日前价
        def _rolling_naive_from_sub(test_sub: pd.DataFrame, current_hh: int) -> np.ndarray:
            if "rt_d_latest_hh" in test_sub.columns and current_hh > 0:
                rt_latest = test_sub["rt_d_latest_hh"].values
            else:
                rt_latest = np.full(len(test_sub), np.nan)
            da_anchor = test_sub["da_same_hh"].values if "da_same_hh" in test_sub.columns else np.full(len(test_sub), np.nan)
            return np.where(~np.isnan(rt_latest), rt_latest, da_anchor)

        if quantile_mode:
            q_models = {}
            for alpha in QUANTILES:
                q_models[alpha] = _fit_quantile_lgbm(
                    X_train, y_train, X_val, y_val, default_params, alpha,
                    sample_weight=sw,
                )
            model = q_models[0.50]
            naive_val_arr = val_all["da_same_hh"].values if "da_same_hh" in val_all.columns else np.full(len(val_all), np.nan)
            q_val_preds = {a: m.predict(X_val) for a, m in q_models.items()}
            qc_params = _tune_quantile_params(q_val_preds, naive_val_arr, y_val)

            if quantile_two_stage:
                clf = _fit_floor_classifier(
                    X_train, y_train, X_val, y_val, default_params,
                    sample_weight=sw,
                )
                val_qc_pred = _quantile_combine(q_val_preds, naive_val_arr, **qc_params)
                val_floor_prob = clf.predict_proba(X_val)[:, 1]
                threshold, _ = _find_best_threshold(
                    val_floor_prob, val_qc_pred, y_val,
                    floor_pred_value=fold_floor_value,
                )
            else:
                clf = None
                threshold = 0.0

        elif two_stage:
            clf = _fit_floor_classifier(
                X_train, y_train, X_val, y_val, default_params,
                sample_weight=sw,
            )
            model = _fit_lgbm(X_train, y_train, X_val, y_val, default_params, sample_weight=sw)
            val_floor_prob = clf.predict_proba(X_val)[:, 1]
            val_reg_pred = model.predict(X_val)
            threshold, _ = _find_best_threshold(
                val_floor_prob, val_reg_pred, y_val,
                floor_pred_value=fold_floor_value,
            )
            q_models = None
            qc_params = {}
        else:
            model = _fit_lgbm(X_train, y_train, X_val, y_val, default_params, sample_weight=sw)
            q_models = None
            qc_params = {}
            clf = None
            threshold = 0.0

        # Evaluate on test set at each rolling point
        horizon_metrics = {}
        all_test_preds = []
        all_test_ys = []
        all_test_naive = []
        test_df_parts = []
        for current_hh in rolling_points:
            feat_test_aug = augment_rolling_features(feat_base, df_raw, region, current_hh)
            test_sub = feat_test_aug[feat_test_aug["trade_date"].isin(test_dates)]
            if len(test_sub) == 0:
                continue
            X_test_sub = test_sub[x_cols]
            y_test_sub = test_sub["y"].values
            naive_sub = _rolling_naive_from_sub(test_sub, current_hh)

            if quantile_mode and q_models is not None:
                q_test_preds = {a: q_models[a].predict(X_test_sub) for a in QUANTILES}
                pred_sub = _quantile_combine(q_test_preds, naive_sub, **qc_params)
                if quantile_two_stage and clf is not None and threshold < 1.0:
                    floor_prob = clf.predict_proba(X_test_sub)[:, 1]
                    pred_sub = np.where(floor_prob >= threshold, fold_floor_value, pred_sub)
            elif two_stage and clf is not None and threshold < 1.0:
                pred_sub = _two_stage_predict(clf, model, X_test_sub, threshold, floor_pred_value=fold_floor_value)
            else:
                pred_sub = model.predict(X_test_sub)

            m = evaluate(y_test_sub, pred_sub)
            horizon_metrics[f"hh>{current_hh}"] = asdict(m)
            all_test_preds.extend(pred_sub)
            all_test_ys.extend(y_test_sub)
            all_test_naive.extend(naive_sub)

            tdf_part = test_sub[["trade_date", "hh_index", "y"]].copy()
            tdf_part["pred"] = pred_sub
            tdf_part["rolling_hh"] = current_hh
            test_df_parts.append(tdf_part)

        if all_test_ys:
            overall = evaluate(np.array(all_test_ys), np.array(all_test_preds))
            test_metrics = asdict(overall)
        else:
            test_metrics = {"mae": np.nan, "rmse": np.nan, "mape": np.nan}

        naive_metrics = None
        if all_test_ys and all_test_naive:
            y_arr = np.array(all_test_ys, dtype=float)
            n_arr = np.array(all_test_naive, dtype=float)
            valid = ~np.isnan(n_arr)
            if valid.any():
                naive_metrics = asdict(evaluate(y_arr[valid], n_arr[valid]))

        fold_info = {
            "fold": fold_idx,
            "train_days": len(train_dates),
            "test_date_start": str(test_dates[0])[:10],
            "test_date_end": str(test_dates[-1])[:10],
            "test": test_metrics,
            "horizon_detail": horizon_metrics,
            "naive_dayahead": naive_metrics,
        }
        if two_stage or quantile_two_stage:
            fold_info["floor_threshold"] = threshold
            fold_info["floor_actual"] = int((np.array(all_test_ys) <= FLOOR_PRICE).sum()) if all_test_ys else 0
            fold_info["floor_pred"] = int((np.array(all_test_preds) <= FLOOR_PRICE + 1).sum()) if all_test_preds else 0
        if quantile_mode:
            fold_info["quantile_params"] = qc_params
        if dynamic_floor_value:
            fold_info["floor_pred_value"] = fold_floor_value
        if time_decay_half_life > 0:
            fold_info["time_decay_half_life"] = time_decay_half_life
        print(f"  Fold {fold_idx}: MAE={test_metrics['mae']:.2f}  "
              f"({test_dates[0]} ~ {test_dates[-1]})")

        folds.append(fold_info)
        last_payload = {
            "model": model,
            "feature_columns": list(x_cols),
            "params": default_params,
        }
        if quantile_mode and q_models is not None:
            last_payload["quantile_models"] = q_models
            last_payload["quantile_params"] = qc_params
        if (two_stage or quantile_two_stage) and clf is not None:
            last_payload["classifier"] = clf
            last_payload["threshold"] = threshold

        if test_df_parts:
            last_test_df = pd.concat(test_df_parts, ignore_index=True)

        fold_idx += 1
        start += step_days

    return folds, last_payload, last_test_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

REGION_LABELS = {"jn": "江南", "jb": "江北"}


def plot_pred_vs_actual(
    test_df: pd.DataFrame,
    feat: pd.DataFrame,
    region: str,
    output_dir: str,
    df_raw: pd.DataFrame = None,
    naive_use_dm1: bool = False,
    spread_mode: bool = False,
) -> List[str]:
    plt.rcParams.update({
        "font.sans-serif": ["Arial Unicode MS", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })

    da_target = f"price_dayahead_{region}_江南" if region == "jn" else f"price_dayahead_{region}_江北"
    region_label = REGION_LABELS.get(region, region)
    test_df = test_df.sort_values(["trade_date", "hh_index"]).copy()
    test_dates = sorted(test_df["trade_date"].unique())
    n_days = len(test_dates)
    saved = []

    # 分日子图
    cols = min(n_days, 4)
    rows = (n_days + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle(f"实时价格预测 vs 真实 — {region_label} (最后一折测试集)", fontsize=14, y=1.02)

    has_quantiles = "q10" in test_df.columns and "q90" in test_df.columns

    for idx, td in enumerate(test_dates):
        ax = axes[idx // cols][idx % cols]
        day = test_df[test_df["trade_date"] == td].sort_values("hh_index")
        hh = day["hh_index"].values

        if spread_mode and "y_abs" in day.columns:
            y_true = day["y_abs"].values
            y_pred = day["pred_rt"].values
            da_anch = day["da_anchor"].values
        else:
            y_true = day["y"].values
            y_pred = day["pred"].values
            da_anch = None
        day_mae = float(mean_absolute_error(y_true, y_pred))

        if has_quantiles and not spread_mode:
            ax.fill_between(hh, day["q10"].values, day["q90"].values,
                            alpha=0.15, color="#d62728", label="q10-q90")
            ax.fill_between(hh, day["q25"].values, day["q75"].values,
                            alpha=0.25, color="#d62728", label="q25-q75")
        elif has_quantiles and spread_mode and da_anch is not None:
            ax.fill_between(hh, day["q10"].values + da_anch, day["q90"].values + da_anch,
                            alpha=0.15, color="#d62728", label="q10-q90")
            ax.fill_between(hh, day["q25"].values + da_anch, day["q75"].values + da_anch,
                            alpha=0.25, color="#d62728", label="q25-q75")
        ax.plot(hh, y_true, color="#1f77b4", linewidth=1.5, label="真实RT")
        ax.plot(hh, y_pred, color="#d62728", linewidth=1.5, linestyle="--", label="预测RT")
        if spread_mode and da_anch is not None:
            ax.plot(hh, da_anch, color="#2ca02c", linewidth=1.0, linestyle=":", alpha=0.6, label="DA锚点")
        mae_label = "RT_MAE" if spread_mode else "MAE"
        ax.set_title(f"{str(td)[:10]}  {mae_label}={day_mae:.1f}", fontsize=10)
        ax.set_xlabel("hh_index")
        ax.set_ylabel("价格")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    for idx in range(n_days, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    daily_path = os.path.join(output_dir, f"realtime_{region}_pred_vs_actual_daily.png")
    fig.savefig(daily_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(daily_path)

    # 全时段拼接时序图
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    if spread_mode and "y_abs" in test_df.columns:
        y_true_all = test_df["y_abs"].values
        y_pred_all = test_df["pred_rt"].values
    else:
        y_true_all = test_df["y"].values
        y_pred_all = test_df["pred"].values
    x_range = np.arange(len(y_true_all))

    if has_quantiles and not spread_mode:
        ax2.fill_between(x_range, test_df["q10"].values, test_df["q90"].values,
                         alpha=0.12, color="#d62728", label="q10-q90")
        ax2.fill_between(x_range, test_df["q25"].values, test_df["q75"].values,
                         alpha=0.22, color="#d62728", label="q25-q75")
    elif has_quantiles and spread_mode and "da_anchor" in test_df.columns:
        da_a = test_df["da_anchor"].values
        ax2.fill_between(x_range, test_df["q10"].values + da_a, test_df["q90"].values + da_a,
                         alpha=0.12, color="#d62728", label="q10-q90")
        ax2.fill_between(x_range, test_df["q25"].values + da_a, test_df["q75"].values + da_a,
                         alpha=0.22, color="#d62728", label="q25-q75")
    ax2.plot(x_range, y_true_all, color="#1f77b4", linewidth=1.2, label="真实RT", alpha=0.9)
    ax2.plot(x_range, y_pred_all, color="#d62728", linewidth=1.2, linestyle="--", label="预测RT", alpha=0.9)
    if spread_mode and "da_anchor" in test_df.columns:
        ax2.plot(x_range, test_df["da_anchor"].values, color="#2ca02c", linewidth=0.8,
                 linestyle=":", label="DA锚点", alpha=0.6)

    naive_y = []
    if df_raw is not None and not spread_mode:
        all_dates_sorted = sorted(df_raw["trade_date"].unique())
        date_to_idx = {d: i for i, d in enumerate(all_dates_sorted)}
        for td in test_dates:
            cur = test_df[test_df["trade_date"] == td].sort_values("hh_index")
            if naive_use_dm1:
                idx = date_to_idx.get(td)
                if idx is not None and idx >= 1:
                    dm1_date = all_dates_sorted[idx - 1]
                    dm1_df = df_raw[df_raw["trade_date"] == dm1_date].sort_values("hh_index")
                    if len(dm1_df) == 96 and da_target in dm1_df.columns:
                        da_map = dict(zip(dm1_df["hh_index"].values, dm1_df[da_target].values))
                        naive_y.extend([da_map.get(h, np.nan) for h in cur["hh_index"].values])
                    else:
                        naive_y.extend([np.nan] * len(cur))
                else:
                    naive_y.extend([np.nan] * len(cur))
            else:
                td_df = df_raw[df_raw["trade_date"] == td].sort_values("hh_index")
                if len(td_df) == 96 and da_target in td_df.columns:
                    da_map = dict(zip(td_df["hh_index"].values, td_df[da_target].values))
                    naive_y.extend([da_map.get(h, np.nan) for h in cur["hh_index"].values])
                else:
                    naive_y.extend([np.nan] * len(cur))

    naive_label = "Naive(D-1日前价)" if naive_use_dm1 else "Naive(日前价)"
    if naive_y:
        ax2.plot(x_range, naive_y, color="#2ca02c", linewidth=1.0, linestyle=":", label=naive_label, alpha=0.7)

    cum = 0
    x_ticks, x_labels = [], []
    for td in test_dates:
        n_pts = len(test_df[test_df["trade_date"] == td])
        x_ticks.append(cum)
        x_labels.append(str(td)[:10])
        if cum > 0:
            ax2.axvline(x=cum, color="gray", linewidth=0.5, alpha=0.5)
        cum += n_pts

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels, rotation=30, fontsize=9)

    overall_mae = float(mean_absolute_error(y_true_all, y_pred_all))
    naive_arr = np.array(naive_y, dtype=float) if naive_y else np.array([])
    valid_naive = ~np.isnan(naive_arr) if len(naive_arr) > 0 else np.array([])
    if isinstance(valid_naive, np.ndarray) and valid_naive.any():
        naive_mae = float(mean_absolute_error(y_true_all[valid_naive], naive_arr[valid_naive]))
        naive_title = "Naive(D-1日前)" if naive_use_dm1 else "Naive(日前)"
        title_str = f"实时价格预测 — {region_label}  |  Model MAE={overall_mae:.1f}  {naive_title} MAE={naive_mae:.1f}"
    elif spread_mode:
        spread_mae = float(mean_absolute_error(test_df["y"].values, test_df["pred"].values))
        title_str = (f"Spread预测还原RT — {region_label}  |  "
                     f"RT_MAE={overall_mae:.1f}  Spread_MAE={spread_mae:.1f}")
    else:
        title_str = f"实时价格预测 — {region_label}  |  Model MAE={overall_mae:.1f}"

    ax2.set_title(title_str, fontsize=12)
    ax2.set_ylabel("价格 (元/MWh)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    series_path = os.path.join(output_dir, f"realtime_{region}_pred_vs_actual_series.png")
    fig2.savefig(series_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    saved.append(series_path)

    if spread_mode and "y_abs" in test_df.columns:
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
        spread_true = test_df["y"].values
        spread_pred = test_df["pred"].values
        ax3a.scatter(spread_true, spread_pred, alpha=0.3, s=8, color="#d62728")
        lim = max(abs(spread_true.min()), abs(spread_true.max()),
                  abs(spread_pred.min()), abs(spread_pred.max())) * 1.1
        ax3a.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.5)
        ax3a.set_xlabel("真实 Spread (RT-DA)")
        ax3a.set_ylabel("预测 Spread")
        sp_mae = float(mean_absolute_error(spread_true, spread_pred))
        ax3a.set_title(f"{region_label} Spread 散点  MAE={sp_mae:.1f}")
        ax3a.grid(True, alpha=0.3)

        residuals = spread_pred - spread_true
        ax3b.hist(residuals, bins=50, color="#1f77b4", alpha=0.7, edgecolor="black")
        ax3b.axvline(0, color="red", linestyle="--", linewidth=1)
        ax3b.set_xlabel("Spread 残差 (预测-真实)")
        ax3b.set_ylabel("频次")
        ax3b.set_title(f"残差分布  mean={residuals.mean():.1f}  std={residuals.std():.1f}")
        ax3b.grid(True, alpha=0.3)

        fig3.tight_layout()
        scatter_path = os.path.join(output_dir, f"realtime_{region}_spread_scatter.png")
        fig3.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        saved.append(scatter_path)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="实时价格预测: 多模式支持 A0/A1/A2/B")
    ap.add_argument("--input-parquet", required=True, help="feature_ready wide parquet")
    ap.add_argument("--region", required=True, choices=["jn", "jb"], help="jn=江南, jb=江北")
    ap.add_argument("--mode", required=True, choices=["a0", "a1", "a2", "b"],
                    help="a0=D-1 09:30, a1=D-1 17:00, a2=D-1 24:00, b=D日滚动")
    ap.add_argument("--output-dir", required=True, help="输出目录")
    ap.add_argument("--min-train-days", type=int, default=60)
    ap.add_argument("--val-days", type=int, default=7)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--lgbm-params-json", default="", help="Optional LightGBM params override")
    ap.add_argument("--feature-select-top-k", type=int, default=0)
    ap.add_argument("--two-stage", action="store_true")
    ap.add_argument("--quantile", action="store_true")
    ap.add_argument("--quantile-two-stage", action="store_true")
    ap.add_argument("--time-decay", type=int, default=0, metavar="DAYS")
    ap.add_argument("--adaptive-naive-blend", action="store_true")
    ap.add_argument("--dynamic-floor-value", action="store_true")
    ap.add_argument("--residual-correction", action="store_true")
    ap.add_argument("--step-hours", type=int, default=1, help="Mode B: rolling step in hours")
    ap.add_argument("--sharpening", type=float, default=1.0,
                     help="Post-processing sharpening factor (>1 amplifies peaks/valleys)")
    ap.add_argument("--target", choices=["absolute", "spread"], default="absolute",
                    help="absolute=预测RT绝对价格(默认), spread=预测DA-RT价差")
    args = ap.parse_args()

    if sum([args.two_stage, args.quantile, args.quantile_two_stage]) > 1:
        print("ERROR: --two-stage, --quantile, --quantile-two-stage are mutually exclusive.")
        return 1

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    mode_label = {
        "a0": "A0 (D-1 09:30)",
        "a1": "A1 (D-1 17:00)",
        "a2": "A2 (D-1 24:00)",
        "b": "B (D日滚动)",
    }[args.mode]
    cutoff_hh_map = {"a0": 34, "a1": 64, "a2": 96, "b": 96}
    cutoff_hh = cutoff_hh_map[args.mode]
    include_d_dayahead = args.mode != "a0"

    print(f"[1/4] Loading data from {args.input_parquet} ...")
    df = _load_wide(args.input_parquet)
    print(f"       {len(df)} rows, {df['trade_date'].nunique()} dates")

    print(f"[2/4] Building features for region={args.region}, mode={mode_label} ...")
    print(f"       cutoff_hh={cutoff_hh} (D-1 实时价截断)")
    if not include_d_dayahead:
        print(f"       D日日前出清价: 不可用 (Mode A0)")
    feat = build_features(df, args.region, cutoff_hh=cutoff_hh,
                          include_d_dayahead=include_d_dayahead)
    feat = feat.dropna(subset=["y"]).reset_index(drop=True)

    spread_mode = args.target == "spread"
    if spread_mode:
        if args.mode == "a0":
            anchor_col = "da_dm1_anchor_same_hh"
        else:
            anchor_col = "da_same_hh"
        if anchor_col not in feat.columns:
            print(f"ERROR: anchor column '{anchor_col}' not found for spread mode.")
            return 1
        feat["y_abs"] = feat["y"].copy()
        feat["da_anchor"] = feat[anchor_col].copy()
        feat = feat.dropna(subset=["da_anchor"]).reset_index(drop=True)
        feat["y"] = feat["y"] - feat["da_anchor"]
        print(f"       [SPREAD] target = RT - DA ({anchor_col}), "
              f"spread mean={feat['y'].mean():.1f}, std={feat['y'].std():.1f}")

    print(f"       Feature table: {feat.shape[0]} rows x {feat.shape[1]} cols")
    x_cols = [c for c in feat.columns if c not in ("trade_date", "y", "y_abs", "da_anchor")]
    print(f"       Features ({len(x_cols)}): {x_cols[:10]}{'...' if len(x_cols) > 10 else ''}")
    nan_rate = feat[x_cols].isnull().mean()
    high_nan = nan_rate[nan_rate > 0.1]
    if len(high_nan) > 0:
        print(f"       High NaN features: {dict(high_nan)}")

    params = {}
    if args.lgbm_params_json and args.lgbm_params_json.strip():
        with open(args.lgbm_params_json, encoding="utf-8") as f:
            params = json.load(f)
        print(f"       Loaded LightGBM params override: {params}")

    print(f"[3/4] Running expanding window CV ...")
    print(f"       min_train={args.min_train_days}d, val={args.val_days}d, test={args.test_days}d, step={args.step_days}d")
    if args.two_stage:
        print(f"       Two-stage mode: floor classifier + regressor")
    if args.quantile:
        print(f"       Quantile mode: {QUANTILES}")
    if args.quantile_two_stage:
        print(f"       Quantile + two-stage hybrid mode")
    if args.time_decay > 0:
        print(f"       Time-decay: half_life={args.time_decay} days")
    if args.adaptive_naive_blend:
        print(f"       Adaptive naive blending: ON")
    if args.dynamic_floor_value:
        print(f"       Dynamic floor value: ON")
    if args.residual_correction:
        print(f"       Residual correction: ON")

    if args.mode == "b":
        folds, last_payload, last_test_df = expanding_window_cv_rolling(
            feat, df, args.region,
            step_hours=args.step_hours,
            min_train_days=args.min_train_days,
            val_days=args.val_days,
            test_days=args.test_days,
            step_days=args.step_days,
            params=params,
            feature_select_top_k=args.feature_select_top_k,
            two_stage=args.two_stage,
            quantile_mode=args.quantile or args.quantile_two_stage,
            quantile_two_stage=args.quantile_two_stage,
            dynamic_floor_value=args.dynamic_floor_value,
            time_decay_half_life=args.time_decay,
        )
    else:
        cv_kwargs = dict(
            min_train_days=args.min_train_days,
            val_days=args.val_days,
            test_days=args.test_days,
            step_days=args.step_days,
            params=params,
            feature_select_top_k=args.feature_select_top_k,
            two_stage=args.two_stage,
            quantile_mode=args.quantile or args.quantile_two_stage,
            quantile_two_stage=args.quantile_two_stage,
            time_decay_half_life=args.time_decay,
            adaptive_naive_blend=args.adaptive_naive_blend,
            dynamic_floor_value=args.dynamic_floor_value,
            residual_correction=args.residual_correction,
            naive_use_dm1=(args.mode == "a0"),
            sharpening=args.sharpening,
            spread_mode=spread_mode,
        )
        folds, last_payload, last_test_df = expanding_window_cv(
            feat, df, args.region, **cv_kwargs
        )

    if not folds:
        print("ERROR: No valid CV folds (not enough data).")
        return 1

    mean_mae = np.mean([f["test"]["mae"] for f in folds])
    mean_rmse = np.mean([f["test"]["rmse"] for f in folds])
    mean_mape = np.mean([f["test"]["mape"] for f in folds])

    naive_key = "naive_dayahead"
    naive_maes = [f[naive_key]["mae"] for f in folds if f.get(naive_key)]
    mean_naive_mae = np.mean(naive_maes) if naive_maes else None

    print(f"\n{'='*60}")
    target_desc = "Spread" if spread_mode else "Absolute"
    print(f"  CV Results ({len(folds)} folds) — Mode {mode_label} — Target: {target_desc}")
    metric_prefix = "Spread" if spread_mode else "Model"
    print(f"  {metric_prefix}  MAE  = {mean_mae:.2f}")
    print(f"  {metric_prefix}  RMSE = {mean_rmse:.2f}")
    print(f"  {metric_prefix}  MAPE = {mean_mape:.4f}")

    if spread_mode:
        rt_restored_maes = [f["rt_restored"]["mae"] for f in folds if f.get("rt_restored")]
        if rt_restored_maes:
            mean_rt_mae = np.mean(rt_restored_maes)
            print(f"  RT还原 MAE  = {mean_rt_mae:.2f}")

    if mean_naive_mae is not None:
        naive_desc = "spread=0" if spread_mode else ("D-1日前价" if args.mode == "a0" else "日前价")
        print(f"  Naive({naive_desc}) MAE = {mean_naive_mae:.2f}")
        improvement = (mean_naive_mae - mean_mae) / mean_naive_mae * 100
        print(f"  Improvement = {improvement:+.1f}%")

    if args.mode == "b":
        print(f"\n  --- Horizon breakdown (last fold) ---")
        if folds and folds[-1].get("horizon_detail"):
            for hkey, hm in folds[-1]["horizon_detail"].items():
                print(f"    {hkey:>10s}: MAE={hm['mae']:.2f}")
    print(f"{'='*60}\n")

    print(f"[4/4] Saving results ...")

    target_col = REALTIME_TARGETS[args.region]
    cv_summary: Dict = {
        "mode": args.mode,
        "target": args.target,
        "cutoff_hh": cutoff_hh,
        "fold_count": len(folds),
        "mean_mae": round(mean_mae, 4),
        "mean_rmse": round(mean_rmse, 4),
        "mean_mape": round(mean_mape, 6),
        "mean_naive_dayahead_mae": round(mean_naive_mae, 4) if mean_naive_mae else None,
    }
    if spread_mode:
        rt_restored_maes = [f["rt_restored"]["mae"] for f in folds if f.get("rt_restored")]
        if rt_restored_maes:
            cv_summary["mean_rt_restored_mae"] = round(float(np.mean(rt_restored_maes)), 4)

    results = {
        "region": args.region,
        "target_col": target_col,
        "mode": args.mode,
        "cutoff_hh": cutoff_hh,
        "data_rows": int(feat.shape[0]),
        "feature_count": len(x_cols),
        "feature_names": x_cols,
        "n_dates": int(feat["trade_date"].nunique()),
        "cv_summary": cv_summary,
        "folds": folds,
    }
    metrics_path = os.path.join(output_dir, f"realtime_{args.region}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Metrics -> {metrics_path}")

    if last_payload:
        model_path = os.path.join(output_dir, f"realtime_{args.region}_model.joblib")
        joblib.dump(last_payload, model_path)
        print(f"  Model  -> {model_path}")

    fold_rows = []
    for fold in folds:
        row = {
            "fold": fold["fold"],
            "train_days": fold["train_days"],
            "test_date_range": f"{fold['test_date_start']}~{fold['test_date_end']}",
            "model_mae": fold["test"]["mae"],
            "model_rmse": fold["test"]["rmse"],
            "model_mape": fold["test"]["mape"],
        }
        if fold.get("naive_dayahead"):
            row["naive_dayahead_mae"] = fold["naive_dayahead"]["mae"]
        fold_rows.append(row)

    summary_df = pd.DataFrame(fold_rows)
    summary_path = os.path.join(output_dir, f"realtime_{args.region}_cv_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  Summary -> {summary_path}")

    if last_payload and "feature_columns" in last_payload and "model" in last_payload:
        model_feat_cols = last_payload["feature_columns"]
        imp = last_payload["model"].feature_importances_
        imp_df = pd.DataFrame({
            "feature": model_feat_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False)
        imp_path = os.path.join(output_dir, f"realtime_{args.region}_feature_importance.csv")
        imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
        print(f"  Feature importance -> {imp_path}")
        print(f"\n  Top-10 features:")
        for _, r in imp_df.head(10).iterrows():
            print(f"    {r['feature']:50s}  {int(r['importance']):>6d}")

    if last_test_df is not None and len(last_test_df) > 0:
        if args.mode == "b" and "rolling_hh" in last_test_df.columns:
            last_test_df = (
                last_test_df
                .sort_values("rolling_hh", ascending=False)
                .drop_duplicates(subset=["trade_date", "hh_index"], keep="first")
                .sort_values(["trade_date", "hh_index"])
                .reset_index(drop=True)
            )

        print(f"\n  Plotting prediction vs actual (last fold) ...")
        plot_files = plot_pred_vs_actual(
            last_test_df, feat, args.region, output_dir, df_raw=df,
            naive_use_dm1=(args.mode == "a0"),
            spread_mode=spread_mode,
        )
        for pf in plot_files:
            print(f"  Plot -> {pf}")

        tdf = last_test_df.copy()
        if spread_mode and "y_abs" in tdf.columns:
            tdf["error"] = (tdf["pred_rt"] - tdf["y_abs"]).abs()
            y_col_for_floor = "y_abs"
        else:
            tdf["error"] = (tdf["pred"] - tdf["y"]).abs()
            y_col_for_floor = "y"
        TIME_BLOCKS = {
            "00:00-06:00 (hh 1-24)": (1, 24),
            "06:00-12:00 (hh 25-48)": (25, 48),
            "12:00-18:00 (hh 49-72)": (49, 72),
            "18:00-24:00 (hh 73-96)": (73, 96),
        }
        err_label = "RT_MAE" if spread_mode else "MAE"
        print(f"\n  分时段误差分析 (最后一折):")
        print(f"  {'时段':<30s}  {err_label:>8s}  {'样本数':>6s}  {'地板价占比':>10s}")
        for label, (hh_lo, hh_hi) in TIME_BLOCKS.items():
            mask = (tdf["hh_index"] >= hh_lo) & (tdf["hh_index"] <= hh_hi)
            sub = tdf[mask]
            if len(sub) == 0:
                continue
            block_mae = sub["error"].mean()
            floor_pct = (sub[y_col_for_floor] <= FLOOR_PRICE).mean() * 100
            print(f"  {label:<30s}  {block_mae:>8.1f}  {len(sub):>6d}  {floor_pct:>9.1f}%")

        floor_mask = tdf[y_col_for_floor] <= FLOOR_PRICE
        normal_mask = tdf[y_col_for_floor] > 200
        if floor_mask.any():
            floor_mae = tdf.loc[floor_mask, "error"].mean()
            print(f"\n  地板价时段 (y<={FLOOR_PRICE}):  MAE={floor_mae:.1f}  ({floor_mask.sum()} 样本)")
        if normal_mask.any():
            normal_mae = tdf.loc[normal_mask, "error"].mean()
            print(f"  正常价时段 (y>200): MAE={normal_mae:.1f}  ({normal_mask.sum()} 样本)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
