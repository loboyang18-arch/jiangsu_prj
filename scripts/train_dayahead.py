"""
日前价格预测 — 全新实现

任务定义:
  D-1 日 09:30，预测 D 日 96 个 15 分钟时段的日前出清价格。

可用信息:
  1) D 日 boundary 曲线（D-1 09:00 已发布）: load_forecast_boundary, wind_forecast_boundary,
     pv_forecast_boundary, gas_plan_boundary, receive_plan_boundary, reserve_positive, reserve_negative
  2) D-1 日及更早的历史日前出清价: price_dayahead_{jn/jb}（D-1 17:00 发布）
  3) 日历特征: dow, month, is_weekend, hh_index

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
# Feature building
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

PRICE_TARGETS = {
    "jn": "price_dayahead_jn_江南",
    "jb": "price_dayahead_jb_江北",
}

HISTORY_PRICE_COLS = [
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


def _load_wide(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["trade_date", "hh_index"]).reset_index(drop=True)
    return df


def _daily_price_stats(df: pd.DataFrame, price_col: str, prefix: str) -> pd.DataFrame:
    """D-1 日历史价格 → 统计量 (mean, std, min, max, median, last_hh)."""
    daily = df.groupby("trade_date")[price_col].agg(
        **{
            f"{prefix}_mean": "mean",
            f"{prefix}_std": "std",
            f"{prefix}_min": "min",
            f"{prefix}_max": "max",
            f"{prefix}_median": "median",
        }
    ).reset_index()
    last_hh = (
        df.sort_values("hh_index")
        .groupby("trade_date")[price_col]
        .last()
        .rename(f"{prefix}_last_hh")
        .reset_index()
    )
    daily = daily.merge(last_hh, on="trade_date", how="left")
    return daily


def _daily_boundary_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """D 日 boundary 曲线 → 按 hh_index 保留原值 + 日均/日标准差."""
    prefix = col.split("_汇总")[0].split("_江")[0]
    daily_stats = df.groupby("trade_date")[col].agg(
        **{
            f"{prefix}_day_mean": "mean",
            f"{prefix}_day_std": "std",
        }
    ).reset_index()
    return daily_stats


def build_features(
    df: pd.DataFrame,
    region: str,
    include_actual_exog: bool = False,
) -> pd.DataFrame:
    """
    构建日前预测特征表。

    返回 DataFrame 每行 = (trade_date, hh_index)，包含:
      - y: 目标价格
      - hh_index: 时段编号 (0-95)
      - boundary 当时段值 (D 日曲线)
      - boundary 日统计量 (D 日)
      - D-1 价格统计量
      - D-2 价格统计量
      - 日历特征

    include_actual_exog: if True, add D-day actual exogenous features
        (oracle experiment — not available in production).
    """
    target_col = PRICE_TARGETS[region]
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

        for col in available_boundary:
            short = col.replace("_汇总", "").replace("_江北", "_jb").replace("_江南", "_jn")
            vals = day_df[col].values
            row[f"bnd_{short}"] = vals

            diff = np.diff(vals, prepend=vals[0])
            row[f"bnd_{short}_diff"] = diff

            row[f"bnd_{short}_day_mean"] = np.nanmean(vals)
            row[f"bnd_{short}_day_std"] = np.nanstd(vals)
            row[f"bnd_{short}_rank"] = pd.Series(vals).rank(pct=True).values

        load_col = "load_forecast_boundary_汇总"
        wind_jb = "wind_forecast_boundary_江北"
        wind_jn = "wind_forecast_boundary_江南"
        pv_jb = "pv_forecast_boundary_江北"
        pv_jn = "pv_forecast_boundary_江南"
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

        for lag_d, lag_label in [(1, "dm1"), (2, "dm2"), (7, "dm7")]:
            if i >= lag_d:
                prev_date = dates[i - lag_d]
                prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
                if len(prev_df) == 96:
                    for pcol in HISTORY_PRICE_COLS:
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

        # --- D-1 实时价格特征 ---
        for lag_d, lag_label in [(1, "dm1"), (2, "dm2")]:
            if i >= lag_d:
                prev_date = dates[i - lag_d]
                prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
                if len(prev_df) == 96:
                    for pcol in REALTIME_PRICE_COLS:
                        if pcol not in prev_df.columns:
                            continue
                        short = pcol.replace("price_realtime_", "rt_").replace("_final", "").replace("_江南", "_jn").replace("_江北", "_jb").replace("_node", "_nd")
                        vals = prev_df[pcol].values
                        hh_vals = row["hh_index"]
                        row[f"{short}_{lag_label}_mean"] = np.nanmean(vals)
                        row[f"{short}_{lag_label}_std"] = np.nanstd(vals)
                        row[f"{short}_{lag_label}_min"] = np.nanmin(vals)
                        row[f"{short}_{lag_label}_max"] = np.nanmax(vals)
                        row[f"{short}_{lag_label}_same_hh"] = vals[hh_vals - 1]
                        row[f"{short}_{lag_label}_floor_rate"] = (vals <= 50).mean()
                        row[f"{short}_{lag_label}_spike_rate"] = (vals >= 400).mean()

        # --- D-1 日前-实时价差 (market spread) ---
        if i >= 1:
            prev_date = dates[i - 1]
            prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
            if len(prev_df) == 96:
                for da_col, rt_col, label in [
                    ("price_dayahead_jn_江南", "price_realtime_jn_final_江南", "jn"),
                    ("price_dayahead_jb_江北", "price_realtime_jb_final_江北", "jb"),
                ]:
                    if da_col in prev_df.columns and rt_col in prev_df.columns:
                        spread = prev_df[da_col].values - prev_df[rt_col].values
                        row[f"dm1_spread_{label}_mean"] = np.mean(spread)
                        row[f"dm1_spread_{label}_std"] = np.std(spread)
                        row[f"dm1_spread_{label}_abs_mean"] = np.mean(np.abs(spread))
                        hh_vals = row["hh_index"]
                        row[f"dm1_spread_{label}_same_hh"] = spread[hh_vals - 1]

        # --- D-1 actual 数据特征 ---
        for lag_d, lag_label in [(1, "dm1")]:
            if i >= lag_d:
                prev_date = dates[i - lag_d]
                prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
                if len(prev_df) == 96:
                    for orig_col, short in ACTUAL_COLS.items():
                        if orig_col not in prev_df.columns:
                            continue
                        vals = prev_df[orig_col].values
                        row[f"{short}_{lag_label}_mean"] = np.nanmean(vals)
                        row[f"{short}_{lag_label}_std"] = np.nanstd(vals)
                        hh_vals = row["hh_index"]
                        row[f"{short}_{lag_label}_same_hh"] = vals[hh_vals - 1]

                    re_actual = (prev_df.get("wind_actual_江北", pd.Series(0)).values
                                 + prev_df.get("wind_actual_江南", pd.Series(0)).values
                                 + prev_df.get("pv_actual_江北", pd.Series(0)).values
                                 + prev_df.get("pv_actual_江南", pd.Series(0)).values)
                    load_actual = prev_df.get("load_actual_total_汇总", pd.Series(0)).values
                    net_load_actual = load_actual - re_actual
                    row[f"net_load_actual_{lag_label}_mean"] = np.nanmean(net_load_actual)
                    row[f"net_load_actual_{lag_label}_min"] = np.nanmin(net_load_actual)
                    row[f"net_load_actual_{lag_label}_same_hh"] = net_load_actual[hh_vals - 1]
                    row[f"re_penetration_actual_{lag_label}"] = np.where(
                        load_actual > 0, re_actual / load_actual, 0.0
                    ).mean()

                    if "load_forecast_boundary_汇总" in prev_df.columns:
                        row[f"load_fcst_err_{lag_label}_mean"] = np.mean(
                            prev_df["load_forecast_boundary_汇总"].values - load_actual
                        )
                    re_bnd = (prev_df.get("wind_forecast_boundary_江北", pd.Series(0)).values
                              + prev_df.get("wind_forecast_boundary_江南", pd.Series(0)).values
                              + prev_df.get("pv_forecast_boundary_江北", pd.Series(0)).values
                              + prev_df.get("pv_forecast_boundary_江南", pd.Series(0)).values)
                    row[f"re_fcst_err_{lag_label}_mean"] = np.mean(re_bnd - re_actual)
                    row[f"re_fcst_err_{lag_label}_std"] = np.std(re_bnd - re_actual)

        # --- D-1 地板价统计 ---
        if i >= 1:
            prev_date = dates[i - 1]
            prev_df = df[df["trade_date"] == prev_date].sort_values("hh_index")
            if len(prev_df) == 96 and target_col in prev_df.columns:
                prev_prices = prev_df[target_col].values
                row["dm1_floor_rate"] = (prev_prices <= 50).mean()
                row["dm1_floor_count"] = (prev_prices <= 50).sum()
                row["dm1_spike_rate"] = (prev_prices >= 400).mean()

        if i >= 3:
            for pcol in HISTORY_PRICE_COLS:
                if pcol not in df.columns:
                    continue
                short = pcol.replace("price_dayahead_", "p_").replace("_江南", "_jn").replace("_江北", "_jb").replace("_node", "_nd")
                recent_dates = dates[max(0, i - 3):i]
                recent_df = df[df["trade_date"].isin(recent_dates)]
                if len(recent_df) > 0:
                    daily_means = recent_df.groupby("trade_date")[pcol].mean()
                    row[f"{short}_3d_avg"] = daily_means.mean()
                    row[f"{short}_3d_trend"] = daily_means.iloc[-1] - daily_means.iloc[0] if len(daily_means) > 1 else 0.0

        # --- 近7天价格趋势和波动率 ---
        if i >= 7:
            for pcol, label in [(target_col, region), *[(c, c.split("_")[2]) for c in HISTORY_PRICE_COLS if c != target_col and c in df.columns]]:
                week_means = []
                for j in range(7):
                    dj = dates[i - 7 + j]
                    djdf = df[df["trade_date"] == dj]
                    if len(djdf) == 96 and pcol in djdf.columns:
                        week_means.append(djdf[pcol].mean())
                if len(week_means) == 7:
                    x = np.arange(7, dtype=float)
                    slope = np.polyfit(x, week_means, 1)[0]
                    row[f"trend_7d_{label}"] = slope
                    row[f"vol_7d_{label}"] = np.std(week_means)

        # --- 历史同时段地板价频率 (近7天) ---
        if i >= 7 and target_col in df.columns:
            hist_dates = dates[max(0, i - 7):i]
            hist_df = df[df["trade_date"].isin(hist_dates)].sort_values(["trade_date", "hh_index"])
            if len(hist_df) > 0:
                hh_floor_rate = hist_df.groupby("hh_index")[target_col].apply(lambda x: (x <= 50).mean())
                hh_idx = row["hh_index"]
                row["hh_floor_rate_7d"] = pd.Series(hh_idx).map(hh_floor_rate).values

        # --- D-1 实时价格同时段地板价指示 (近7天) ---
        if i >= 7:
            for rt_col, label in [("price_realtime_jn_final_江南", "rt_jn"), ("price_realtime_jb_final_江北", "rt_jb")]:
                if rt_col in df.columns:
                    hist_dates = dates[max(0, i - 7):i]
                    hist_df = df[df["trade_date"].isin(hist_dates)].sort_values(["trade_date", "hh_index"])
                    if len(hist_df) > 0:
                        hh_rt_floor = hist_df.groupby("hh_index")[rt_col].apply(lambda x: (x <= 50).mean())
                        hh_idx = row["hh_index"]
                        row[f"hh_{label}_floor_rate_7d"] = pd.Series(hh_idx).map(hh_rt_floor).values

        # --- D 日 actual 外生变量 (oracle experiment only) ---
        if include_actual_exog:
            act_load = day_df.get("load_actual_total_汇总")
            act_wind_jb = day_df.get("wind_actual_江北")
            act_wind_jn = day_df.get("wind_actual_江南")
            act_pv_jb = day_df.get("pv_actual_江北")
            act_pv_jn = day_df.get("pv_actual_江南")
            act_gas_jb = day_df.get("gas_actual_江北")
            act_gas_jn = day_df.get("gas_actual_江南")
            act_recv = day_df.get("receive_actual_huadong_华东")

            if act_load is not None and act_load.notna().all():
                row["actual_load"] = act_load.values
                row["actual_load_day_mean"] = np.nanmean(act_load.values)
                row["actual_load_rank"] = pd.Series(act_load.values).rank(pct=True).values

            act_re = np.zeros(96)
            has_re = False
            for src in [act_wind_jb, act_wind_jn, act_pv_jb, act_pv_jn]:
                if src is not None and src.notna().all():
                    act_re = act_re + src.values
                    has_re = True
            if has_re:
                row["actual_re_total"] = act_re
                row["actual_re_day_mean"] = np.nanmean(act_re)
                if act_load is not None and act_load.notna().all():
                    load_v = act_load.values
                    net = load_v - act_re
                    row["actual_net_load"] = net
                    row["actual_net_load_day_mean"] = np.nanmean(net)
                    row["actual_net_load_rank"] = pd.Series(net).rank(pct=True).values
                    row["actual_re_penetration"] = np.where(load_v > 0, act_re / load_v, 0.0)

            if act_pv_jb is not None and act_pv_jb.notna().all():
                row["actual_pv_jb"] = act_pv_jb.values
            if act_pv_jn is not None and act_pv_jn.notna().all():
                row["actual_pv_jn"] = act_pv_jn.values
            if act_wind_jb is not None and act_wind_jb.notna().all():
                row["actual_wind_jb"] = act_wind_jb.values
            if act_wind_jn is not None and act_wind_jn.notna().all():
                row["actual_wind_jn"] = act_wind_jn.values
            if act_gas_jb is not None and act_gas_jb.notna().all():
                row["actual_gas_jb"] = act_gas_jb.values
            if act_gas_jn is not None and act_gas_jn.notna().all():
                row["actual_gas_jn"] = act_gas_jn.values
            if act_recv is not None and act_recv.notna().all():
                row["actual_receive"] = act_recv.values

        dt = pd.Timestamp(d)
        row["dow"] = dt.dayofweek
        row["month"] = dt.month
        row["is_weekend"] = int(dt.dayofweek >= 5)

        records.append(pd.DataFrame(row))

    feat = pd.concat(records, ignore_index=True)
    feat["trade_date"] = pd.to_datetime(feat["trade_date"])
    return feat


# ---------------------------------------------------------------------------
# Training & CV
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


# ---------------------------------------------------------------------------
# Quantile regression
# ---------------------------------------------------------------------------

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


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
) -> np.ndarray:
    """Combine multi-quantile predictions into a single point forecast.

    Floor strategy: when q10 is low, blend q10 into the prediction with
    a configurable weight instead of hard-replacing. This avoids the
    catastrophic errors of the binary classifier approach while still
    pulling predictions downward when floor prices are likely.
    """
    q10 = q_preds[0.10]
    q25 = q_preds[0.25]
    q50 = q_preds[0.50]
    q90 = q_preds[0.90]

    pred = q50.copy()

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
) -> Dict:
    """Grid-search quantile combination parameters on the validation set."""
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

FLOOR_PRICE = 50.0
FLOOR_PRED_VALUE = 30.0   # default fallback; overridden dynamically per fold


# ---------------------------------------------------------------------------
# Time-decay sample weighting
# ---------------------------------------------------------------------------

def _compute_time_decay_weights(
    trade_dates: pd.Series,
    half_life_days: int = 30,
) -> np.ndarray:
    """Exponential time-decay weights: more recent samples get higher weight.

    weight_i = 2 ** ((date_i - date_max) / half_life_days)
    So the newest sample has weight 1.0, and a sample half_life_days older
    has weight 0.5.
    """
    date_nums = pd.to_datetime(trade_dates).astype(np.int64) // 10**9
    max_t = date_nums.max()
    diff_days = (date_nums - max_t) / 86400.0
    weights = np.power(2.0, diff_days / half_life_days)
    return weights


# ---------------------------------------------------------------------------
# Adaptive model-naive blending (post-processing)
# ---------------------------------------------------------------------------

def _tune_adaptive_naive_blend(
    pred: np.ndarray,
    naive: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[float, float]:
    """Grid-search per-fold optimal blend ratio between model and naive on val set.

    Returns (best_alpha, best_mae) where final = (1-alpha)*model + alpha*naive.
    alpha=0 means pure model.
    """
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
    """Apply the tuned blend ratio."""
    if alpha <= 0.0:
        return pred
    result = pred.copy()
    valid = ~np.isnan(naive)
    result[valid] = (1 - alpha) * pred[valid] + alpha * naive[valid]
    return result


def _compute_dynamic_floor_value(y_train: np.ndarray) -> float:
    """Use median of actual floor prices from training set as the fill value."""
    floor_vals = y_train[y_train <= FLOOR_PRICE]
    if len(floor_vals) > 0:
        return float(np.median(floor_vals))
    return FLOOR_PRED_VALUE


# ---------------------------------------------------------------------------
# Residual auto-correction
# ---------------------------------------------------------------------------

def _compute_residual_bias(
    pred_val: np.ndarray,
    y_val: np.ndarray,
    hh_val: np.ndarray,
) -> Dict[int, float]:
    """Compute per-hh_index mean residual bias on the validation set."""
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
    """Tune damping factor gamma for residual correction on val set.

    Uses leave-one-day-out within the val set: for each val day,
    compute bias from the other val days, apply correction, measure MAE.
    If val set has only 1 unique day, fall back to full-set bias with gamma search.
    """
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
    """Apply per-hh bias correction with damping factor."""
    if gamma <= 0.0:
        return pred
    result = pred.copy()
    for hh, b in bias.items():
        mask = hh_arr == hh
        result[mask] += gamma * b
    return result


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
    """Search threshold on val set that minimizes MAE.

    Uses threshold >= 0.40 with precision >= 0.60 as minimum bar.
    The final check is pure MAE improvement: only adopt if combined MAE < reg MAE.
    """
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


def expanding_window_cv(
    feat: pd.DataFrame,
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
) -> Tuple[List[Dict], Optional[Dict], Optional[pd.DataFrame]]:
    """
    Expanding window CV by trade_date.

    Returns (fold_results, last_model_payload, last_fold_test_df).
    last_fold_test_df contains columns: trade_date, hh_index, y, pred.
    In quantile_mode, last_fold_test_df also has q10, q25, q50, q75, q90.
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

    dates = sorted(feat["trade_date"].unique())
    n_dates = len(dates)

    x_cols = [c for c in feat.columns if c not in ("trade_date", "y")]

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
        val_dates = dates[start: start + val_days]
        test_dates = dates[start + val_days: start + val_days + test_days]

        train_mask = feat["trade_date"].isin(train_dates)
        val_mask = feat["trade_date"].isin(val_dates)
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

        # --- Sample weights (time-decay) ---
        sw = None
        if time_decay_half_life > 0:
            sw = _compute_time_decay_weights(
                feat.loc[train_mask, "trade_date"], time_decay_half_life
            )

        # --- Dynamic floor prediction value ---
        fold_floor_value = (
            _compute_dynamic_floor_value(y_train)
            if dynamic_floor_value
            else FLOOR_PRED_VALUE
        )

        # --- Build naive predictions for val and test ---
        def _get_naive(target_dates):
            naive_vals = []
            for td in target_dates:
                td_idx = list(dates).index(td)
                if td_idx > 0:
                    prev_d = dates[td_idx - 1]
                    prev_v = feat.loc[
                        feat["trade_date"] == prev_d, ["hh_index", "y"]
                    ].set_index("hh_index")["y"]
                    cur = feat.loc[feat["trade_date"] == td, ["hh_index", "y"]].copy()
                    cur["naive_prev"] = cur["hh_index"].map(prev_v)
                    naive_vals.append(cur)
                else:
                    cur = feat.loc[feat["trade_date"] == td, ["hh_index", "y"]].copy()
                    cur["naive_prev"] = np.nan
                    naive_vals.append(cur)
            if naive_vals:
                return pd.concat(naive_vals)
            return pd.DataFrame(columns=["hh_index", "y", "naive_prev"])

        naive_test_df = _get_naive(test_dates)
        naive_test_arr = naive_test_df["naive_prev"].values

        if quantile_mode:
            q_models = {}
            for alpha in QUANTILES:
                q_models[alpha] = _fit_quantile_lgbm(
                    X_train, y_train, X_val, y_val, default_params, alpha,
                    sample_weight=sw,
                )
            model = q_models[0.50]

            naive_val_df = _get_naive(val_dates)
            naive_val_arr = naive_val_df["naive_prev"].values
            q_val_preds = {a: m.predict(X_val) for a, m in q_models.items()}
            qc_params = _tune_quantile_params(q_val_preds, naive_val_arr, y_val)

            q_test_preds = {a: m.predict(X_test) for a, m in q_models.items()}
            pred_test = _quantile_combine(q_test_preds, naive_test_arr, **qc_params)

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
                X_train, y_train, X_val, y_val, default_params,
                sample_weight=sw,
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

        # --- Compute val predictions (needed for adaptive blend & residual) ---
        need_val_pred = adaptive_naive_blend or residual_correction
        pred_val = None
        if need_val_pred:
            naive_val_df_blend = _get_naive(val_dates)
            naive_val_arr_blend = naive_val_df_blend["naive_prev"].values
            if quantile_mode:
                pred_val = _quantile_combine(
                    {a: q_models[a].predict(X_val) for a in QUANTILES},
                    naive_val_arr_blend, **qc_params
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

        # --- Adaptive model-naive blending (post-processing) ---
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

        # --- Residual auto-correction (post-processing) ---
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
        if quantile_mode and q_test_preds is not None:
            for alpha in QUANTILES:
                test_df[f"q{int(alpha*100):02d}"] = q_test_preds[alpha]

        naive_valid = naive_test_df.dropna(subset=["naive_prev"])
        if len(naive_valid) > 0:
            naive_metrics = asdict(evaluate(
                naive_valid["y"].values, naive_valid["naive_prev"].values
            ))
        else:
            naive_metrics = None

        fold_info = {
            "fold": fold_idx,
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "test_days": len(test_dates),
            "test_date_start": str(test_dates[0])[:10],
            "test_date_end": str(test_dates[-1])[:10],
            "test": test_metrics,
            "naive_yesterday": naive_metrics,
        }
        if two_stage:
            fold_info["floor_threshold"] = threshold
            floor_actual = (y_test <= FLOOR_PRICE).sum()
            floor_pred = (pred_test <= FLOOR_PRICE + 1).sum()
            fold_info["floor_actual"] = int(floor_actual)
            fold_info["floor_pred"] = int(floor_pred)
        if quantile_mode:
            fold_info["quantile_params"] = qc_params
            fold_info["floor_actual"] = int((y_test <= FLOOR_PRICE).sum())
            fold_info["floor_pred"] = int((pred_test <= FLOOR_PRICE + 1).sum())
            if quantile_two_stage:
                fold_info["floor_threshold"] = threshold if quantile_two_stage else 0.0
        if adaptive_naive_blend:
            fold_info["naive_blend_alpha"] = naive_blend_alpha
        if dynamic_floor_value:
            fold_info["floor_pred_value"] = fold_floor_value
        if time_decay_half_life > 0:
            fold_info["time_decay_half_life"] = time_decay_half_life
        if residual_correction:
            fold_info["residual_gamma"] = residual_gamma
            if residual_bias:
                top_biases = sorted(residual_bias.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                fold_info["residual_top_biases"] = {f"hh{k}": round(v, 1) for k, v in top_biases}

        extra_info = []
        if time_decay_half_life > 0:
            extra_info.append(f"decay_hl={time_decay_half_life}")
        if dynamic_floor_value:
            extra_info.append(f"floor_val={fold_floor_value:.1f}")
        if adaptive_naive_blend and naive_blend_alpha > 0:
            extra_info.append(f"naive_alpha={naive_blend_alpha:.2f}")
        if residual_correction and residual_gamma > 0:
            extra_info.append(f"res_gamma={residual_gamma:.1f}")
        extra_str = f" | {', '.join(extra_info)}" if extra_info else ""
        print(f"  Fold {fold_idx}: MAE={test_metrics['mae']:.2f}  "
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
# Plotting
# ---------------------------------------------------------------------------

REGION_LABELS = {"jn": "江南", "jb": "江北"}


def plot_pred_vs_actual(
    test_df: pd.DataFrame,
    feat: pd.DataFrame,
    region: str,
    output_dir: str,
) -> List[str]:
    """
    绘制最后一折测试集的预测 vs 真实曲线。

    生成两张图:
      1) 分日对比 (每天一个子图)
      2) 全时段拼接时序图
    """
    plt.rcParams.update({
        "font.sans-serif": ["Arial Unicode MS", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })

    region_label = REGION_LABELS.get(region, region)
    test_df = test_df.sort_values(["trade_date", "hh_index"]).copy()
    test_dates = sorted(test_df["trade_date"].unique())
    n_days = len(test_dates)
    saved = []

    # --- 1) 分日子图 ---
    cols = min(n_days, 4)
    rows = (n_days + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle(f"日前价格预测 vs 真实 — {region_label} (最后一折测试集)", fontsize=14, y=1.02)

    has_quantiles = "q10" in test_df.columns and "q90" in test_df.columns

    for idx, td in enumerate(test_dates):
        ax = axes[idx // cols][idx % cols]
        day = test_df[test_df["trade_date"] == td].sort_values("hh_index")
        hh = day["hh_index"].values
        y_true = day["y"].values
        y_pred = day["pred"].values
        day_mae = float(mean_absolute_error(y_true, y_pred))

        if has_quantiles:
            ax.fill_between(hh, day["q10"].values, day["q90"].values,
                            alpha=0.15, color="#d62728", label="q10-q90")
            ax.fill_between(hh, day["q25"].values, day["q75"].values,
                            alpha=0.25, color="#d62728", label="q25-q75")
        ax.plot(hh, y_true, color="#1f77b4", linewidth=1.5, label="真实")
        ax.plot(hh, y_pred, color="#d62728", linewidth=1.5, linestyle="--", label="预测")
        ax.set_title(f"{str(td)[:10]}  MAE={day_mae:.1f}", fontsize=10)
        ax.set_xlabel("hh_index")
        ax.set_ylabel("价格")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    for idx in range(n_days, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    daily_path = os.path.join(output_dir, f"dayahead_{region}_pred_vs_actual_daily.png")
    fig.savefig(daily_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(daily_path)

    # --- 2) 全时段拼接时序 ---
    dates_all = sorted(feat["trade_date"].unique())
    fig2, ax2 = plt.subplots(figsize=(14, 5))

    x_ticks = []
    x_labels = []
    y_true_all = test_df["y"].values
    y_pred_all = test_df["pred"].values
    x_range = np.arange(len(y_true_all))

    if has_quantiles:
        ax2.fill_between(x_range, test_df["q10"].values, test_df["q90"].values,
                         alpha=0.12, color="#d62728", label="q10-q90")
        ax2.fill_between(x_range, test_df["q25"].values, test_df["q75"].values,
                         alpha=0.22, color="#d62728", label="q25-q75")
    ax2.plot(x_range, y_true_all, color="#1f77b4", linewidth=1.2, label="真实", alpha=0.9)
    ax2.plot(x_range, y_pred_all, color="#d62728", linewidth=1.2, linestyle="--", label="预测", alpha=0.9)

    naive_y = []
    for td in test_dates:
        td_idx = list(dates_all).index(td)
        if td_idx > 0:
            prev_d = dates_all[td_idx - 1]
            prev_vals = feat.loc[
                feat["trade_date"] == prev_d, ["hh_index", "y"]
            ].set_index("hh_index")["y"]
            cur = test_df.loc[test_df["trade_date"] == td].sort_values("hh_index")
            naive_y.extend(cur["hh_index"].map(prev_vals).values)
        else:
            naive_y.extend([np.nan] * 96)

    if naive_y:
        ax2.plot(x_range, naive_y, color="#2ca02c", linewidth=1.0, linestyle=":", label="Naive(昨日同时段)", alpha=0.7)

    cum = 0
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
    naive_arr = np.array(naive_y, dtype=float)
    valid_naive = ~np.isnan(naive_arr)
    if valid_naive.any():
        naive_mae = float(mean_absolute_error(y_true_all[valid_naive], naive_arr[valid_naive]))
        title_str = f"日前价格预测 — {region_label}  |  Model MAE={overall_mae:.1f}  Naive MAE={naive_mae:.1f}"
    else:
        title_str = f"日前价格预测 — {region_label}  |  Model MAE={overall_mae:.1f}"

    ax2.set_title(title_str, fontsize=12)
    ax2.set_ylabel("价格")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    series_path = os.path.join(output_dir, f"dayahead_{region}_pred_vs_actual_series.png")
    fig2.savefig(series_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    saved.append(series_path)

    return saved


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def compute_baselines(feat: pd.DataFrame, min_train_days: int, test_days: int) -> Dict:
    """Compute naive baselines for comparison on the test portion."""
    dates = sorted(feat["trade_date"].unique())
    if len(dates) < min_train_days + test_days:
        return {}
    test_dates = dates[-test_days:]
    results = {}

    naive_records = []
    naive_week_records = []
    for td in test_dates:
        td_idx = list(dates).index(td)
        cur = feat.loc[feat["trade_date"] == td, ["hh_index", "y"]].copy()

        if td_idx > 0:
            prev_d = dates[td_idx - 1]
            prev_vals = feat.loc[
                feat["trade_date"] == prev_d, ["hh_index", "y"]
            ].set_index("hh_index")["y"]
            cur["naive_prev"] = cur["hh_index"].map(prev_vals)
            naive_records.append(cur)

        if td_idx >= 7:
            prev_w = dates[td_idx - 7]
            prev_w_vals = feat.loc[
                feat["trade_date"] == prev_w, ["hh_index", "y"]
            ].set_index("hh_index")["y"]
            cur_w = cur.copy()
            cur_w["naive_week"] = cur_w["hh_index"].map(prev_w_vals)
            naive_week_records.append(cur_w)

    if naive_records:
        ndf = pd.concat(naive_records).dropna(subset=["naive_prev"])
        if len(ndf) > 0:
            results["naive_yesterday_same_hh"] = asdict(evaluate(ndf["y"].values, ndf["naive_prev"].values))

    if naive_week_records:
        wdf = pd.concat(naive_week_records).dropna(subset=["naive_week"])
        if len(wdf) > 0:
            results["naive_lastweek_same_hh"] = asdict(evaluate(wdf["y"].values, wdf["naive_week"].values))

    return results


# ---------------------------------------------------------------------------
# Period-grouped CV
# ---------------------------------------------------------------------------

PERIOD_GROUPS_FULL = {
    "A_solar": (37, 60),    # 09:00-15:00  光伏谷段 (高波动/高地板)
    "B_evening": (61, 92),  # 15:00-23:00  傍晚峰段 (低波动/高价)
    "C_night": None,        # 23:00-09:00  夜间/早间 (剩余: hh 1-36, 93-96)
}

PERIOD_GROUPS_PARTIAL = {
    "B_evening": (61, 92),  # 15:00-23:00  傍晚峰段 (独立建模)
    "AC_rest": None,        # 其余 (hh 1-60, 93-96) 统一建模以保持样本量
}

PERIOD_GROUPS = PERIOD_GROUPS_FULL  # default; overridden by --group-partial


def _hh_mask_for_group(hh: pd.Series, group_name: str, groups: Dict) -> pd.Series:
    rng = groups[group_name]
    if rng is not None:
        return (hh >= rng[0]) & (hh <= rng[1])
    # "remainder" group = everything not covered by named ranges
    explicit = pd.Series(False, index=hh.index)
    for gn, grng in groups.items():
        if grng is not None:
            explicit |= (hh >= grng[0]) & (hh <= grng[1])
    return ~explicit


def run_grouped_cv(
    feat: pd.DataFrame,
    cv_kwargs: Dict,
    groups: Optional[Dict] = None,
    group_configs: Optional[Dict[str, Dict]] = None,
) -> Tuple[List[Dict], Optional[Dict], Optional[pd.DataFrame]]:
    """Run per-group expanding_window_cv and merge results.

    group_configs allows per-group overrides, e.g.
      {"A_solar": {"two_stage": False, "quantile_mode": True, ...}}
    """
    if groups is None:
        groups = PERIOD_GROUPS
    if group_configs is None:
        group_configs = {}

    group_folds: Dict[str, List[Dict]] = {}
    group_payloads: Dict[str, Optional[Dict]] = {}
    group_test_dfs: Dict[str, Optional[pd.DataFrame]] = {}

    for gname in groups:
        mask = _hh_mask_for_group(feat["hh_index"], gname, groups)
        g_feat = feat.loc[mask].reset_index(drop=True)
        n_hh = g_feat["hh_index"].nunique()
        n_rows = len(g_feat)
        print(f"\n  {'='*50}")
        print(f"  GROUP [{gname}]  hh_count={n_hh}  rows={n_rows}")
        print(f"  {'='*50}")

        kw = dict(cv_kwargs)

        # Scale feature_select_top_k proportionally to group size
        base_top_k = kw.get("feature_select_top_k", 0)
        if base_top_k > 0:
            ratio = n_hh / 96.0
            scaled_k = max(30, int(base_top_k * ratio))
            kw["feature_select_top_k"] = scaled_k
            print(f"  [GROUP] feature_select_top_k: {base_top_k} → {scaled_k} (ratio={ratio:.2f})")

        # More conservative LightGBM params for smaller groups
        g_params = dict(kw.get("params", {}) or {})
        g_params.setdefault("min_child_samples", 30)
        g_params["min_child_samples"] = max(g_params["min_child_samples"],
                                            int(50 * (96 / max(n_hh, 1))))
        kw["params"] = g_params

        if gname in group_configs:
            kw.update(group_configs[gname])

        folds, payload, test_df = expanding_window_cv(g_feat, **kw)
        group_folds[gname] = folds
        group_payloads[gname] = payload
        group_test_dfs[gname] = test_df

    # Merge fold-level results
    n_folds = max(len(v) for v in group_folds.values()) if group_folds else 0
    merged_folds = []
    for fi in range(n_folds):
        fold_maes = []
        fold_rmses = []
        fold_mapes = []
        fold_n = 0
        naive_maes = []
        group_detail = {}

        for gname, gfolds in group_folds.items():
            if fi >= len(gfolds):
                continue
            gf = gfolds[fi]
            g_mask = _hh_mask_for_group(feat["hh_index"], gname, groups)
            g_n = g_mask.sum() // feat["trade_date"].nunique()  # approx hh per day

            group_detail[gname] = {
                "mae": gf["test"]["mae"],
                "rmse": gf["test"]["rmse"],
            }
            fold_maes.append((gf["test"]["mae"], g_n))
            fold_rmses.append((gf["test"]["rmse"], g_n))
            fold_mapes.append((gf["test"]["mape"], g_n))
            fold_n += g_n
            if gf.get("naive_yesterday"):
                naive_maes.append((gf["naive_yesterday"]["mae"], g_n))

        total_n = sum(w for _, w in fold_maes)
        wmae = sum(v * w for v, w in fold_maes) / max(total_n, 1)
        wrmse = sum(v * w for v, w in fold_rmses) / max(total_n, 1)
        wmape = sum(v * w for v, w in fold_mapes) / max(total_n, 1)

        ref = group_folds[list(group_folds.keys())[0]][fi]
        merged = {
            "fold": fi,
            "train_days": ref.get("train_days", 0),
            "val_days": ref.get("val_days", 0),
            "test_days": ref.get("test_days", 0),
            "test_date_start": ref.get("test_date_start", ""),
            "test_date_end": ref.get("test_date_end", ""),
            "test": {"mae": wmae, "rmse": wrmse, "mape": wmape},
            "group_detail": group_detail,
        }
        if naive_maes:
            total_naive_n = sum(w for _, w in naive_maes)
            w_naive = sum(v * w for v, w in naive_maes) / max(total_naive_n, 1)
            merged["naive_yesterday"] = {"mae": w_naive, "rmse": 0, "mape": 0}
        merged_folds.append(merged)

    # Merge test_dfs
    all_test_dfs = [tdf for tdf in group_test_dfs.values() if tdf is not None]
    merged_test_df = pd.concat(all_test_dfs).sort_values(
        ["trade_date", "hh_index"]
    ).reset_index(drop=True) if all_test_dfs else None

    return merged_folds, group_payloads, merged_test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="日前价格预测: D-1 09:30 → D 日 96 时段")
    ap.add_argument("--input-parquet", required=True, help="feature_ready wide parquet")
    ap.add_argument("--region", required=True, choices=["jn", "jb"], help="jn=江南, jb=江北")
    ap.add_argument("--output-dir", required=True, help="输出目录")
    ap.add_argument("--min-train-days", type=int, default=60)
    ap.add_argument("--val-days", type=int, default=7)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--lgbm-params-json", default="", help="Optional LightGBM params override")
    ap.add_argument("--feature-select-top-k", type=int, default=0, help="If > 0, pre-train and keep only top-K features")
    ap.add_argument("--two-stage", action="store_true", help="Enable two-stage model: floor classifier + normal regressor")
    ap.add_argument("--quantile", action="store_true", help="Enable multi-quantile regression mode")
    ap.add_argument("--quantile-two-stage", action="store_true", help="Quantile regression + floor classifier hybrid")
    ap.add_argument("--time-decay", type=int, default=0, metavar="DAYS",
                    help="Half-life in days for exponential time-decay sample weighting (0=off)")
    ap.add_argument("--adaptive-naive-blend", action="store_true",
                    help="Post-processing: tune model-naive blend ratio per fold on val set")
    ap.add_argument("--dynamic-floor-value", action="store_true",
                    help="Use training-set floor price median instead of fixed 30")
    ap.add_argument("--residual-correction", action="store_true",
                    help="Post-processing: per-hh_index residual bias correction tuned on val set")
    ap.add_argument("--oracle-actual-exog", action="store_true",
                    help="Oracle experiment: include D-day actual exogenous features (not for production)")
    ap.add_argument("--group-by-period", action="store_true",
                    help="Split hh_index into 3 period groups and train separate models")
    ap.add_argument("--group-partial", action="store_true",
                    help="Partial grouping: only split B_evening out, keep rest unified")
    args = ap.parse_args()

    if sum([args.two_stage, args.quantile, args.quantile_two_stage]) > 1:
        print("ERROR: --two-stage, --quantile, --quantile-two-stage are mutually exclusive.")
        return 1

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/4] Loading data from {args.input_parquet} ...")
    df = _load_wide(args.input_parquet)
    print(f"       {len(df)} rows, {df['trade_date'].nunique()} dates")

    print(f"[2/4] Building features for region={args.region} ...")
    feat = build_features(df, args.region,
                          include_actual_exog=args.oracle_actual_exog)
    feat = feat.dropna(subset=["y"]).reset_index(drop=True)
    print(f"       Feature table: {feat.shape[0]} rows x {feat.shape[1]} cols")
    if args.oracle_actual_exog:
        actual_cols = [c for c in feat.columns if c.startswith("actual_")]
        print(f"       Oracle actual features: {len(actual_cols)} columns added")
    x_cols = [c for c in feat.columns if c not in ("trade_date", "y")]
    print(f"       Features ({len(x_cols)}): {x_cols[:10]}{'...' if len(x_cols) > 10 else ''}")
    nan_rate = feat[x_cols].isnull().mean()
    high_nan = nan_rate[nan_rate > 0.1]
    if len(high_nan) > 0:
        print(f"       ⚠ High NaN features: {dict(high_nan)}")

    params = {}
    if args.lgbm_params_json and args.lgbm_params_json.strip():
        with open(args.lgbm_params_json, encoding="utf-8") as f:
            params = json.load(f)
        print(f"       Loaded LightGBM params override: {params}")

    print(f"[3/4] Running expanding window CV (min_train={args.min_train_days}d, val={args.val_days}d, test={args.test_days}d, step={args.step_days}d) ...")
    if args.two_stage:
        print(f"       Two-stage mode: floor classifier + normal regressor")
    if args.quantile:
        print(f"       Quantile mode: multi-quantile regression ({QUANTILES})")
    if args.quantile_two_stage:
        print(f"       Quantile + two-stage hybrid mode")
    if args.time_decay > 0:
        print(f"       Time-decay sample weighting: half_life={args.time_decay} days")
    if args.adaptive_naive_blend:
        print(f"       Adaptive model-naive blending: ON")
    if args.dynamic_floor_value:
        print(f"       Dynamic floor prediction value: ON (training median)")
    if args.residual_correction:
        print(f"       Residual auto-correction: ON (per-hh bias from val set)")
    use_grouping = args.group_by_period or args.group_partial
    active_groups = (PERIOD_GROUPS_PARTIAL if args.group_partial
                     else PERIOD_GROUPS_FULL if args.group_by_period
                     else None)
    if use_grouping:
        print(f"       Period-grouped mode: {list(active_groups.keys())}")

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
    )

    if use_grouping:
        folds, last_payload, last_test_df = run_grouped_cv(
            feat, cv_kwargs, groups=active_groups)
    else:
        folds, last_payload, last_test_df = expanding_window_cv(feat, **cv_kwargs)

    if not folds:
        print("ERROR: No valid CV folds (not enough data).")
        return 1

    mean_mae = np.mean([f["test"]["mae"] for f in folds])
    mean_rmse = np.mean([f["test"]["rmse"] for f in folds])
    mean_mape = np.mean([f["test"]["mape"] for f in folds])
    naive_maes = [f["naive_yesterday"]["mae"] for f in folds if f.get("naive_yesterday")]
    mean_naive_mae = np.mean(naive_maes) if naive_maes else None

    print(f"\n{'='*60}")
    print(f"  CV Results ({len(folds)} folds)")
    print(f"  Model  MAE  = {mean_mae:.2f}")
    print(f"  Model  RMSE = {mean_rmse:.2f}")
    print(f"  Model  MAPE = {mean_mape:.4f}")
    if mean_naive_mae is not None:
        print(f"  Naive  MAE  = {mean_naive_mae:.2f}  (yesterday same hh)")
        improvement = (mean_naive_mae - mean_mae) / mean_naive_mae * 100
        print(f"  Improvement = {improvement:+.1f}%")
    if use_grouping and folds and folds[0].get("group_detail"):
        print(f"  --- Per-group MAE (averaged across folds) ---")
        for gname in active_groups:
            gmaes = [f["group_detail"][gname]["mae"]
                     for f in folds if f.get("group_detail") and gname in f["group_detail"]]
            if gmaes:
                print(f"    {gname:15s}: {np.mean(gmaes):.2f}")
    print(f"{'='*60}\n")

    print(f"[4/4] Computing baselines & saving results ...")
    baselines = compute_baselines(feat, args.min_train_days, args.test_days)

    target_col = PRICE_TARGETS[args.region]
    cv_summary: Dict = {
        "fold_count": len(folds),
        "mean_mae": round(mean_mae, 4),
        "mean_rmse": round(mean_rmse, 4),
        "mean_mape": round(mean_mape, 6),
        "mean_naive_yesterday_mae": round(mean_naive_mae, 4) if mean_naive_mae else None,
    }
    if use_grouping and folds and folds[0].get("group_detail"):
        group_maes = {}
        for gname in active_groups:
            gm = [f["group_detail"][gname]["mae"]
                  for f in folds if f.get("group_detail") and gname in f["group_detail"]]
            if gm:
                group_maes[gname] = round(float(np.mean(gm)), 4)
        cv_summary["group_mean_mae"] = group_maes

    results = {
        "region": args.region,
        "target_col": target_col,
        "data_rows": int(feat.shape[0]),
        "feature_count": len(x_cols),
        "feature_names": x_cols,
        "n_dates": int(feat["trade_date"].nunique()),
        "group_by_period": use_grouping,
        "period_groups": {k: v for k, v in active_groups.items()} if use_grouping else None,
        "cv_summary": cv_summary,
        "baselines": baselines,
        "folds": folds,
    }
    metrics_path = os.path.join(output_dir, f"dayahead_{args.region}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Metrics → {metrics_path}")

    if use_grouping and isinstance(last_payload, dict):
        for gname, gpayload in last_payload.items():
            if gpayload is None:
                continue
            gpath = os.path.join(output_dir, f"dayahead_{args.region}_{gname}_model.joblib")
            joblib.dump(gpayload, gpath)
            print(f"  Model ({gname}) → {gpath}")
    elif last_payload and not isinstance(last_payload, dict):
        model_path = os.path.join(output_dir, f"dayahead_{args.region}_model.joblib")
        joblib.dump(last_payload, model_path)
        print(f"  Model  → {model_path}")

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
        if fold.get("naive_yesterday"):
            row["naive_mae"] = fold["naive_yesterday"]["mae"]
        if fold.get("group_detail"):
            for gn, gd in fold["group_detail"].items():
                row[f"{gn}_mae"] = gd["mae"]
        fold_rows.append(row)

    summary_df = pd.DataFrame(fold_rows)
    summary_path = os.path.join(output_dir, f"dayahead_{args.region}_cv_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  Summary → {summary_path}")

    if use_grouping and isinstance(last_payload, dict):
        for gname, gpayload in last_payload.items():
            if gpayload is None:
                continue
            gcols = gpayload["feature_columns"]
            gimp = gpayload["model"].feature_importances_
            gimp_df = pd.DataFrame({"feature": gcols, "importance": gimp}).sort_values("importance", ascending=False)
            gimp_path = os.path.join(output_dir, f"dayahead_{args.region}_{gname}_feature_importance.csv")
            gimp_df.to_csv(gimp_path, index=False, encoding="utf-8-sig")
            print(f"  Feature importance ({gname}) → {gimp_path}")
            print(f"    Top-5: {', '.join(gimp_df.head(5)['feature'].tolist())}")
    elif last_payload and not isinstance(last_payload, dict):
        model_feat_cols = last_payload["feature_columns"]
        imp = last_payload["model"].feature_importances_
        imp_df = pd.DataFrame({
            "feature": model_feat_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False)
        imp_path = os.path.join(output_dir, f"dayahead_{args.region}_feature_importance.csv")
        imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
        print(f"  Feature importance → {imp_path}")
        print(f"\n  Top-10 features:")
        for _, r in imp_df.head(10).iterrows():
            print(f"    {r['feature']:50s}  {int(r['importance']):>6d}")

    if last_test_df is not None and len(last_test_df) > 0:
        print(f"\n  Plotting prediction vs actual (last fold) ...")
        plot_files = plot_pred_vs_actual(last_test_df, feat, args.region, output_dir)
        for pf in plot_files:
            print(f"  Plot → {pf}")

        # --- 分时段误差分析 ---
        tdf = last_test_df.copy()
        tdf["error"] = (tdf["pred"] - tdf["y"]).abs()
        TIME_BLOCKS = {
            "00:00-06:00 (hh 1-24)": (1, 24),
            "06:00-12:00 (hh 25-48)": (25, 48),
            "12:00-18:00 (hh 49-72)": (49, 72),
            "18:00-24:00 (hh 73-96)": (73, 96),
        }
        print(f"\n  分时段误差分析 (最后一折):")
        print(f"  {'时段':<30s}  {'MAE':>8s}  {'样本数':>6s}  {'地板价占比':>10s}")
        for label, (hh_lo, hh_hi) in TIME_BLOCKS.items():
            mask = (tdf["hh_index"] >= hh_lo) & (tdf["hh_index"] <= hh_hi)
            sub = tdf[mask]
            if len(sub) == 0:
                continue
            block_mae = sub["error"].mean()
            floor_pct = (sub["y"] <= 50).mean() * 100
            print(f"  {label:<30s}  {block_mae:>8.1f}  {len(sub):>6d}  {floor_pct:>9.1f}%")

        floor_mask = tdf["y"] <= 50
        normal_mask = tdf["y"] > 200
        if floor_mask.any():
            floor_mae = tdf.loc[floor_mask, "error"].mean()
            print(f"\n  地板价时段 (y<=50):  MAE={floor_mae:.1f}  ({floor_mask.sum()} 样本)")
        if normal_mask.any():
            normal_mae = tdf.loc[normal_mask, "error"].mean()
            print(f"  正常价时段 (y>200): MAE={normal_mae:.1f}  ({normal_mask.sum()} 样本)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
