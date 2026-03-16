"""
基于特征工程输出的 parquet 训练 LightGBM 回归基线，并输出模型与验证/测试集指标。

- 输入：feature_engineering.py 生成的特征 parquet（含 timestamp、目标列及特征列）。
- 按时间顺序划分 train/val/test（默认 0.7/0.15/0.15），在 val 上早停，在 test 上报告最终指标。
- 输出：model_dir 下的 baseline_lgbm.joblib、metrics.json（含 val/test 及 naive_last 基线对比）。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import early_stopping, log_evaluation


@dataclass
class Metrics:
    """回归评估指标：MAE、RMSE、MAPE。"""

    mae: float
    rmse: float
    mape: float


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 MAPE；真实值接近 0 的样本在分母中置为 nan 后取 nanmean。"""
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)))


def time_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间顺序将 df 切分为 train / val / test 三份，比例由 train_frac、val_frac 决定，剩余为 test。"""
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """计算 MAE、RMSE、MAPE，返回 Metrics 实例。"""
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(mean_squared_error(y_true, y_pred, squared=False)),
        mape=mape(y_true, y_pred),
    )

def naive_baselines(y: np.ndarray) -> Dict[str, np.ndarray]:
    """持久化基线：y_hat(t) = y(t-1)；返回键 'naive_last'，首点为 nan，调用方需对齐索引。"""
    # caller must align properly; we keep simple here: predict last known value for each point
    if len(y) == 0:
        return {}
    return {
        "naive_last": np.r_[np.nan, y[:-1]],
    }


def main() -> int:
    """解析参数、读特征表、时间划分、训练 LightGBM、保存模型与 metrics.json，返回 0。"""
    ap = argparse.ArgumentParser(description="Train a baseline LightGBM regressor on engineered features.")
    ap.add_argument("--features-parquet", required=True, help="Input features parquet from feature_engineering.py")
    ap.add_argument("--model-dir", required=True, help="Output directory for model and metrics")
    ap.add_argument("--target-col", default="y", help="Label column name (default: y)")
    args = ap.parse_args()

    feat_path = os.path.abspath(args.features_parquet)
    model_dir = os.path.abspath(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_parquet(feat_path)
    if "timestamp" not in df.columns:
        raise ValueError("features 缺少 timestamp 列。")
    if args.target_col not in df.columns:
        raise ValueError(f"features 缺少目标列：{args.target_col}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # drop rows where y is missing
    df = df.dropna(subset=[args.target_col]).reset_index(drop=True)

    y = df[args.target_col].astype(float)
    X = df.drop(columns=["timestamp", args.target_col])

    # LightGBM can handle NaN, keep as-is.
    train_df, val_df, test_df = time_split(pd.concat([df[["timestamp", args.target_col]], X], axis=1))

    def split_xy(d: pd.DataFrame):
        y_ = d[args.target_col].astype(float).to_numpy()
        X_ = d.drop(columns=["timestamp", args.target_col])
        return X_, y_

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=50)],
    )

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    val_metrics = evaluate(y_val, pred_val)
    test_metrics = evaluate(y_test, pred_test)

    # naive baseline on val/test (align by using previous actual within each split)
    # For time-ordered splits, this is a fair minimal baseline.
    naive_val = naive_baselines(y_val).get("naive_last")
    naive_test = naive_baselines(y_test).get("naive_last")
    naive_val_metrics = evaluate(y_val[1:], naive_val[1:]) if naive_val is not None and len(y_val) > 1 else None
    naive_test_metrics = evaluate(y_test[1:], naive_test[1:]) if naive_test is not None and len(y_test) > 1 else None

    model_path = os.path.join(model_dir, "baseline_lgbm.joblib")
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X_train.columns),
            "label_column": args.target_col,
        },
        model_path,
    )

    metrics_path = os.path.join(model_dir, "metrics.json")
    out = {
        "features_parquet": feat_path,
        "model_path": model_path,
        "rows": int(df.shape[0]),
        "train_rows": int(train_df.shape[0]),
        "val_rows": int(val_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "val": asdict(val_metrics),
        "test": asdict(test_metrics),
        "val_naive_last": asdict(naive_val_metrics) if naive_val_metrics else None,
        "test_naive_last": asdict(naive_test_metrics) if naive_test_metrics else None,
        "feature_count": int(X_train.shape[1]),
    }
    with open(metrics_path, "w", encoding="utf-8") as w:
        json.dump(out, w, ensure_ascii=False, indent=2)

    print(f"Wrote model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")
    print("VAL:", out["val"])
    print("TEST:", out["test"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

