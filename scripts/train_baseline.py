"""
基于特征工程输出 parquet 训练 LightGBM 基线。

能力：
- 单文件训练（train/val/test）
- 多文件（多 horizon）批量训练
- 滚动窗口回测
- 分位回归接口（为 V2 预留）
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(mean_squared_error(y_true, y_pred, squared=False)),
        mape=mape(y_true, y_pred),
    )


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def naive_last(y: np.ndarray) -> np.ndarray:
    return np.r_[np.nan, y[:-1]] if len(y) else np.array([])


def parse_feature_files(single_path: str, multi_paths: str) -> List[str]:
    out: List[str] = []
    if single_path.strip():
        out.append(os.path.abspath(single_path))
    if multi_paths.strip():
        out.extend([os.path.abspath(p.strip()) for p in multi_paths.split(",") if p.strip()])
    if not out:
        raise ValueError("至少提供 --features-parquet 或 --features-parquets")
    # 去重保序
    return list(dict.fromkeys(out))


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    y = df[target_col].astype(float).to_numpy()
    X = df.drop(columns=["timestamp", target_col])
    return X, y


def read_features(path: str, target_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"features 缺少 timestamp：{path}")
    if target_col not in df.columns:
        raise ValueError(f"features 缺少目标列 {target_col}：{path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    return df


def _merge_lgbm_params(n_estimators: int, extra: Dict | None = None) -> Dict:
    defaults = {
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    if extra:
        for k, v in extra.items():
            if k in ("random_state", "n_jobs", "objective", "alpha"):
                continue
            defaults[k] = v
    return defaults


def fit_point_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    n_estimators: int,
    lgbm_extra: Dict | None = None,
) -> LGBMRegressor:
    params = _merge_lgbm_params(n_estimators, lgbm_extra)
    model = LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_eval, y_eval)],
        eval_metric="l1",
        callbacks=[early_stopping(stopping_rounds=80), log_evaluation(period=100)],
    )
    return model


def fit_quantile_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    alphas: List[float],
    n_estimators: int,
    lgbm_extra: Dict | None = None,
) -> Dict[float, LGBMRegressor]:
    models: Dict[float, LGBMRegressor] = {}
    for alpha in alphas:
        params = _merge_lgbm_params(n_estimators, lgbm_extra)
        params["objective"] = "quantile"
        params["alpha"] = float(alpha)
        m = LGBMRegressor(**params)
        m.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric="l1",
            callbacks=[early_stopping(stopping_rounds=80), log_evaluation(period=100)],
        )
        models[alpha] = m
    return models


def infer_horizon_from_path(path: str) -> int:
    m = re.search(r"h(?:orizon)?[_-]?(\d+)", Path(path).stem)
    return int(m.group(1)) if m else -1


def single_split_train(
    df: pd.DataFrame,
    target_col: str,
    train_frac: float,
    val_frac: float,
    objective: str,
    quantile_alphas: List[float],
    n_estimators: int,
    lgbm_extra: Dict | None = None,
) -> Dict:
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    X_train, y_train = split_xy(train_df, target_col)
    X_val, y_val = split_xy(val_df, target_col)
    X_test, y_test = split_xy(test_df, target_col)

    if objective == "point":
        model = fit_point_model(X_train, y_train, X_val, y_val, n_estimators=n_estimators, lgbm_extra=lgbm_extra)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)
        out = {
            "val": asdict(evaluate(y_val, pred_val)),
            "test": asdict(evaluate(y_test, pred_test)),
            "model_payload": {"model": model, "feature_columns": list(X_train.columns), "label_column": target_col},
        }
    else:
        qmodels = fit_quantile_models(X_train, y_train, X_val, y_val, quantile_alphas, n_estimators=n_estimators, lgbm_extra=lgbm_extra)
        pred_med_val = qmodels[0.5].predict(X_val) if 0.5 in qmodels else list(qmodels.values())[0].predict(X_val)
        pred_med_test = qmodels[0.5].predict(X_test) if 0.5 in qmodels else list(qmodels.values())[0].predict(X_test)
        quantile_eval = {
            str(a): {
                "pinball_val": pinball_loss(y_val, qmodels[a].predict(X_val), a),
                "pinball_test": pinball_loss(y_test, qmodels[a].predict(X_test), a),
            }
            for a in qmodels
        }
        out = {
            "val": asdict(evaluate(y_val, pred_med_val)),
            "test": asdict(evaluate(y_test, pred_med_test)),
            "quantile_eval": quantile_eval,
            "model_payload": {"quantile_models": qmodels, "feature_columns": list(X_train.columns), "label_column": target_col},
        }

    naive_val = naive_last(y_val)
    naive_test = naive_last(y_test)
    out["val_naive_last"] = asdict(evaluate(y_val[1:], naive_val[1:])) if len(y_val) > 1 else None
    out["test_naive_last"] = asdict(evaluate(y_test[1:], naive_test[1:])) if len(y_test) > 1 else None
    out["rows"] = int(n)
    out["train_rows"] = int(len(train_df))
    out["val_rows"] = int(len(val_df))
    out["test_rows"] = int(len(test_df))
    return out


def rolling_backtest_train(
    df: pd.DataFrame,
    target_col: str,
    train_frac: float,
    val_frac: float,
    rolling_step_size: int,
    objective: str,
    quantile_alphas: List[float],
    n_estimators: int,
    lgbm_extra: Dict | None = None,
) -> Dict:
    n = len(df)
    min_train = int(n * train_frac)
    fold_size = max(16, int(n * val_frac))
    step = rolling_step_size if rolling_step_size > 0 else fold_size
    folds: List[Dict] = []
    last_payload = None

    for split_end in range(min_train, n - fold_size, step):
        train_df = df.iloc[:split_end].copy()
        test_df = df.iloc[split_end : split_end + fold_size].copy()
        # 从 train 尾部切一小段做早停验证
        val_cut = max(32, int(len(train_df) * 0.15))
        fit_df = train_df.iloc[:-val_cut].copy()
        eval_df = train_df.iloc[-val_cut:].copy()
        if len(fit_df) < 64 or len(test_df) < 8:
            continue

        X_fit, y_fit = split_xy(fit_df, target_col)
        X_eval, y_eval = split_xy(eval_df, target_col)
        X_test, y_test = split_xy(test_df, target_col)

        if objective == "point":
            model = fit_point_model(X_fit, y_fit, X_eval, y_eval, n_estimators=n_estimators, lgbm_extra=lgbm_extra)
            pred_test = model.predict(X_test)
            point_metrics = asdict(evaluate(y_test, pred_test))
            payload = {"model": model, "feature_columns": list(X_fit.columns), "label_column": target_col}
            quantile_eval = None
        else:
            qmodels = fit_quantile_models(X_fit, y_fit, X_eval, y_eval, quantile_alphas, n_estimators=n_estimators, lgbm_extra=lgbm_extra)
            pred_med_test = qmodels[0.5].predict(X_test) if 0.5 in qmodels else list(qmodels.values())[0].predict(X_test)
            point_metrics = asdict(evaluate(y_test, pred_med_test))
            payload = {"quantile_models": qmodels, "feature_columns": list(X_fit.columns), "label_column": target_col}
            quantile_eval = {str(a): pinball_loss(y_test, qmodels[a].predict(X_test), a) for a in qmodels}

        naive = naive_last(y_test)
        naive_m = asdict(evaluate(y_test[1:], naive[1:])) if len(y_test) > 1 else None
        folds.append(
            {
                "train_end_idx": int(split_end),
                "test_rows": int(len(test_df)),
                "test": point_metrics,
                "test_naive_last": naive_m,
                "quantile_pinball_test": quantile_eval,
            }
        )
        last_payload = payload

    if not folds:
        raise ValueError("rolling_backtest 无可用折，请检查样本量或回测参数。")

    mean_test = {
        "mae": float(np.mean([f["test"]["mae"] for f in folds])),
        "rmse": float(np.mean([f["test"]["rmse"] for f in folds])),
        "mape": float(np.mean([f["test"]["mape"] for f in folds])),
    }
    naive_vals = [f["test_naive_last"] for f in folds if f["test_naive_last"]]
    mean_naive = None
    if naive_vals:
        mean_naive = {
            "mae": float(np.mean([x["mae"] for x in naive_vals])),
            "rmse": float(np.mean([x["rmse"] for x in naive_vals])),
            "mape": float(np.mean([x["mape"] for x in naive_vals])),
        }

    return {
        "rows": int(n),
        "fold_count": int(len(folds)),
        "rolling_test_mean": mean_test,
        "rolling_test_naive_last_mean": mean_naive,
        "folds": folds,
        "model_payload": last_payload,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train LightGBM baseline for one or multiple feature parquets.")
    ap.add_argument("--features-parquet", default="", help="Input features parquet (single)")
    ap.add_argument("--features-parquets", default="", help="Input features parquet list, comma-separated")
    ap.add_argument("--model-dir", required=True, help="Output directory for model and metrics")
    ap.add_argument("--target-col", default="y", help="Label column name (default: y)")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--rolling-backtest", action="store_true", help="Enable rolling-window backtest")
    ap.add_argument("--rolling-step-size", type=int, default=0, help="Rolling step size rows (0 means use val-size)")
    ap.add_argument("--objective", choices=["point", "quantile"], default="point", help="Train objective")
    ap.add_argument("--quantile-alphas", default="0.1,0.5,0.9", help="Quantile alphas for quantile objective")
    ap.add_argument("--n-estimators", type=int, default=1200)
    ap.add_argument("--lgbm-params-json", default="", help="Optional JSON file with extra LightGBM params to override defaults")
    ap.add_argument("--feature-selection-top-k", type=int, default=0, help="If > 0, run a quick pre-train and keep only top-K features by importance")
    ap.add_argument("--metrics-json", default="", help="Metrics output path")
    ap.add_argument("--summary-csv", default="", help="Summary csv output path")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    features = parse_feature_files(args.features_parquet, args.features_parquets)
    alphas = [float(x) for x in args.quantile_alphas.split(",") if x.strip()]
    if args.objective == "quantile" and not alphas:
        raise ValueError("quantile objective 至少需要一个 alpha。")

    lgbm_extra: Dict | None = None
    if args.lgbm_params_json and args.lgbm_params_json.strip():
        lgbm_json_path = os.path.abspath(args.lgbm_params_json)
        with open(lgbm_json_path, encoding="utf-8") as f:
            lgbm_extra = json.load(f)
        print(f"[INFO] Loaded extra LightGBM params from {lgbm_json_path}: {lgbm_extra}")

    metrics_path = os.path.abspath(args.metrics_json) if args.metrics_json else os.path.join(model_dir, "metrics.json")
    summary_csv = os.path.abspath(args.summary_csv) if args.summary_csv else os.path.join(model_dir, "backtest_summary.csv")

    all_results: List[Dict] = []
    summary_rows: List[Dict] = []

    top_k = int(args.feature_selection_top_k)

    for fp in features:
        df = read_features(fp, args.target_col)

        if top_k > 0:
            X_all, y_all = split_xy(df.copy(), args.target_col)
            n_pretrain = int(len(X_all) * 0.6)
            X_pre = X_all.iloc[:n_pretrain]
            y_pre = y_all[:n_pretrain]
            X_pre_val = X_all.iloc[n_pretrain:int(len(X_all) * 0.75)]
            y_pre_val = y_all[n_pretrain:int(len(X_all) * 0.75)]
            pre_model = fit_point_model(X_pre, y_pre, X_pre_val, y_pre_val, n_estimators=200, lgbm_extra=lgbm_extra)
            imp = pre_model.feature_importances_
            sorted_idx = np.argsort(imp)[::-1]
            keep_cols = [X_all.columns[i] for i in sorted_idx[:top_k] if imp[sorted_idx[0]] > 0]
            if len(keep_cols) < 5:
                keep_cols = [X_all.columns[i] for i in sorted_idx[:min(top_k, len(sorted_idx))]]
            keep_set = set(keep_cols)
            drop_cols = [c for c in df.columns if c not in keep_set and c not in ("timestamp", args.target_col)]
            df = df.drop(columns=drop_cols)
            print(f"[FEATURE SELECTION] Kept {len(keep_cols)}/{len(X_all.columns)} features (top-{top_k})")

        if args.rolling_backtest:
            result = rolling_backtest_train(
                df=df,
                target_col=args.target_col,
                train_frac=float(args.train_frac),
                val_frac=float(args.val_frac),
                rolling_step_size=int(args.rolling_step_size),
                objective=args.objective,
                quantile_alphas=alphas,
                n_estimators=int(args.n_estimators),
                lgbm_extra=lgbm_extra,
            )
        else:
            result = single_split_train(
                df=df,
                target_col=args.target_col,
                train_frac=float(args.train_frac),
                val_frac=float(args.val_frac),
                objective=args.objective,
                quantile_alphas=alphas,
                n_estimators=int(args.n_estimators),
                lgbm_extra=lgbm_extra,
            )

        horizon = infer_horizon_from_path(fp)
        model_out = os.path.join(model_dir, f"baseline_lgbm_h{horizon}.joblib" if horizon > 0 else "baseline_lgbm.joblib")
        joblib.dump(result["model_payload"], model_out)
        result.pop("model_payload", None)
        result.update(
            {
                "features_parquet": fp,
                "model_path": model_out,
                "horizon": int(horizon),
                "feature_count": int(df.drop(columns=["timestamp", args.target_col]).shape[1]),
            }
        )
        all_results.append(result)

        if args.rolling_backtest:
            row = {
                "features_parquet": fp,
                "horizon": horizon,
                "mode": "rolling_backtest",
                "fold_count": result["fold_count"],
                "test_mae": result["rolling_test_mean"]["mae"],
                "test_rmse": result["rolling_test_mean"]["rmse"],
                "test_mape": result["rolling_test_mean"]["mape"],
                "naive_last_mae": (result["rolling_test_naive_last_mean"] or {}).get("mae"),
            }
        else:
            row = {
                "features_parquet": fp,
                "horizon": horizon,
                "mode": "single_split",
                "test_mae": result["test"]["mae"],
                "test_rmse": result["test"]["rmse"],
                "test_mape": result["test"]["mape"],
                "naive_last_mae": (result["test_naive_last"] or {}).get("mae"),
            }
        summary_rows.append(row)

    out = {
        "target_col": args.target_col,
        "objective": args.objective,
        "rolling_backtest": bool(args.rolling_backtest),
        "results": all_results,
    }
    with open(metrics_path, "w", encoding="utf-8") as w:
        json.dump(out, w, ensure_ascii=False, indent=2)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

