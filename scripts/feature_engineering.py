from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd


PRICE_HINT_RE = re.compile(r"(电价|价格|均价|LMP|出清价)", re.IGNORECASE)


def pick_target_column(df: pd.DataFrame, target: Optional[str]) -> str:
    if target:
        if target not in df.columns:
            raise ValueError(f"target_col 不存在：{target}")
        return target

    candidates = [c for c in df.columns if c != "timestamp" and PRICE_HINT_RE.search(str(c))]
    if candidates:
        return candidates[0]

    # fallback: first numeric col
    num = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        raise ValueError("找不到可用的数值列作为目标（请在 build_dataset 输出中检查列名）。")
    return num[0]


def make_lag_features(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for lag in lags:
            out[f"{c}__lag{lag}"] = out[c].shift(lag)
    return out


def make_rolling_features(df: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = out[c]
        for w in windows:
            out[f"{c}__roll_mean{w}"] = s.rolling(w, min_periods=max(2, w // 3)).mean()
            out[f"{c}__roll_std{w}"] = s.rolling(w, min_periods=max(2, w // 3)).std()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Feature engineering for unified time-series parquet.")
    ap.add_argument("--input-parquet", required=True, help="Input parquet from build_dataset.py")
    ap.add_argument("--output-parquet", required=True, help="Output parquet for features table")
    ap.add_argument("--output-meta", default="", help="Optional meta json output")
    ap.add_argument("--target-col", default="", help="Target column name (default: auto)")
    ap.add_argument("--lags", default="1,2,4,8", help="Comma-separated lags (rows) for lag features")
    ap.add_argument("--roll-windows", default="4,8,24", help="Comma-separated rolling windows (rows)")
    ap.add_argument("--drop-na", action="store_true", help="Drop rows with any NA in X/y")
    args = ap.parse_args()

    inp = os.path.abspath(args.input_parquet)
    outp = os.path.abspath(args.output_parquet)
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    df = pd.read_parquet(inp)
    if "timestamp" not in df.columns:
        # backward compatibility: older build_dataset versions used __timestamp
        if "__timestamp" in df.columns:
            df = df.rename(columns={"__timestamp": "timestamp"})
        else:
            raise ValueError("输入数据缺少 timestamp 列。")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    target = pick_target_column(df, args.target_col or None)

    # keep only numeric predictors
    feature_base_cols = [
        c for c in df.columns if c not in {"timestamp"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    if target not in feature_base_cols:
        # if target isn't numeric due to parsing, coerce
        df[target] = pd.to_numeric(df[target], errors="coerce")
        feature_base_cols = [
            c for c in df.columns if c not in {"timestamp"} and pd.api.types.is_numeric_dtype(df[c])
        ]

    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    wins = [int(x) for x in args.roll_windows.split(",") if x.strip()]

    feat = df[["timestamp"] + feature_base_cols].copy()

    # y is next-step target by default (t+1 prediction)
    feat["y"] = feat[target].shift(-1)

    # X excludes y and the raw target at time t is allowed (for one-step ahead, it's ok)
    x_cols = feature_base_cols

    feat = make_lag_features(feat, x_cols, lags=lags)
    feat = make_rolling_features(feat, x_cols, windows=wins)

    # calendar features
    ts = feat["timestamp"]
    feat["dow"] = ts.dt.dayofweek.astype("int16")
    feat["hour"] = ts.dt.hour.astype("int16")
    feat["month"] = ts.dt.month.astype("int16")

    # keep columns order: timestamp, y, X...
    keep = ["timestamp", "y"] + [c for c in feat.columns if c not in {"timestamp", "y"}]
    feat = feat[keep]

    if args.drop_na:
        feat = feat.dropna().reset_index(drop=True)

    feat.to_parquet(outp, index=False)

    if args.output_meta:
        meta_path = os.path.abspath(args.output_meta)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        meta = {
            "input_parquet": inp,
            "output_parquet": outp,
            "target_col_t": target,
            "label": "y = target(t+1)",
            "lags": lags,
            "rolling_windows": wins,
            "rows": int(feat.shape[0]),
            "cols": int(feat.shape[1]),
        }
        with open(meta_path, "w", encoding="utf-8") as w:
            json.dump(meta, w, ensure_ascii=False, indent=2)

    print(f"Target(t): {target}")
    print(f"Wrote features: {outp} ({feat.shape[0]} rows, {feat.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

