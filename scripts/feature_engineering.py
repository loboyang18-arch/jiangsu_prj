from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Optional, Pattern, Tuple

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


def _parse_freq(freq: str) -> str:
    try:
        pd.tseries.frequencies.to_offset(freq)
        return freq
    except Exception as e:
        raise ValueError(f"非法 freq：{freq}（示例：15min / 1H / 1D）") from e


def _parse_horizon(horizon: str, freq: str) -> int:
    # horizon can be like "1" (steps) or "15min"/"1H"/"1D" (timedelta)
    h = horizon.strip()
    if re.fullmatch(r"\d+", h):
        return int(h)
    try:
        td = pd.to_timedelta(h)
    except Exception as e:
        raise ValueError(f"非法 horizon：{h}（示例：1 或 15min / 1H / 1D）") from e
    off = pd.tseries.frequencies.to_offset(freq)
    # convert offset to timedelta best-effort
    base = pd.to_timedelta(off.n, unit=off.name) if hasattr(off, "name") and off.name else pd.to_timedelta(off.delta)
    steps = int(round(td / base))
    if steps <= 0:
        raise ValueError("horizon 必须 > 0")
    return steps


def _compile_optional_regex(p: str) -> Optional[Pattern[str]]:
    p = (p or "").strip()
    if not p:
        return None
    return re.compile(p)


def _select_feature_cols(
    df: pd.DataFrame,
    include_re: Optional[Pattern[str]],
    exclude_re: Optional[Pattern[str]],
    max_missing_rate: float,
) -> List[str]:
    cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
    out: List[str] = []
    for c in cols:
        name = str(c)
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        miss = float(df[c].isna().mean())
        if miss > max_missing_rate:
            continue
        out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Feature engineering for unified time-series parquet.")
    ap.add_argument("--input-parquet", required=True, help="Input parquet from build_dataset.py")
    ap.add_argument("--output-parquet", required=True, help="Output parquet for features table")
    ap.add_argument("--output-meta", default="", help="Optional meta json output")
    ap.add_argument("--target-col", default="", help="Target column name (default: auto)")
    ap.add_argument("--freq", default="", help="统一频率（必须与 build_dataset 的 --freq 一致，例如 15min/1H/1D）")
    ap.add_argument("--horizon", default="1", help="预测步长：'1' 表示 1 步；或 '15min/1H/1D' 这种时间跨度（需要 --freq）")
    ap.add_argument("--lags", default="1,2,4,8", help="Comma-separated lags (steps) for lag features")
    ap.add_argument("--roll-windows", default="4,8,24", help="Comma-separated rolling windows (steps)")
    ap.add_argument("--include-regex", default="", help="只保留匹配该正则的特征列名（可选）")
    ap.add_argument("--exclude-regex", default="", help="排除匹配该正则的特征列名（可选）")
    ap.add_argument("--max-missing-rate", type=float, default=0.95, help="特征列缺失率上限（默认 0.95）")
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

    if args.freq:
        freq = _parse_freq(args.freq)
        df = df.set_index("timestamp").resample(freq).mean().asfreq(freq).reset_index()

    target = pick_target_column(df, args.target_col or None)

    # ensure numeric target
    if not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = pd.to_numeric(df[target], errors="coerce")

    include_re = _compile_optional_regex(args.include_regex)
    exclude_re = _compile_optional_regex(args.exclude_regex)
    feature_base_cols = _select_feature_cols(
        df,
        include_re=include_re,
        exclude_re=exclude_re,
        max_missing_rate=float(args.max_missing_rate),
    )
    if target not in feature_base_cols:
        feature_base_cols.append(target)

    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    wins = [int(x) for x in args.roll_windows.split(",") if x.strip()]

    feat = df[["timestamp"] + feature_base_cols].copy()

    if args.freq:
        horizon_steps = _parse_horizon(args.horizon, args.freq)
        label_desc = f"y = target(t+{horizon_steps} step @ {args.freq})"
    else:
        # Without a declared uniform frequency, refuse to pretend shift(-1) is a time-aware horizon.
        raise ValueError("必须提供 --freq（先把时间频率固化），才能定义严格的 horizon/lag/rolling。")

    feat["y"] = feat[target].shift(-horizon_steps)

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
            "label": label_desc,
            "freq": args.freq,
            "horizon_steps": horizon_steps,
            "lags": lags,
            "rolling_windows": wins,
            "include_regex": args.include_regex,
            "exclude_regex": args.exclude_regex,
            "max_missing_rate": float(args.max_missing_rate),
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

