#!/usr/bin/env python3
"""导出 V0 全量缺失统计（支持连续零值视作缺失）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

KEY_COLS = {"timestamp", "trade_date", "hh_index"}


def _long_zero_mask(series: pd.Series, threshold: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    is_zero = s.eq(0) & s.notna()
    groups = is_zero.ne(is_zero.shift(fill_value=False)).cumsum()
    run_len = is_zero.groupby(groups).transform("sum")
    return is_zero & (run_len >= threshold)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    qflag_cols = [c for c in df.columns if str(c).endswith("_qflag")]
    return [c for c in df.columns if c not in KEY_COLS and c not in qflag_cols]


def export_base_missing(df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    total = len(df)
    rows: list[dict] = []
    for c in _feature_cols(df):
        s = pd.to_numeric(df[c], errors="coerce")
        miss = int(s.isna().sum())
        rows.append(
            {
                "feature": c,
                "missing_count": miss,
                "missing_ratio": (miss / total) if total else 0.0,
                "non_null_count": int(total - miss),
            }
        )
    out = (
        pd.DataFrame(rows)
        .sort_values(["missing_ratio", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out


def export_zero_run_missing(
    df: pd.DataFrame, out_prefix: Path, thresholds: Iterable[int]
) -> list[Path]:
    total = len(df)
    outputs: list[Path] = []
    for threshold in thresholds:
        rows: list[dict] = []
        for c in _feature_cols(df):
            s = pd.to_numeric(df[c], errors="coerce")
            nan_mask = s.isna()
            zero_long_mask = _long_zero_mask(s, int(threshold))
            adjusted_mask = nan_mask | zero_long_mask
            rows.append(
                {
                    "feature": c,
                    "base_missing_count": int(nan_mask.sum()),
                    f"zero_run_missing_count_ge_{threshold}": int(
                        (~nan_mask & zero_long_mask).sum()
                    ),
                    "adjusted_missing_count": int(adjusted_mask.sum()),
                    "adjusted_missing_ratio": (adjusted_mask.mean()) if total else 0.0,
                    "adjusted_non_null_count": int(total - adjusted_mask.sum()),
                }
            )

        out = (
            pd.DataFrame(rows)
            .sort_values(["adjusted_missing_ratio", "feature"], ascending=[False, True])
            .reset_index(drop=True)
        )
        out_csv = Path(f"{out_prefix}_ge{threshold}.csv")
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        outputs.append(out_csv)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 V0 缺失统计（含连续零值规则）")
    parser.add_argument("--plan", required=True, help="warehouse_plan.json 路径")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    outputs = plan["outputs"]
    v0 = plan["sample_windows"]["v0"]

    parquet_path = Path(outputs["feature_ready_v0_parquet"])
    base_out = Path(outputs["feature_missing_full_v0_csv"])
    zero_prefix = Path(outputs["feature_missing_zero_run_v0_csv_prefix"])
    thresholds = v0.get("zero_run_missing_thresholds", [96])

    df = pd.read_parquet(parquet_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    base = export_base_missing(df, base_out)
    zero_files = export_zero_run_missing(df, zero_prefix, thresholds)

    print(f"[OK] base missing csv: {base_out}")
    print(f"[INFO] features: {len(base)}")
    print(base.head(10).to_string(index=False))
    for p in zero_files:
        print(f"[OK] zero-run csv: {p}")


if __name__ == "__main__":
    main()

