from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


TIME_COL_NAME_RE = re.compile(r"(date|time|日期|时间|交易日|运行日|统计日|查询日期)", re.IGNORECASE)


@dataclass(frozen=True)
class TimeSummary:
    time_column: str
    start: Optional[str]
    end: Optional[str]
    granularity: str
    coverage: float  # fraction of non-null parseable as datetime


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.Series([pd.NaT] * len(s))


def infer_time_summary(df: pd.DataFrame) -> Optional[TimeSummary]:
    if df is None or df.empty:
        return None

    candidates: List[str] = []
    for c in df.columns:
        name = str(c)
        if TIME_COL_NAME_RE.search(name):
            candidates.append(name)

    # fall back: try first column as potential time
    if not candidates and len(df.columns) > 0:
        candidates = [str(df.columns[0])]

    best: Optional[TimeSummary] = None
    for c in candidates:
        if c not in df.columns:
            # column names could be non-str; try match by str()
            matched = [col for col in df.columns if str(col) == c]
            if not matched:
                continue
            col = matched[0]
        else:
            col = c

        ser = df[col]
        if ser is None:
            continue
        non_null = ser.dropna()
        if non_null.empty:
            continue

        dt = _safe_to_datetime(non_null)
        ok = dt.dropna()
        if ok.empty:
            continue

        coverage = float(len(ok)) / float(len(non_null))
        start = ok.min()
        end = ok.max()

        # granularity inference from unique sorted times
        uniq = sorted(set(ok.dt.tz_localize(None)))
        gran = "单点"
        if len(uniq) >= 2:
            deltas = np.diff(np.array(uniq, dtype="datetime64[ns]")).astype("timedelta64[s]").astype(int)
            if len(deltas) > 0:
                med = float(np.median(deltas))
                if abs(med - 86400) < 2:
                    gran = "日"
                elif abs(med - 3600) < 2:
                    gran = "小时"
                elif abs(med - 1800) < 2:
                    gran = "30分钟"
                elif abs(med - 900) < 2:
                    gran = "15分钟"
                elif abs(med - 300) < 2:
                    gran = "5分钟"
                else:
                    gran = f"{int(med)}秒(近似)"

        ts = TimeSummary(
            time_column=str(col),
            start=start.isoformat() if hasattr(start, "isoformat") else str(start),
            end=end.isoformat() if hasattr(end, "isoformat") else str(end),
            granularity=gran,
            coverage=coverage,
        )

        # prefer higher coverage, then earlier candidate
        if best is None or ts.coverage > best.coverage:
            best = ts

    return best


def summarize_columns(df: pd.DataFrame, max_unique: int = 30) -> List[Dict[str, str]]:
    if df is None or df.empty:
        return []

    out: List[Dict[str, str]] = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        non_null = s.dropna()
        nn = int(non_null.shape[0])
        unique = int(non_null.nunique(dropna=True)) if nn else 0
        example = ""
        if nn:
            try:
                example = str(non_null.iloc[0])
            except Exception:
                example = ""
        top_values = ""
        if nn and unique <= max_unique:
            try:
                vc = non_null.astype(str).value_counts().head(5)
                top_values = "; ".join([f"{k}({v})" for k, v in vc.items()])
            except Exception:
                top_values = ""

        out.append(
            {
                "column": str(col),
                "dtype": dtype,
                "non_null": str(nn),
                "unique": str(unique),
                "example": example[:200],
                "top_values": top_values[:500],
            }
        )
    return out


def read_excel_preview(path: str, sheet_name: str, nrows: int = 300) -> pd.DataFrame:
    # Many files in this repo use .xls extension but are actually OOXML;
    # let pandas choose engine first, then fall back to openpyxl explicitly.
    try:
        return pd.read_excel(path, sheet_name=sheet_name, nrows=nrows)
    except Exception:
        return pd.read_excel(path, sheet_name=sheet_name, nrows=nrows, engine="openpyxl")

