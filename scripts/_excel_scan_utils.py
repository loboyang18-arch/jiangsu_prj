"""
Excel 扫描共用工具：时间列识别、时间范围/粒度推断、列摘要、安全读取。

供 scan_excel_dictionary.py、build_dataset.py 等脚本复用。
不依赖项目业务配置，仅提供通用的 DataFrame 分析与 Excel 读取封装。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# 用于识别“可能是时间列”的列名正则（中英文常见命名）
TIME_COL_NAME_RE = re.compile(r"(date|time|日期|时间|交易日|运行日|统计日|查询日期)", re.IGNORECASE)


@dataclass(frozen=True)
class TimeSummary:
    """单列时间解析结果：列名、起止时间、粒度、解析率与总行覆盖率。"""
    time_column: str
    start: Optional[str]
    end: Optional[str]
    granularity: str
    parse_rate_non_null: float  # 可解析数 / 非空数
    coverage_total: float       # 可解析数 / 总行数


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    """将序列转为 datetime，解析失败处为 NaT，不抛异常。"""
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.Series([pd.NaT] * len(s))


def infer_time_summary(df: pd.DataFrame) -> Optional[TimeSummary]:
    """
    从表中识别最佳时间列并返回 TimeSummary。
    候选列：列名匹配 TIME_COL_NAME_RE，或（无匹配时）首列。
    粒度根据相邻时间差中位数推断（15min/1H/1D 等）；择优依据 coverage_total、parse_rate_non_null。
    """
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
        total_rows = int(len(ser))
        non_null = ser.dropna()
        if non_null.empty:
            continue

        dt = _safe_to_datetime(non_null)
        ok = dt.dropna()
        if ok.empty:
            continue

        parse_rate_non_null = float(len(ok)) / float(len(non_null))
        coverage_total = float(len(ok)) / float(total_rows) if total_rows else 0.0
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
            parse_rate_non_null=parse_rate_non_null,
            coverage_total=coverage_total,
        )

        # prefer higher coverage, then earlier candidate
        if best is None or (ts.coverage_total, ts.parse_rate_non_null) > (best.coverage_total, best.parse_rate_non_null):
            best = ts

    return best


def summarize_columns(df: pd.DataFrame, max_unique: int = 30) -> List[Dict[str, str]]:
    """
    为每列生成摘要：dtype、non_null、unique、example、top_values（仅当唯一值数<=max_unique 时填充）。
    返回字典列表，供 data_dictionary 等使用。
    """
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
    """
    读取指定 sheet 的前 nrows 行。Engine 优先级：calamine -> pandas 默认 -> openpyxl，
    以兼容 .xls 与 .xlsx（含扩展名为 .xls 实为 OOXML 的情况）。
    """
    try:
        return pd.read_excel(path, sheet_name=sheet_name, nrows=nrows, engine="calamine")
    except Exception:
        try:
            return pd.read_excel(path, sheet_name=sheet_name, nrows=nrows)
        except Exception:
            return pd.read_excel(path, sheet_name=sheet_name, nrows=nrows, engine="openpyxl")

