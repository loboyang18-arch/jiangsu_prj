#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
江苏电力市场原始数据质量审计脚本。

对 江苏/ 与 总表/ 下的 Excel（.xls/.xlsx/.xlsm/.xlsb）进行逐 sheet 扫描，
统计时间列解析率、96 点/日完整性、列缺失率、疑似废弃列、按 dataset_group 的日期缺口等，
输出 CSV/JSON/Markdown 报告，供 rules 设计、缺失日确认和特征选列使用。

输出文件（均写入 --output-dir）：
    - raw_data_quality_summary.csv   sheet 级汇总
    - raw_data_quality_details.csv   列级明细（角色、缺失率、min/max/mean、是否疑似废弃）
    - raw_data_gap_report.csv        按 dataset_group 的日期缺口（仅针对文件名含日期的日度文件）
    - raw_data_quality_report.json   机器可读的汇总与高缺失列 Top50
    - raw_data_quality_report.md     人读版报告（总体结论、主要发现、建议）

依赖：
    - pandas, numpy（与项目一致）
    - 可选：tabulate（用于 Markdown 表格美化；未安装时自动回退为 to_string）

Usage:
    python scripts/raw_data_quality_audit.py \\
      --data-root "/path/to/jiangsu_prj" \\
      --output-dir "/path/to/jiangsu_prj/report" \\
      [--sample-files-per-group 0]

    sample-files-per-group=0 表示扫描全部匹配文件；>0 时按 dataset_group 每组只取前 N 个文件（便于快速试跑）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# =============================================================================
# 配置常量：时间列候选名、主字段关键词、废弃列模式、文件名日期正则等
# =============================================================================

TIME_COL_CANDIDATES = [
    "时间", "time", "Time", "TIME",
    "时段", "交易时段",
]
DATE_COL_CANDIDATES = [
    "查询日期", "日期", "date", "Date", "DATE", "交易日期",
]
DATETIME_COL_CANDIDATES = [
    "timestamp", "Timestamp", "datetime", "Datetime", "DATETIME",
]

UNNAMED_RE = re.compile(r"^Unnamed:\s*\d+$", re.IGNORECASE)
DATE_IN_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm", ".xlsb"}

# 主字段关键词：用于粗识别高价值列
KEYWORD_GROUPS = {
    "price": re.compile(r"(电价|价格|均价|LMP|出清价|边际电价)", re.IGNORECASE),
    "load": re.compile(r"(负荷|电力|load)", re.IGNORECASE),
    "wind": re.compile(r"(风|wind)", re.IGNORECASE),
    "solar": re.compile(r"(光|solar|pv)", re.IGNORECASE),
    "import": re.compile(r"(受电|联络线|输入|import|华东)", re.IGNORECASE),
    "storage": re.compile(r"(储能|storage)", re.IGNORECASE),
    "reserve": re.compile(r"(备用|reserve)", re.IGNORECASE),
}

# 常见疑似废弃/模板列
TRASH_COL_RE = re.compile(
    r"(^Unnamed:\s*\d+$)|(^序号$)|(^index$)|(^编号$)",
    re.IGNORECASE
)


# =============================================================================
# 工具函数：目录创建、Excel 读取（calamine 优先）、列名规范化、时间解析等
# =============================================================================

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def try_excel_file(path: Path) -> pd.ExcelFile:
    """优先 calamine（.xls/.xlsx 兼容好），失败则用 pandas 默认 engine。"""
    engines = ["calamine", None]
    last_err = None
    for eng in engines:
        try:
            return pd.ExcelFile(path, engine=eng) if eng else pd.ExcelFile(path)
        except Exception as e:
            last_err = e
    raise last_err


def try_read_excel(
    path: Path,
    sheet_name: Any,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    engines = ["calamine", None]
    last_err = None
    for eng in engines:
        try:
            return pd.read_excel(
                path,
                sheet_name=sheet_name,
                engine=eng,
                nrows=nrows,
            )
        except Exception as e:
            last_err = e
    raise last_err


def normalize_colname(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return s


def is_unnamed_col(c: str) -> bool:
    return bool(UNNAMED_RE.match(c))


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def parse_time_only(s: pd.Series) -> pd.Series:
    """
    Parse '00:15', '01:00:00' etc. Returns Timedelta series.
    """
    ss = s.astype(str).str.strip()
    td = pd.to_timedelta(ss, errors="coerce")
    return td


def infer_role(col: str) -> str:
    for role, pat in KEYWORD_GROUPS.items():
        if pat.search(col):
            return role
    if is_unnamed_col(col):
        return "unnamed"
    return "other"


def detect_date_col(df: pd.DataFrame) -> Optional[str]:
    cols = [normalize_colname(c) for c in df.columns]
    for c in cols:
        if c in DATE_COL_CANDIDATES:
            return c
    for c in cols:
        if re.search(r"(日期|date)", c, re.IGNORECASE):
            return c
    return None


def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    cols = [normalize_colname(c) for c in df.columns]
    for c in cols:
        if c in TIME_COL_CANDIDATES:
            return c
    for c in cols:
        if re.search(r"(时间|time|时段)", c, re.IGNORECASE):
            return c
    return None


def detect_datetime_col(df: pd.DataFrame) -> Optional[str]:
    cols = [normalize_colname(c) for c in df.columns]
    for c in cols:
        if c in DATETIME_COL_CANDIDATES:
            return c
    # fallbacks: columns that parse well as datetime
    best = None
    best_score = -1.0
    for c in cols:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            score = parsed.notna().mean()
            if score > best_score and score >= 0.6:
                best = c
                best_score = score
        except Exception:
            continue
    return best


def build_timestamp(df: pd.DataFrame) -> Tuple[Optional[pd.Series], str]:
    """
    从表中构建统一时间戳序列及来源类型。
    返回 (timestamp_series, source_type)。
    source_type: datetime_col | date_plus_time | time_only | none
    date_plus_time：查询日期 + 时间列拼接，适用于总表/电价日表；time_only 仅时间列时用占位日期。
    """
    dt_col = detect_datetime_col(df)
    if dt_col:
        ts = coerce_datetime(df[dt_col])
        if ts.notna().mean() >= 0.6:
            return ts, "datetime_col"

    date_col = detect_date_col(df)
    time_col = detect_time_col(df)

    if date_col and time_col:
        date_part = coerce_datetime(df[date_col])
        time_part = parse_time_only(df[time_col])
        ts = date_part + time_part
        return ts, "date_plus_time"

    if time_col:
        # time-only sheet: cannot create full timestamp without context
        # keep as NaT here; caller may infer daily structure from row count
        td = parse_time_only(df[time_col])
        anchor = pd.Timestamp("2000-01-01")
        ts = anchor + td
        return ts, "time_only"

    return None, "none"


def get_file_date_from_name(path: Path) -> Optional[pd.Timestamp]:
    m = DATE_IN_FILENAME_RE.search(path.name)
    if not m:
        return None
    try:
        return pd.Timestamp(m.group(1))
    except Exception:
        return None


def classify_path(rel: str) -> Dict[str, str]:
    """按相对路径推断 phase(事前/事后)、family(总表/电价数据)、region(江南/江北)、dataset_group(用于缺口统计)。"""
    rel_norm = rel.replace("\\", "/")
    info = {
        "phase": "other",
        "family": "other",
        "region": "all",
        "dataset_group": "other",
    }

    if "/总表/事前/" in rel_norm or rel_norm.startswith("总表/事前/"):
        info["phase"] = "事前"
    elif "/总表/事后/" in rel_norm or rel_norm.startswith("总表/事后/"):
        info["phase"] = "事后"

    if "江北" in rel_norm:
        info["region"] = "江北"
    elif "江南" in rel_norm:
        info["region"] = "江南"

    if "电价数据" in rel_norm:
        info["family"] = "电价数据"
    elif "总表" in rel_norm:
        info["family"] = "总表"

    # dataset_group for gap analysis
    name = Path(rel_norm).name
    parent = Path(rel_norm).parent.name
    stem = Path(rel_norm).stem

    patterns = [
        ("实时出清结果统合", "实时出清结果统合"),
        ("日前出清结果统合", "日前出清结果统合"),
        ("实时市场加权均价", "实时市场加权均价"),
        ("实际受电情况-华东", "实际受电情况-华东"),
        ("实际系统负荷", "实际系统负荷"),
        ("实际燃机固定出力总值-汇总", "实际燃机固定出力总值-汇总"),
        ("储能发电计划实际数据-汇总", "储能发电计划实际数据-汇总"),
        ("煤电发电计划实际数据-汇总", "煤电发电计划实际数据-汇总"),
    ]
    for k, v in patterns:
        if k in stem:
            info["dataset_group"] = v
            break
    else:
        info["dataset_group"] = parent or "other"

    return info


def detect_nominal_freq(ts: pd.Series) -> Optional[str]:
    s = ts.dropna().sort_values()
    if len(s) < 3:
        return None
    diffs = s.diff().dropna()
    if diffs.empty:
        return None
    mode = diffs.mode()
    if mode.empty:
        return None
    d = mode.iloc[0]
    minutes = d / pd.Timedelta(minutes=1)
    if abs(minutes - 15) < 1e-6:
        return "15min"
    if abs(minutes - 60) < 1e-6:
        return "1H"
    if abs(minutes - 1440) < 1e-6:
        return "1D"
    return str(d)


def daily_slot_quality(ts: pd.Series, expected_slots: int = 96) -> Dict[str, Any]:
    """按天统计时间点数量，与 expected_slots（默认 96，即 15min 一日）对比，得到完整日数、异常日数等。"""
    s = ts.dropna().sort_values()
    if s.empty:
        return {
            "unique_days": 0,
            "expected_slots": expected_slots,
            "full_days": 0,
            "abnormal_days": 0,
            "daily_min_slots": None,
            "daily_max_slots": None,
        }
    days = s.dt.floor("D")
    cnt = days.value_counts().sort_index()
    full_days = int((cnt == expected_slots).sum())
    abnormal_days = int((cnt != expected_slots).sum())
    return {
        "unique_days": int(cnt.shape[0]),
        "expected_slots": expected_slots,
        "full_days": full_days,
        "abnormal_days": abnormal_days,
        "daily_min_slots": int(cnt.min()),
        "daily_max_slots": int(cnt.max()),
    }


def series_missing_rate(s: pd.Series) -> float:
    return float(s.isna().mean())


def numeric_summary(s: pd.Series) -> Dict[str, Any]:
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().sum() == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(sn.min()),
        "max": float(sn.max()),
        "mean": float(sn.mean()),
    }


@dataclass
class SheetAudit:
    """单 sheet 审计结果：路径、时间来源、解析率、96 点完整日、各类列计数、时间范围等。"""
    rel_path: str
    sheet_name: str
    phase: str
    family: str
    region: str
    dataset_group: str
    rows: int
    cols: int
    timestamp_source: str
    datetime_col: Optional[str]
    date_col: Optional[str]
    time_col: Optional[str]
    timestamp_parse_rate: float
    timestamp_dup_count: int
    nominal_freq: Optional[str]
    unique_days: int
    full_days_96: int
    abnormal_days_96: int
    daily_min_slots: Optional[int]
    daily_max_slots: Optional[int]
    unnamed_col_count: int
    all_nan_col_count: int
    high_missing_col_count_50: int
    high_missing_col_count_90: int
    likely_deprecated_col_count: int
    key_price_cols: int
    key_load_cols: int
    key_wind_cols: int
    key_solar_cols: int
    key_import_cols: int
    key_storage_cols: int
    key_reserve_cols: int
    file_date: Optional[str]
    min_timestamp: Optional[str]
    max_timestamp: Optional[str]
    error: Optional[str]


# =============================================================================
# 核心审计：单 sheet 审计、缺口报告、Markdown 报告生成（含 to_markdown 安全回退）
# =============================================================================

def _df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """
    将 DataFrame 转为 Markdown 表格。若 pandas 的 to_markdown 不可用（缺 tabulate），
    则回退为 to_string()，避免脚本因输出格式依赖而失败。
    """
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def audit_sheet(path: Path, rel_path: str, sheet_name: Any) -> Tuple[SheetAudit, pd.DataFrame]:
    dims = classify_path(rel_path)
    try:
        df = try_read_excel(path, sheet_name=sheet_name, nrows=None)
        df.columns = [normalize_colname(c) for c in df.columns]
    except Exception as e:
        audit = SheetAudit(
            rel_path=rel_path,
            sheet_name=str(sheet_name),
            phase=dims["phase"],
            family=dims["family"],
            region=dims["region"],
            dataset_group=dims["dataset_group"],
            rows=0,
            cols=0,
            timestamp_source="none",
            datetime_col=None,
            date_col=None,
            time_col=None,
            timestamp_parse_rate=0.0,
            timestamp_dup_count=0,
            nominal_freq=None,
            unique_days=0,
            full_days_96=0,
            abnormal_days_96=0,
            daily_min_slots=None,
            daily_max_slots=None,
            unnamed_col_count=0,
            all_nan_col_count=0,
            high_missing_col_count_50=0,
            high_missing_col_count_90=0,
            likely_deprecated_col_count=0,
            key_price_cols=0,
            key_load_cols=0,
            key_wind_cols=0,
            key_solar_cols=0,
            key_import_cols=0,
            key_storage_cols=0,
            key_reserve_cols=0,
            file_date=str(get_file_date_from_name(path).date()) if get_file_date_from_name(path) is not None else None,
            min_timestamp=None,
            max_timestamp=None,
            error=f"{type(e).__name__}: {e}",
        )
        return audit, pd.DataFrame()

    rows, cols = df.shape
    datetime_col = detect_datetime_col(df)
    date_col = detect_date_col(df)
    time_col = detect_time_col(df)

    ts, ts_source = build_timestamp(df)
    parse_rate = 0.0
    dup_count = 0
    nominal_freq = None
    min_ts = None
    max_ts = None
    q = {
        "unique_days": 0,
        "full_days": 0,
        "abnormal_days": 0,
        "daily_min_slots": None,
        "daily_max_slots": None,
    }

    if ts is not None:
        parse_rate = float(ts.notna().mean())
        valid = ts.dropna().sort_values()
        dup_count = int(valid.duplicated().sum())
        nominal_freq = detect_nominal_freq(valid)
        if not valid.empty:
            min_ts = str(valid.min())
            max_ts = str(valid.max())
        # only date_plus_time / datetime_col should be judged by 96/day
        if ts_source in ("date_plus_time", "datetime_col"):
            q = daily_slot_quality(valid, expected_slots=96)

    unnamed_col_count = sum(is_unnamed_col(c) for c in df.columns)
    missing_rates = df.isna().mean()
    all_nan_col_count = int((missing_rates == 1.0).sum())
    high_missing_50 = int((missing_rates >= 0.5).sum())
    high_missing_90 = int((missing_rates >= 0.9).sum())

    likely_deprecated = 0
    key_counts = defaultdict(int)

    for c in df.columns:
        role = infer_role(c)
        if TRASH_COL_RE.search(c):
            likely_deprecated += 1
        if role in KEYWORD_GROUPS:
            key_counts[role] += 1
        # 高缺失数值列也记作疑似废弃
        if missing_rates.get(c, 0.0) >= 0.95:
            likely_deprecated += 1

    audit = SheetAudit(
        rel_path=rel_path,
        sheet_name=str(sheet_name),
        phase=dims["phase"],
        family=dims["family"],
        region=dims["region"],
        dataset_group=dims["dataset_group"],
        rows=int(rows),
        cols=int(cols),
        timestamp_source=ts_source,
        datetime_col=datetime_col,
        date_col=date_col,
        time_col=time_col,
        timestamp_parse_rate=parse_rate,
        timestamp_dup_count=dup_count,
        nominal_freq=nominal_freq,
        unique_days=int(q["unique_days"]),
        full_days_96=int(q["full_days"]),
        abnormal_days_96=int(q["abnormal_days"]),
        daily_min_slots=q["daily_min_slots"],
        daily_max_slots=q["daily_max_slots"],
        unnamed_col_count=int(unnamed_col_count),
        all_nan_col_count=int(all_nan_col_count),
        high_missing_col_count_50=int(high_missing_50),
        high_missing_col_count_90=int(high_missing_90),
        likely_deprecated_col_count=int(likely_deprecated),
        key_price_cols=int(key_counts["price"]),
        key_load_cols=int(key_counts["load"]),
        key_wind_cols=int(key_counts["wind"]),
        key_solar_cols=int(key_counts["solar"]),
        key_import_cols=int(key_counts["import"]),
        key_storage_cols=int(key_counts["storage"]),
        key_reserve_cols=int(key_counts["reserve"]),
        file_date=str(get_file_date_from_name(path).date()) if get_file_date_from_name(path) is not None else None,
        min_timestamp=min_ts,
        max_timestamp=max_ts,
        error=None,
    )

    detail_rows = []
    for c in df.columns:
        role = infer_role(c)
        mr = float(missing_rates[c])
        entry = {
            "rel_path": rel_path,
            "sheet_name": str(sheet_name),
            "column_name": c,
            "dtype": str(df[c].dtype),
            "role_guess": role,
            "missing_rate": mr,
            "non_null_count": int(df[c].notna().sum()),
            "unique_count": int(df[c].nunique(dropna=True)),
            "all_nan": bool(mr == 1.0),
            "is_unnamed": bool(is_unnamed_col(c)),
            "likely_deprecated": bool(TRASH_COL_RE.search(c) or mr >= 0.95),
        }
        if is_numeric_series(df[c]):
            entry.update(numeric_summary(df[c]))
        else:
            entry.update({"min": None, "max": None, "mean": None})
        detail_rows.append(entry)

    detail_df = pd.DataFrame(detail_rows)
    return audit, detail_df


def build_gap_report(file_index_df: pd.DataFrame) -> pd.DataFrame:
    """按 dataset_group 聚合，根据文件名中的日期统计缺失日历日。仅对文件名含 YYYY-MM-DD 的日度文件有效。"""
    rows = []
    if file_index_df.empty:
        return pd.DataFrame()

    for group, g in file_index_df.groupby("dataset_group"):
        g = g.copy()
        g["file_date"] = pd.to_datetime(g["file_date"], errors="coerce")
        g = g[g["file_date"].notna()].sort_values("file_date")
        if g.empty:
            continue

        start = g["file_date"].min()
        end = g["file_date"].max()
        full = pd.date_range(start, end, freq="D")
        have = set(g["file_date"].dt.floor("D"))
        missing = [d for d in full if d not in have]

        rows.append({
            "dataset_group": group,
            "file_count": int(g.shape[0]),
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "expected_days": int(len(full)),
            "missing_days": int(len(missing)),
            "missing_dates_preview": ", ".join(str(d.date()) for d in missing[:20]),
        })

    return pd.DataFrame(rows).sort_values(["missing_days", "dataset_group"], ascending=[False, True])


def load_excel_inventory(data_root: Path) -> List[Path]:
    """递归列出 data_root 下所有 .xls/.xlsx/.xlsm/.xlsb 文件（不含 .git）。"""
    files = []
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXCEL_SUFFIXES:
            files.append(p)
    return sorted(files)


def sample_group_files(paths: List[Path], data_root: Path, sample_files_per_group: int) -> List[Path]:
    """当 sample_files_per_group>0 时，按 dataset_group 每组只保留前 N 个文件，用于快速试跑。"""
    if sample_files_per_group <= 0:
        return paths

    grouped = defaultdict(list)
    for p in paths:
        rel = str(p.relative_to(data_root)).replace("\\", "/")
        info = classify_path(rel)
        grouped[info["dataset_group"]].append(p)

    selected = []
    for _, group_paths in grouped.items():
        group_paths = sorted(group_paths)
        selected.extend(group_paths[:sample_files_per_group])

    return sorted(selected)


def build_markdown_report(
    summary_df: pd.DataFrame,
    details_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    out_json: Dict[str, Any],
) -> str:
    lines: List[str] = []

    lines.append("# 江苏原始数据质量报告")
    lines.append("")

    overall = out_json["overall"]
    lines.append("## 一、总体结论")
    lines.append("")
    lines.append(f"- Excel 文件总数：**{overall['excel_file_count']}**")
    lines.append(f"- 扫描 sheet 总数：**{overall['sheet_count']}**")
    lines.append(f"- 可解析时间戳的 sheet 数：**{overall['timestamp_sheet_count']}**")
    lines.append(f"- 15min 频率 sheet 数：**{overall['freq_15min_sheet_count']}**")
    lines.append(f"- 含 96 点完整日结构的 sheet 数：**{overall['full_96_sheet_count']}**")
    lines.append(f"- 含错误的 sheet 数：**{overall['error_sheet_count']}**")
    lines.append("")

    lines.append("## 二、主要发现")
    lines.append("")

    if not gap_df.empty:
        top_gap = gap_df.sort_values("missing_days", ascending=False).head(10)
        for _, r in top_gap.iterrows():
            if int(r["missing_days"]) > 0:
                lines.append(
                    f"- `{r['dataset_group']}`：文件覆盖 {r['start_date']} ~ {r['end_date']}，"
                    f"理论 {r['expected_days']} 天，缺 **{r['missing_days']}** 天。"
                )
    else:
        lines.append("- 未生成文件级日期缺口报告。")

    trash_cols = details_df[details_df["likely_deprecated"] == True]
    if not trash_cols.empty:
        lines.append(
            f"- 疑似模板/废弃列共 **{trash_cols.shape[0]}** 个字段实例，"
            f"建议在 rules 中排除 `Unnamed:*` 与高缺失旧列。"
        )

    bad_sheet = summary_df[
        (summary_df["timestamp_dup_count"] > 0) |
        (summary_df["abnormal_days_96"] > 0) |
        (summary_df["high_missing_col_count_90"] > 0)
    ]
    if not bad_sheet.empty:
        lines.append(
            f"- 发现 **{bad_sheet.shape[0]}** 个 sheet 存在时间重复、96 点不完整或高缺失列偏多的现象，需要进一步规则化清洗。"
        )

    lines.append("")
    lines.append("## 三、按 sheet 汇总（前 20 行）")
    lines.append("")
    if not summary_df.empty:
        show = summary_df.head(20).copy()
        lines.append(_df_to_markdown(show, index=False))
    else:
        lines.append("无数据。")

    lines.append("")
    lines.append("## 四、文件级日期缺口")
    lines.append("")
    if not gap_df.empty:
        lines.append(_df_to_markdown(gap_df, index=False))
    else:
        lines.append("无可统计日期缺口的数据集。（仅对文件名含 YYYY-MM-DD 的日度文件统计）")

    lines.append("")
    lines.append("## 五、建议")
    lines.append("")
    lines.append("- 对 `Unnamed:*`、高缺失旧列、模板列在 `dataset_rules.json` 中统一排除。")
    lines.append("- 对价格日表、汇总总表分别建立规则，不要混用时间字段拼接逻辑。")
    lines.append("- 对 2025 年价格日表的缺失月份做专项确认：确认是源数据缺失还是未入库。")
    lines.append("- 对多节点日前价格表先做区域聚合/价差特征，不建议直接全节点展开入模。")
    lines.append("- 把 `actual / forecast / plan / public` 的可用性标签加入 feature registry，避免泄露。")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# 入口：解析参数、扫描 Excel、汇总/明细/缺口、写 CSV/JSON/MD
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="江苏电力市场原始数据质量审计（只读）。输出 summary/details/gap CSV 与 JSON/MD 报告。"
    )
    ap.add_argument("--data-root", required=True, help="Project root containing 江苏/ and 总表/")
    ap.add_argument("--output-dir", required=True, help="Directory to save audit outputs")
    ap.add_argument(
        "--sample-files-per-group",
        type=int,
        default=0,
        help="0 = scan all files; >0 = sample first N files per dataset_group",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    safe_mkdir(output_dir)

    excel_files = load_excel_inventory(data_root)
    excel_files = sample_group_files(excel_files, data_root, args.sample_files_per_group)

    summary_rows = []
    detail_dfs = []
    file_index_rows = []

    for path in excel_files:
        rel = str(path.relative_to(data_root)).replace("\\", "/")
        info = classify_path(rel)
        file_date = get_file_date_from_name(path)
        file_index_rows.append({
            "rel_path": rel,
            "dataset_group": info["dataset_group"],
            "phase": info["phase"],
            "family": info["family"],
            "region": info["region"],
            "file_date": str(file_date.date()) if file_date is not None else None,
        })

        try:
            xf = try_excel_file(path)
            sheet_names = xf.sheet_names
        except Exception as e:
            summary_rows.append(asdict(SheetAudit(
                rel_path=rel,
                sheet_name="__workbook__",
                phase=info["phase"],
                family=info["family"],
                region=info["region"],
                dataset_group=info["dataset_group"],
                rows=0,
                cols=0,
                timestamp_source="none",
                datetime_col=None,
                date_col=None,
                time_col=None,
                timestamp_parse_rate=0.0,
                timestamp_dup_count=0,
                nominal_freq=None,
                unique_days=0,
                full_days_96=0,
                abnormal_days_96=0,
                daily_min_slots=None,
                daily_max_slots=None,
                unnamed_col_count=0,
                all_nan_col_count=0,
                high_missing_col_count_50=0,
                high_missing_col_count_90=0,
                likely_deprecated_col_count=0,
                key_price_cols=0,
                key_load_cols=0,
                key_wind_cols=0,
                key_solar_cols=0,
                key_import_cols=0,
                key_storage_cols=0,
                key_reserve_cols=0,
                file_date=str(file_date.date()) if file_date is not None else None,
                min_timestamp=None,
                max_timestamp=None,
                error=f"{type(e).__name__}: {e}",
            )))
            continue

        for sh in sheet_names:
            audit, detail_df = audit_sheet(path, rel, sh)
            summary_rows.append(asdict(audit))
            if not detail_df.empty:
                detail_dfs.append(detail_df)

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.concat(detail_dfs, ignore_index=True) if detail_dfs else pd.DataFrame()
    file_index_df = pd.DataFrame(file_index_rows)
    gap_df = build_gap_report(file_index_df)

    overall = {
        "excel_file_count": int(len(excel_files)),
        "sheet_count": int(summary_df.shape[0]),
        "timestamp_sheet_count": int((summary_df["timestamp_parse_rate"] > 0.6).sum()) if not summary_df.empty else 0,
        "freq_15min_sheet_count": int((summary_df["nominal_freq"] == "15min").sum()) if not summary_df.empty else 0,
        "full_96_sheet_count": int((summary_df["full_days_96"] > 0).sum()) if not summary_df.empty else 0,
        "error_sheet_count": int(summary_df["error"].notna().sum()) if not summary_df.empty else 0,
    }

    json_obj = {
        "overall": overall,
        "top_gap_groups": gap_df.to_dict(orient="records") if not gap_df.empty else [],
        "high_missing_columns_top50": (
            details_df.sort_values("missing_rate", ascending=False)
            .head(50)
            .to_dict(orient="records")
            if not details_df.empty else []
        ),
        "summary_preview": summary_df.head(100).to_dict(orient="records") if not summary_df.empty else [],
    }

    # save
    summary_path = output_dir / "raw_data_quality_summary.csv"
    details_path = output_dir / "raw_data_quality_details.csv"
    gap_path = output_dir / "raw_data_gap_report.csv"
    json_path = output_dir / "raw_data_quality_report.json"
    md_path = output_dir / "raw_data_quality_report.md"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    gap_df.to_csv(gap_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=2)

    md = build_markdown_report(summary_df, details_df, gap_df, json_obj)
    md_path.write_text(md, encoding="utf-8")

    print(f"[OK] summary -> {summary_path}")
    print(f"[OK] details -> {details_path}")
    print(f"[OK] gap report -> {gap_path}")
    print(f"[OK] json -> {json_path}")
    print(f"[OK] markdown -> {md_path}")


if __name__ == "__main__":
    main()