"""
从异构 Excel 构建统一时间序列宽表（parquet）及元数据（JSON）。

- 支持 full/preview 模式（全量读取或按行数/ sheet 数抽样）。
- 可选 --freq 将时间轴统一为 15min/1H/1D 并重采样。
- 可选 --rules 指定 JSON 规则文件，按路径/sheet 匹配聚合方式（mean/sum/last/weighted_avg）与 max_metrics_per_sheet。
- 列排除：与 raw_data_quality_audit 结论一致，抽取时始终排除 Unnamed、序号、index、编号等模板列；
  规则中的 exclude_columns_regex 可在此基础上再排除（如节点、ID）。详见 scripts/README.md「数据质量审计启示与处理约定」。
- 无时间列时：若 sheet 无可用时间列但文件名含日期（如 2025-06-01），则按“数据按时间排序”的假设，用文件名日期 + 行数推断时间轴（1 行→当日 0 点，24→整点，96→15min×96 等），仍可抽取价格等数值列；meta 中 time_source 为 "filename" 便于追溯。
- 输出：单表 parquet（timestamp + 多指标列）、meta JSON（含 metric_meta 每列来源与 phase/region/family、time_source）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from _excel_scan_utils import TIME_COL_NAME_RE, infer_time_summary, read_excel_preview


# 启发式识别“价格列”与“权重列”（用于加权平均），无规则时使用
PRICE_COL_RE = re.compile(r"(电价|价格|均价|LMP|出清价)", re.IGNORECASE)
WEIGHT_COL_RE = re.compile(r"(电量|成交电量|MW|MWh|成交量)", re.IGNORECASE)

# 默认排除列（与 raw_data_quality_audit.TRASH_COL_RE 一致）：抽取时始终过滤，减少噪声与废弃列入模
DEFAULT_EXCLUDE_COL_RE = re.compile(
    r"(^Unnamed:\s*\d+$)|(^序号$)|(^index$)|(^编号$)",
    re.IGNORECASE,
)

# 文件名中的日期，用于无时间列时按“同类文件”思路推断时间轴（参考质量审计 DATE_IN_FILENAME_RE）
DATE_IN_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


# -----------------------------------------------------------------------------
# 文件枚举、sheet 列表、列名规范化、路径分类与维度解析
# -----------------------------------------------------------------------------

def iter_excel_files(data_root: str) -> List[str]:
    """递归列出 data_root 下所有 .xls/.xlsx 路径，跳过 .git。"""
    out: List[str] = []
    for dp, dn, fn in os.walk(data_root):
        if os.path.basename(dp) == ".git" or "/.git/" in dp:
            dn[:] = []
            continue
        for f in fn:
            # Excel 临时锁文件（~$*.xlsx）会导致读取报错，直接跳过
            if f.startswith("~$"):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in {".xls", ".xlsx"}:
                out.append(os.path.join(dp, f))
    return out


def list_sheets_quick(path: str) -> List[str]:
    """获取工作簿 sheet 列表，优先 calamine 以兼容 .xls/.xlsx。"""
    try:
        xls = pd.ExcelFile(path, engine="calamine")
        return list(xls.sheet_names)
    except Exception:
        try:
            xls = pd.ExcelFile(path)
            return list(xls.sheet_names)
        except Exception:
            try:
                import openpyxl

                wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                return list(wb.sheetnames)
            except Exception:
                return []


def normalize_metric_name(parts: List[str]) -> str:
    """将名称片段拼接为合法指标名（无空格、特殊字符转下划线，长度截断 200）。"""
    s = "__".join([p.strip().replace(" ", "") for p in parts if p and str(p).strip()])
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", s)
    s = re.sub(r"__+", "__", s).strip("_")
    return s[:200]


def classify_from_path(rel_path: str) -> str:
    """按相对路径返回粗分类标签（江苏电价数据/总表_事前/总表_事后/其他），用于指标命名前缀。"""
    if "江苏/电价数据" in rel_path:
        return "江苏电价数据"
    if "总表/事前" in rel_path:
        return "总表_事前"
    if "总表/事后" in rel_path:
        return "总表_事后"
    return "其他"

def parse_dimensions(rel_path: str) -> Dict[str, str]:
    """从相对路径解析 phase(事前/事后)、region(江南/江北/汇总/华东)、family(出清/电价/负荷/受电/储能等)。

    说明：
    - 早期版本仅在路径包含 \"总表/事前\" / \"总表/事后\" 时设置 phase，导致以 \"总表\" 作为 data_root
      抽取时 meta 中 phase 大多为空（因为 source_file 仅为 \"事前/...\" 或 \"事后/...\"）。
    - 现放宽判断条件：凡相对路径中包含 \"事前/\" 视为事前，包含 \"事后/\" 视为事后，
      以便 total_wide_15m.meta.json 中的 phase 能正确反映总表事前/事后口径。
    """
    d: Dict[str, str] = {}
    if "事前/" in rel_path:
        d["phase"] = "事前"
    elif "事后/" in rel_path:
        d["phase"] = "事后"
    else:
        d["phase"] = ""
    if "江北" in rel_path:
        d["region"] = "江北"
    elif "江南" in rel_path:
        d["region"] = "江南"
    elif "汇总" in rel_path:
        d["region"] = "汇总"
    elif "华东" in rel_path:
        d["region"] = "华东"
    else:
        d["region"] = ""

    # coarse data family
    if "出清" in rel_path:
        d["family"] = "出清"
    elif "电价" in rel_path or "价格" in rel_path:
        d["family"] = "电价"
    elif "负荷" in rel_path:
        d["family"] = "负荷"
    elif "受电" in rel_path:
        d["family"] = "受电"
    elif "储能" in rel_path:
        d["family"] = "储能"
    elif "煤电" in rel_path:
        d["family"] = "煤电"
    elif "燃机" in rel_path:
        d["family"] = "燃机"
    elif "风力" in rel_path or "风电" in rel_path:
        d["family"] = "风电"
    elif "光伏" in rel_path or "太阳能" in rel_path:
        d["family"] = "光伏"
    else:
        d["family"] = ""

    return d


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """将 object 列尝试转为数值（去除千分位逗号后 to_numeric），失败保留原样。"""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            parsed = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="coerce")
            # 仅在“多数可解析为数值”时才转型，避免把纯文本维度列错误转成全 NaN 数值列
            if parsed.notna().mean() >= 0.5:
                out[c] = parsed
    return out


def infer_time_column_from_filename(rel_path: str, df: pd.DataFrame) -> Optional[str]:
    """
    当 sheet 无可用时间列时，用文件名中的日期 + 行数推断时间轴（数据视为按时间排序）。
    在 df 上新增列 __filename_time 并返回该列名；无法推断时返回 None。
    行数约定：1→当日 00:00；24→整点 0..23；96→15min×96（首点 00:15，与有时间列文件一致）；48→30min×48；288→5min×288；其它→15min 步长、首点 00:15 共 n 点。
    """
    m = DATE_IN_FILENAME_RE.search(rel_path)
    if not m:
        return None
    base_date = m.group(1)
    n = len(df)
    if n == 0:
        return None
    base = pd.Timestamp(f"{base_date} 00:00:00")
    if n == 1:
        timestamps = [base]
    elif n == 24:
        timestamps = [base + pd.Timedelta(hours=i) for i in range(24)]
    elif n == 96:
        # 15 分钟节点：首点为 00:15（与有时间列的文件内容一致），末点为次日 00:00
        timestamps = [base + pd.Timedelta(minutes=15 * (i + 1)) for i in range(96)]
    elif n == 48:
        timestamps = [base + pd.Timedelta(minutes=30 * i) for i in range(48)]
    elif n == 288:
        timestamps = [base + pd.Timedelta(minutes=5 * i) for i in range(288)]
    else:
        # 15min 步长时首点 00:15
        timestamps = [base + pd.Timedelta(minutes=15 * (i + 1)) for i in range(n)]
    ts = pd.Series(timestamps[:n], index=df.index)
    col = "__filename_time"
    df[col] = ts
    return col


def pick_time_col(df: pd.DataFrame) -> Optional[str]:
    """选用最佳时间列：infer_time_summary 结果需 coverage_total>=0.6 且 parse_rate_non_null>=0.9，否则按列名正则尝试。"""
    ts = infer_time_summary(df)
    if ts and ts.coverage_total >= 0.6 and ts.parse_rate_non_null >= 0.9:
        return ts.time_column
    # fallback: try any column name matching regex and convertible
    for c in df.columns:
        if TIME_COL_NAME_RE.search(str(c)):
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().mean() >= 0.6:
                return str(c)
    return None


# -----------------------------------------------------------------------------
# 时间列选取、数值聚合（含规则驱动的 default_agg / price_weighted_avg）
# -----------------------------------------------------------------------------

def extract_metrics_from_df(
    df: pd.DataFrame,
    time_col: str,
    base_metric_parts: List[str],
    rel_path: str = "",
    sheet_name: str = "",
    max_metrics_per_sheet: int = 30,
    rule: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    """
    按 time_col 对齐时间戳，对数值列按规则或启发式做聚合（mean/sum/last/weighted_avg），
    返回 (指标名 -> Series, 指标名 -> 原始列名原文)。原始列名用于入库追溯（source_column）。
    """
    df = df.copy()
    # unify time
    dt = pd.to_datetime(df[time_col], errors="coerce")
    df = df.assign(__timestamp=dt).dropna(subset=["__timestamp"])
    if df.empty:
        return {}, {}

    df = df.drop(columns=[time_col], errors="ignore")
    df = coerce_numeric(df)

    numeric_cols = [c for c in df.columns if c != "__timestamp" and pd.api.types.is_numeric_dtype(df[c])]
    # 【数据质量审计结论】第一层：始终排除模板/废弃列（Unnamed、序号、index、编号）
    numeric_cols = [c for c in numeric_cols if not DEFAULT_EXCLUDE_COL_RE.search(str(c))]
    # 【规则扩展】第二层：规则中的 exclude_columns_regex 可再排除（如 节点|node|ID），见 dataset_rules.example.json
    if rule:
        exc_re = rule.get("exclude_columns_regex")
        if exc_re:
            exc_pat = re.compile(exc_re, re.IGNORECASE)
            numeric_cols = [c for c in numeric_cols if not exc_pat.search(str(c))]
    if not numeric_cols:
        # 无可用数值列时，既没有指标也没有列名映射
        return {}, {}

    # -----------------------------------------------------------------------------
    # 特殊情况：总表事后两个出清汇总表，每表 4 列价格（分区价 + 节点均价），共 8 条，按列名白名单抽取。
    # 必须用 rel_path + sheet 区分，避免与其它文件同名列混淆。
    # -----------------------------------------------------------------------------
    basename = os.path.basename(rel_path) if rel_path else ""
    g = df.groupby("__timestamp", sort=True)

    # 日前/实时出清表：用短前缀(phase+region+file_tag)避免 200 字截断导致同名列碰撞
    short_base = [base_metric_parts[1] if len(base_metric_parts) > 1 else "", base_metric_parts[2] if len(base_metric_parts) > 2 else "", ""]
    if basename == "日前出清结果（公开）_合并总表.xlsx" and "日前出清结果（公开）" in sheet_name:
        metrics = {}
        source_columns = {}
        short_base[2] = "日前出清"
        target_cols = [
            "江南分区价格（元/MWh）",
            "江北分区价格（元/MWh）",
            "江南分区节点边际电价均价（元/MWh）",
            "江北分区节点边际电价均价（元/MWh）",
        ]
        for col in target_cols:
            if col in df.columns:
                series = g[col].mean()
                mname = normalize_metric_name(short_base + ["mean", str(col)])
                metrics[mname] = series
                source_columns[mname] = str(col)
        return metrics, source_columns

    if basename == "实时出清结果（公开）_合并总表.xlsx" and "实时出清结果（公开）" in sheet_name:
        metrics = {}
        source_columns = {}
        short_base[2] = "实时出清"
        # 只取终发布分区价 + 两条节点均价，避免与「江南/江北分区价格（元/MWh）」重复覆盖
        # 终发布标识可能是半角/全角括号，统一按“包含 终发布”判断
        for col in df.columns:
            if col == "__timestamp" or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            col_str = str(col)
            if "江南分区价格" in col_str and "终发布" in col_str and "元/MWh" in col_str and "节点" not in col_str:
                series = g[col].mean()
                mname = normalize_metric_name(short_base + ["mean", col_str])
                metrics[mname] = series
                source_columns[mname] = col_str
            elif "江北分区价格" in col_str and "终发布" in col_str and "元/MWh" in col_str and "节点" not in col_str:
                series = g[col].mean()
                mname = normalize_metric_name(short_base + ["mean", col_str])
                metrics[mname] = series
                source_columns[mname] = col_str
            elif "江南分区节点边际电价均价" in col_str and "元/MWh" in col_str:
                series = g[col].mean()
                mname = normalize_metric_name(short_base + ["mean", col_str])
                metrics[mname] = series
                source_columns[mname] = col_str
            elif "江北分区节点边际电价均价" in col_str and "元/MWh" in col_str:
                series = g[col].mean()
                mname = normalize_metric_name(short_base + ["mean", col_str])
                metrics[mname] = series
                source_columns[mname] = col_str
        return metrics, source_columns

    # Decide aggregation strategy
    agg_conf = (rule or {}).get("aggregations", {})
    default_agg = agg_conf.get("default", "mean")
    if default_agg not in {"mean", "sum", "last", "none"}:
        default_agg = "mean"

    # Prefer对所有价格列做处理（rule 可覆盖 regex）；权重列优先选 \"电量/成交电量\"，避免误把价格列当权重
    pw_conf = agg_conf.get("price_weighted_avg") if agg_conf else None
    if pw_conf:
        price_re = re.compile(pw_conf.get("price_col_regex", ""), re.IGNORECASE)
        price_cols = [c for c in numeric_cols if price_re.search(str(c))]
        weight_re = re.compile(pw_conf.get("weight_col_regex", ""), re.IGNORECASE)
        weight_candidates = [c for c in numeric_cols if weight_re.search(str(c))]
    else:
        price_cols = [c for c in numeric_cols if PRICE_COL_RE.search(str(c))]
        weight_candidates = [c for c in numeric_cols if WEIGHT_COL_RE.search(str(c))]
    metrics: Dict[str, pd.Series] = {}
    source_columns: Dict[str, str] = {}  # metric_key -> 原始列名原文

    g = df.groupby("__timestamp", sort=True)

    # 对所有价格列逐个生成指标（优先生成加权均价，再生成默认聚合），确保江南/江北等不会因列顺序或 max_metrics 限制被漏掉
    if price_cols:
        wcol = weight_candidates[0] if weight_candidates else None
        for price_col in price_cols:
            if wcol:
                sub = df[["__timestamp", price_col, wcol]].dropna()
                if not sub.empty:
                    def _wavg(x: pd.DataFrame) -> float:
                        w = x[wcol].to_numpy()
                        p = x[price_col].to_numpy()
                        s = np.nansum(w)
                        if s == 0:
                            return float(np.nanmean(p))
                        return float(np.nansum(p * w) / s)

                    wavg = sub.groupby("__timestamp").apply(_wavg)
                    mname = normalize_metric_name(base_metric_parts + ["weighted_avg", str(price_col)])
                    metrics[mname] = wavg
                    source_columns[mname] = str(price_col)

            if default_agg != "none":
                if default_agg == "mean":
                    series = g[price_col].mean()
                elif default_agg == "sum":
                    series = g[price_col].sum()
                else:
                    series = g[price_col].last()
                mname = normalize_metric_name(base_metric_parts + [default_agg, str(price_col)])
                metrics[mname] = series
                source_columns[mname] = str(price_col)

    # add up to N numeric columns as aggregated series
    used = set(metrics.keys())

    count = 0
    for c in numeric_cols:
        # 已经作为价格列处理过的就不再走默认聚合，避免重复
        if c in price_cols:
            continue
        if count >= max_metrics_per_sheet:
            break
        if default_agg == "none":
            break
        if default_agg == "mean":
            series = g[c].mean()
        elif default_agg == "sum":
            series = g[c].sum()
        else:
            series = g[c].last()
        mname = normalize_metric_name(base_metric_parts + [default_agg, str(c)])
        if mname in used:
            continue
        metrics[mname] = series
        source_columns[mname] = str(c)
        used.add(mname)
        count += 1

    return metrics, source_columns


def _read_excel(path: str, sheet_name: str, nrows: Optional[int]) -> pd.DataFrame:
    """读取指定 sheet：nrows 不为 None 时用 read_excel_preview；否则全表读，engine 优先 calamine。"""
    if nrows is not None:
        return read_excel_preview(path, sheet_name=sheet_name, nrows=nrows)
    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="calamine")
    except Exception:
        try:
            return pd.read_excel(path, sheet_name=sheet_name)
        except Exception:
            return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def _safe_parse_freq(freq: str) -> str:
    """校验并返回 pandas 可识别的频率字符串（如 15min/1H/1D），非法则抛 ValueError。"""
    try:
        pd.tseries.frequencies.to_offset(freq)
        return freq
    except Exception as e:
        raise ValueError(f"非法 freq：{freq}（示例：15min / 1H / 1D）") from e


def load_rules(path: str) -> Dict[str, Any]:
    """从 JSON 文件加载规则字典；空路径返回 {}。"""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as r:
        return json.load(r)


def find_rule_for_sheet(rules: Dict[str, Any], rel_path: str, sheet_name: str) -> Optional[Dict[str, Any]]:
    """按 file_rules 中 match_path_regex（及可选 sheet_regex）匹配，返回第一条命中规则，否则 None。"""
    file_rules = rules.get("file_rules") or []
    for rule in file_rules:
        pat = rule.get("match_path_regex")
        if not pat:
            continue
        if not re.search(pat, rel_path):
            continue
        sheet_pat = rule.get("sheet_regex")
        if sheet_pat and not re.search(sheet_pat, str(sheet_name)):
            continue
        return rule
    return None


# -----------------------------------------------------------------------------
# 入口：参数解析、遍历文件/sheet、抽取指标、重采样、写 parquet 与 meta
# -----------------------------------------------------------------------------

def main() -> int:
    """解析命令行、执行抽取与可选重采样、写入 parquet 与 meta JSON，返回 0。"""
    ap = argparse.ArgumentParser(description="Build a unified time-series dataset from heterogeneous Excel files.")
    ap.add_argument("--data-root", required=True, help="Repo root containing data folders.")
    ap.add_argument("--output-parquet", required=True, help="Output parquet path for unified dataset.")
    ap.add_argument("--output-meta", required=True, help="Output json path for extraction metadata.")
    ap.add_argument(
        "--mode",
        choices=["full", "preview"],
        default="full",
        help="full=全量读取（默认）；preview=仅用于调试的截断抽样读取。",
    )
    ap.add_argument("--max-files", type=int, default=0, help="Limit Excel files for debugging (0 = all).")
    ap.add_argument(
        "--preview-rows",
        type=int,
        default=5000,
        help="preview 模式下每个 sheet 读取的行数上限（top N）。full 模式下会忽略该参数。",
    )
    ap.add_argument(
        "--max-sheets-per-file",
        type=int,
        default=0,
        help="每个文件最多处理多少个 sheet（0=全部）。preview 模式可用来加速。",
    )
    ap.add_argument(
        "--max-metrics-per-sheet",
        type=int,
        default=30,
        help="每个 sheet 最多抽取多少条指标（列）。规则文件中可单独覆盖。",
    )
    ap.add_argument(
        "--freq",
        default="",
        help="输出统一频率（例如 15min/1H/1D）。为空则不重采样，仅做时间点 union。",
    )
    ap.add_argument(
        "--resample-agg",
        choices=["mean", "sum", "last"],
        default="mean",
        help="统一频率重采样时的聚合方式。",
    )
    ap.add_argument(
        "--rules",
        default="",
        help="（可选）抽取规则 JSON 文件路径。示例见 scripts/dataset_rules.example.json。",
    )
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    out_parquet = os.path.abspath(args.output_parquet)
    out_meta = os.path.abspath(args.output_meta)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)

    files = iter_excel_files(data_root)
    files.sort()
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    metric_series: Dict[str, pd.Series] = {}
    metric_meta: Dict[str, Dict[str, str]] = {}
    per_file_meta: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []
    rules = load_rules(args.rules) if args.rules else {}

    t0 = time.time()
    for path in tqdm(files, desc="Extracting"):
        rel = os.path.relpath(path, data_root)
        kind = classify_from_path(rel)
        all_sheets = list_sheets_quick(path)
        if args.max_sheets_per_file and args.max_sheets_per_file > 0:
            sheets = all_sheets[: args.max_sheets_per_file]
        else:
            sheets = all_sheets
        if not sheets:
            errors.append({"file": rel, "error": "无法读取sheet列表"})
            continue

        file_metrics = 0
        for sh in sheets:
            try:
                nrows = args.preview_rows if args.mode == "preview" else None
                df = _read_excel(path, sheet_name=sh, nrows=nrows)
            except Exception as e:
                errors.append({"file": rel, "sheet": str(sh), "error": repr(e)[:500]})
                continue

            # 优先处理「查询日期 + 时间」双列：组合为精确到 15 分钟的时间戳
            tcol = None
            time_source = ""
            date_cols = [c for c in df.columns if "查询日期" in str(c)]
            time_cols = [c for c in df.columns if "时间" == str(c) or str(c).endswith("时间")]
            if date_cols and time_cols:
                dcol = str(date_cols[0])
                tcol_raw = str(time_cols[0])
                try:
                    date_part = pd.to_datetime(df[dcol], errors="coerce")
                    time_part = df[tcol_raw].astype(str).str.strip()
                    # 兼容 24:00 / 24:00:00：按次日 00:00 处理
                    is_24h = time_part.str.fullmatch(r"24:00(?::00)?")
                    time_part = time_part.where(~is_24h, "00:00")
                    date_part = date_part + pd.to_timedelta(is_24h.fillna(False).astype(int), unit="D")
                    combined = pd.to_datetime(
                        date_part.dt.strftime("%Y-%m-%d") + " " + time_part,
                        errors="coerce",
                    )
                    if combined.notna().mean() >= 0.8:
                        df = df.copy()
                        df["__combined_datetime"] = combined
                        tcol = "__combined_datetime"
                        time_source = "date+time_columns"
                except Exception:
                    tcol = None
                    time_source = ""

            # 默认按单列时间识别 + 文件名推断
            if not tcol:
                tcol = pick_time_col(df)
                time_source = "column"
                if not tcol:
                    tcol = infer_time_column_from_filename(rel, df)
                    time_source = "filename" if tcol else ""
            if not tcol:
                continue

            rule = find_rule_for_sheet(rules, rel, str(sh)) if rules else None
            # allow per-rule override of max-metrics
            max_metrics = args.max_metrics_per_sheet
            if rule and isinstance(rule.get("max_metrics_per_sheet"), int):
                max_metrics = int(rule["max_metrics_per_sheet"])

            dims = parse_dimensions(rel)
            base_parts = [kind, dims.get("phase", ""), dims.get("region", ""), dims.get("family", ""), os.path.basename(rel), str(sh)]
            ms, source_cols = extract_metrics_from_df(
                df,
                tcol,
                base_parts,
                rel_path=rel,
                sheet_name=str(sh),
                max_metrics_per_sheet=max_metrics,
                rule=rule,
            )
            for k, s in ms.items():
                raw_col = source_cols.get(k, "")
                if k in metric_series:
                    # merge by preferring existing; if collision, suffix
                    kk = k
                    i = 2
                    while kk in metric_series:
                        kk = f"{k}__v{i}"
                        i += 1
                    metric_series[kk] = s
                    metric_meta[kk] = {
                        "source_file": rel,
                        "sheet": str(sh),
                        "time_col": str(tcol),
                        "time_source": time_source,
                        "kind": kind,
                        "source_column": raw_col,
                        **dims,
                    }
                else:
                    metric_series[k] = s
                    metric_meta[k] = {
                        "source_file": rel,
                        "sheet": str(sh),
                        "time_col": str(tcol),
                        "time_source": time_source,
                        "kind": kind,
                        "source_column": raw_col,
                        **dims,
                    }
            file_metrics += len(ms)

        per_file_meta.append({"file": rel, "kind": kind, "sheets_processed": sheets, "metrics_extracted": file_metrics})

    # assemble wide table
    if not metric_series:
        raise RuntimeError("未抽取到任何可用时间序列（请检查 Excel 读取与时间列识别规则）。")

    all_index = None
    for s in metric_series.values():
        all_index = s.index if all_index is None else all_index.union(s.index)
    all_index = all_index.sort_values()

    df_out = pd.DataFrame(index=all_index)
    for name, s in metric_series.items():
        df_out[name] = s.reindex(all_index)

    df_out.index.name = "timestamp"
    df_out = df_out.reset_index()

    # enforce datetime
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"], errors="coerce")
    df_out = df_out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # optional uniform frequency
    if args.freq:
        freq = _safe_parse_freq(args.freq)
        df_rs = df_out.set_index("timestamp")
        if args.resample_agg == "mean":
            df_rs = df_rs.resample(freq).mean()
        elif args.resample_agg == "sum":
            df_rs = df_rs.resample(freq).sum()
        else:
            df_rs = df_rs.resample(freq).last()
        # regular grid
        df_rs = df_rs.asfreq(freq)
        df_out = df_rs.reset_index()

    # write
    df_out.to_parquet(out_parquet, index=False)

    meta = {
        "data_root": data_root,
        "excel_files": len(files),
        "metrics": len(metric_series),
        "rows": int(df_out.shape[0]),
        "cols": int(df_out.shape[1]),
        "duration_sec": round(time.time() - t0, 2),
        "mode": args.mode,
        "preview_rows": args.preview_rows if args.mode == "preview" else None,
        "max_sheets_per_file": args.max_sheets_per_file,
        "freq": args.freq,
        "resample_agg": args.resample_agg if args.freq else None,
        "metric_meta": metric_meta,
        "errors": errors[:1000],
        "per_file": per_file_meta,
    }
    with open(out_meta, "w", encoding="utf-8") as w:
        json.dump(meta, w, ensure_ascii=False, indent=2)

    print(f"Wrote parquet: {out_parquet}")
    print(f"Wrote meta: {out_meta}")
    print(f"Metrics: {len(metric_series)}, rows: {df_out.shape[0]}, cols: {df_out.shape[1]}")
    if errors:
        print(f"Errors: {len(errors)} (see meta, truncated to 1000)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

