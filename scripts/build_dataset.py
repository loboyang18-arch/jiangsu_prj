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


PRICE_COL_RE = re.compile(r"(电价|价格|均价|LMP|出清价)", re.IGNORECASE)
WEIGHT_COL_RE = re.compile(r"(电量|成交电量|MW|MWh|成交量)", re.IGNORECASE)


def iter_excel_files(data_root: str) -> List[str]:
    out: List[str] = []
    for dp, dn, fn in os.walk(data_root):
        if os.path.basename(dp) == ".git" or "/.git/" in dp:
            dn[:] = []
            continue
        for f in fn:
            ext = os.path.splitext(f)[1].lower()
            if ext in {".xls", ".xlsx"}:
                out.append(os.path.join(dp, f))
    return out


def list_sheets_quick(path: str) -> List[str]:
    # Prefer calamine for broad xls/xlsx compatibility.
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
    s = "__".join([p.strip().replace(" ", "") for p in parts if p and str(p).strip()])
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^0-9A-Za-z_\\u4e00-\\u9fff]+", "_", s)
    s = re.sub(r"__+", "__", s).strip("_")
    return s[:200]


def classify_from_path(rel_path: str) -> str:
    # A lightweight classifier used for metric naming
    if "江苏/电价数据" in rel_path:
        return "江苏电价数据"
    if "总表/事前" in rel_path:
        return "总表_事前"
    if "总表/事后" in rel_path:
        return "总表_事后"
    return "其他"

def parse_dimensions(rel_path: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    d["phase"] = "事前" if "总表/事前" in rel_path else ("事后" if "总表/事后" in rel_path else "")
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
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            # try numeric coercion for strings
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="ignore")
    return out


def pick_time_col(df: pd.DataFrame) -> Optional[str]:
    ts = infer_time_summary(df)
    # Prefer columns that cover most rows (avoid sparse header-like date columns)
    if ts and ts.coverage_total >= 0.6 and ts.parse_rate_non_null >= 0.9:
        return ts.time_column
    # fallback: try any column name matching regex and convertible
    for c in df.columns:
        if TIME_COL_NAME_RE.search(str(c)):
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().mean() >= 0.6:
                return str(c)
    return None


def extract_metrics_from_df(
    df: pd.DataFrame,
    time_col: str,
    base_metric_parts: List[str],
    max_metrics_per_sheet: int = 30,
    rule: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.Series]:
    df = df.copy()
    # unify time
    dt = pd.to_datetime(df[time_col], errors="coerce")
    df = df.assign(__timestamp=dt).dropna(subset=["__timestamp"])
    if df.empty:
        return {}

    df = df.drop(columns=[time_col], errors="ignore")
    df = coerce_numeric(df)

    numeric_cols = [c for c in df.columns if c != "__timestamp" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return {}

    # Decide aggregation strategy
    agg_conf = (rule or {}).get("aggregations", {})
    default_agg = agg_conf.get("default", "mean")
    if default_agg not in {"mean", "sum", "last", "none"}:
        default_agg = "mean"

    # Prefer price/weight aggregation if we detect them (rule can override regex)
    pw_conf = agg_conf.get("price_weighted_avg") if agg_conf else None
    if pw_conf:
        price_re = re.compile(pw_conf.get("price_col_regex", ""), re.IGNORECASE)
        weight_re = re.compile(pw_conf.get("weight_col_regex", ""), re.IGNORECASE)
        price_cols = [c for c in numeric_cols if price_re.search(str(c))]
        weight_cols = [c for c in numeric_cols if weight_re.search(str(c))]
    else:
        price_cols = [c for c in numeric_cols if PRICE_COL_RE.search(str(c))]
        weight_cols = [c for c in numeric_cols if WEIGHT_COL_RE.search(str(c))]
    metrics: Dict[str, pd.Series] = {}

    g = df.groupby("__timestamp", sort=True)

    if price_cols:
        price_col = price_cols[0]
        if weight_cols:
            wcol = weight_cols[0]
            sub = df[["__timestamp", price_col, wcol]].dropna()
            if not sub.empty:
                # weighted average
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

        # also aggregate the main price col with default_agg
        if default_agg != "none":
            if default_agg == "mean":
                series = g[price_col].mean()
            elif default_agg == "sum":
                series = g[price_col].sum()
            else:
                series = g[price_col].last()
            mname = normalize_metric_name(base_metric_parts + [default_agg, str(price_col)])
            metrics[mname] = series

    # add up to N numeric columns as aggregated series
    used = set()
    for name in list(metrics.keys()):
        used.add(name)

    count = 0
    for c in numeric_cols:
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
        used.add(mname)
        count += 1

    return metrics


def _read_excel(path: str, sheet_name: str, nrows: Optional[int]) -> pd.DataFrame:
    # Use shared reader (with calamine fallback) when nrows is small,
    # otherwise read full sheet (nrows=None) with engine priority.
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
    # Accept '15min', '1H', '1D' etc.
    try:
        pd.tseries.frequencies.to_offset(freq)
        return freq
    except Exception as e:
        raise ValueError(f"非法 freq：{freq}（示例：15min / 1H / 1D）") from e


def load_rules(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as r:
        return json.load(r)


def find_rule_for_sheet(rules: Dict[str, Any], rel_path: str, sheet_name: str) -> Optional[Dict[str, Any]]:
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


def main() -> int:
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

            tcol = pick_time_col(df)
            if not tcol:
                continue

            rule = find_rule_for_sheet(rules, rel, str(sh)) if rules else None
            # allow per-rule override of max-metrics
            max_metrics = args.max_metrics_per_sheet
            if rule and isinstance(rule.get("max_metrics_per_sheet"), int):
                max_metrics = int(rule["max_metrics_per_sheet"])

            dims = parse_dimensions(rel)
            base_parts = [kind, dims.get("phase", ""), dims.get("region", ""), dims.get("family", ""), os.path.basename(rel), str(sh)]
            ms = extract_metrics_from_df(df, tcol, base_parts, max_metrics_per_sheet=max_metrics, rule=rule)
            for k, s in ms.items():
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
                        "kind": kind,
                        **dims,
                    }
                else:
                    metric_series[k] = s
                    metric_meta[k] = {
                        "source_file": rel,
                        "sheet": str(sh),
                        "time_col": str(tcol),
                        "kind": kind,
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

