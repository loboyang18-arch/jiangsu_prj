from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            # try numeric coercion for strings
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="ignore")
    return out


def pick_time_col(df: pd.DataFrame) -> Optional[str]:
    ts = infer_time_summary(df)
    if ts and ts.coverage >= 0.6:
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

    # Prefer price/weight aggregation if we detect them
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

        # also plain mean for the main price col
        mname = normalize_metric_name(base_metric_parts + ["mean", str(price_col)])
        metrics[mname] = g[price_col].mean()

    # add up to N numeric columns as mean series
    used = set()
    for name in list(metrics.keys()):
        used.add(name)

    count = 0
    for c in numeric_cols:
        if count >= max_metrics_per_sheet:
            break
        mname = normalize_metric_name(base_metric_parts + ["mean", str(c)])
        if mname in used:
            continue
        metrics[mname] = g[c].mean()
        used.add(mname)
        count += 1

    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a unified time-series dataset from heterogeneous Excel files.")
    ap.add_argument("--data-root", required=True, help="Repo root containing data folders.")
    ap.add_argument("--output-parquet", required=True, help="Output parquet path for unified dataset.")
    ap.add_argument("--output-meta", required=True, help="Output json path for extraction metadata.")
    ap.add_argument("--max-files", type=int, default=0, help="Limit Excel files for debugging (0 = all).")
    ap.add_argument("--preview-rows", type=int, default=5000, help="Rows to read per sheet (top N).")
    ap.add_argument("--max-sheets-per-file", type=int, default=5, help="Max sheets per file to process.")
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
    per_file_meta: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    t0 = time.time()
    for path in tqdm(files, desc="Extracting"):
        rel = os.path.relpath(path, data_root)
        kind = classify_from_path(rel)
        sheets = list_sheets_quick(path)[: args.max_sheets_per_file]
        if not sheets:
            errors.append({"file": rel, "error": "无法读取sheet列表"})
            continue

        file_metrics = 0
        for sh in sheets:
            try:
                df = pd.read_excel(path, sheet_name=sh, nrows=args.preview_rows)
            except Exception:
                try:
                    df = pd.read_excel(path, sheet_name=sh, nrows=args.preview_rows, engine="openpyxl")
                except Exception as e:
                    errors.append({"file": rel, "sheet": str(sh), "error": repr(e)[:500]})
                    continue

            tcol = pick_time_col(df)
            if not tcol:
                continue

            base_parts = [kind, os.path.basename(rel), str(sh)]
            ms = extract_metrics_from_df(df, tcol, base_parts)
            for k, s in ms.items():
                if k in metric_series:
                    # merge by preferring existing; if collision, suffix
                    kk = k
                    i = 2
                    while kk in metric_series:
                        kk = f"{k}__v{i}"
                        i += 1
                    metric_series[kk] = s
                else:
                    metric_series[k] = s
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

    # write
    df_out.to_parquet(out_parquet, index=False)

    meta = {
        "data_root": data_root,
        "excel_files": len(files),
        "metrics": len(metric_series),
        "rows": int(df_out.shape[0]),
        "cols": int(df_out.shape[1]),
        "duration_sec": round(time.time() - t0, 2),
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

