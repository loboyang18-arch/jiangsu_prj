"""
扫描项目内所有 Excel 文件，生成字段级数据字典（data_dictionary）。

对每个文件的每个 sheet：识别时间列与时间范围/粒度，汇总每列 dtype、非空数、唯一值、示例值，
输出 CSV（明细）与 Markdown（摘要 + 错误列表）。异常文件/ sheet 不阻断流程，仅记录到错误表。
供数据盘点、rules 设计与 build_dataset 前的结构摸底使用。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from _excel_scan_utils import infer_time_summary, read_excel_preview, summarize_columns


def iter_excel_files(data_root: str) -> List[str]:
    """递归列出 data_root 下所有 .xls/.xlsx 文件路径（跳过 .git）。"""
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


def list_sheets(path: str) -> List[str]:
    """获取工作簿的 sheet 名称列表；失败时尝试 openpyxl（兼容 .xls 实为 OOXML）。"""
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


def main() -> int:
    """解析参数、遍历 Excel、写 data_dictionary.csv 与 data_dictionary.md，返回 0。"""
    ap = argparse.ArgumentParser(description="Scan Excel files and build a data dictionary.")
    ap.add_argument("--data-root", required=True, help="Project root containing data folders (e.g. repo root).")
    ap.add_argument("--output-dir", required=True, help="Output directory for data_dictionary.md/csv.")
    ap.add_argument("--max-files", type=int, default=0, help="Limit number of Excel files for debugging (0 = all).")
    ap.add_argument("--preview-rows", type=int, default=300, help="Rows to preview per sheet.")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    excel_files = iter_excel_files(data_root)
    excel_files.sort()
    if args.max_files and args.max_files > 0:
        excel_files = excel_files[: args.max_files]

    csv_path = os.path.join(out_dir, "data_dictionary.csv")
    md_path = os.path.join(out_dir, "data_dictionary.md")

    rows: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    t0 = time.time()
    for path in tqdm(excel_files, desc="Scanning Excel"):
        rel = os.path.relpath(path, data_root)
        sheets = list_sheets(path)
        if not sheets:
            errors.append({"file": rel, "error": "无法读取sheet列表"})
            continue

        for sh in sheets:
            try:
                df = read_excel_preview(path, sh, nrows=args.preview_rows)
            except Exception as e:
                errors.append({"file": rel, "sheet": str(sh), "error": repr(e)[:500]})
                continue

            time_sum = infer_time_summary(df)
            cols = summarize_columns(df)

            # record sheet-level summary row
            rows.append(
                {
                    "file": rel,
                    "sheet": str(sh),
                    "row_type": "sheet_summary",
                    "time_column": time_sum.time_column if time_sum else "",
                    "time_start": time_sum.start if time_sum else "",
                    "time_end": time_sum.end if time_sum else "",
                    "time_granularity": time_sum.granularity if time_sum else "",
                    "time_coverage_total": f"{time_sum.coverage_total:.3f}" if time_sum else "",
                    "time_parse_rate_non_null": f"{time_sum.parse_rate_non_null:.3f}" if time_sum else "",
                    "column": "",
                    "dtype": "",
                    "non_null": "",
                    "unique": "",
                    "example": "",
                    "top_values": "",
                }
            )

            for c in cols:
                rows.append(
                    {
                        "file": rel,
                        "sheet": str(sh),
                        "row_type": "column",
                        "time_column": "",
                        "time_start": "",
                        "time_end": "",
                        "time_granularity": "",
                        "time_coverage": "",
                        "column": c["column"],
                        "dtype": c["dtype"],
                        "non_null": c["non_null"],
                        "unique": c["unique"],
                        "example": c["example"],
                        "top_values": c["top_values"],
                    }
                )

    # write CSV
    fieldnames = [
        "file",
        "sheet",
        "row_type",
        "time_column",
        "time_start",
        "time_end",
        "time_granularity",
        "time_coverage_total",
        "time_parse_rate_non_null",
        "column",
        "dtype",
        "non_null",
        "unique",
        "example",
        "top_values",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as w:
        cw = csv.DictWriter(w, fieldnames=fieldnames)
        cw.writeheader()
        cw.writerows(rows)

    # write Markdown
    dur = time.time() - t0
    with open(md_path, "w", encoding="utf-8") as w:
        w.write("# Data Dictionary（自动生成）\n\n")
        w.write(f"- 扫描根目录：`{data_root}`\n")
        w.write(f"- Excel 文件数：{len(excel_files)}\n")
        w.write(f"- 输出：`{csv_path}`（明细CSV），`{md_path}`（本文件）\n")
        w.write(f"- 用时：{dur:.1f}s\n\n")

        if errors:
            w.write("## 读取失败/异常（不阻断流程）\n\n")
            w.write("|文件|sheet|错误|\n|---|---|---|\n")
            for e in errors[:200]:
                w.write(f"|`{e.get('file','')}`|{e.get('sheet','')}|{e.get('error','')}|\n")
            if len(errors) > 200:
                w.write(f"\n> 仅展示前 200 条错误，总计 {len(errors)} 条。\n")
            w.write("\n")

        # build a compact per-sheet table (from sheet_summary rows)
        w.write("## Sheet 级摘要（时间范围/粒度优先来自表内识别）\n\n")
        w.write("|文件|sheet|时间列|时间范围|粒度|coverage_total|parse_rate_non_null|\n|---|---|---|---|---|---:|---:|\n")
        for r in rows:
            if r["row_type"] != "sheet_summary":
                continue
            tr = ""
            if r["time_start"] and r["time_end"]:
                tr = f"{r['time_start'][:10]} ~ {r['time_end'][:10]}"
            w.write(
                f"|`{r['file']}`|{r['sheet']}|{r['time_column']}|{tr}|{r['time_granularity']}|{r['time_coverage_total']}|{r['time_parse_rate_non_null']}|\n"
            )

        w.write("\n## 字段明细\n\n")
        w.write("字段明细较长，请以 `data_dictionary.csv` 为准（可筛选 `row_type=column`）。\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    if errors:
        print(f"Errors: {len(errors)} (see markdown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

