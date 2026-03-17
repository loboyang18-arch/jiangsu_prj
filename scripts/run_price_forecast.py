#!/usr/bin/env python3
"""价格预测一键驱动：读取计划，串联特征工程与训练回测。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _abs(path: str) -> str:
    return os.path.abspath(path)


def _int_list(values: List[int], override: str) -> List[int]:
    if not override.strip():
        return list(values)
    return [int(x) for x in override.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run configured V1 price forecasting workflow.")
    ap.add_argument("--plan", required=True, help="Path to scripts/price_forecast_plan.json")
    ap.add_argument("--only-task", default="", help="Only run one task_name")
    ap.add_argument("--horizons-override", default="", help="Override horizons, e.g. 1,2,4")
    ap.add_argument("--smoke", action="store_true", help="Smoke mode: force one task + horizon=1")
    args = ap.parse_args()

    plan_path = Path(_abs(args.plan))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    repo_root = Path(plan["paths"]["repo_root"])
    py = sys.executable

    input_parquet = _abs(plan["paths"]["input_parquet"])
    registry = _abs(plan["paths"]["feature_registry_json"])
    out_root = Path(_abs(plan["paths"]["output_root"]))
    out_root.mkdir(parents=True, exist_ok=True)

    fe = plan["feature_engineering"]
    tr = plan["training"]
    v1 = plan["v1"]
    tasks: List[Dict] = list(v1["tasks"])
    horizons = _int_list(v1["horizons"], args.horizons_override)

    if args.only_task.strip():
        tasks = [t for t in tasks if t["task_name"] == args.only_task.strip()]
        if not tasks:
            raise ValueError(f"task_name 不存在：{args.only_task}")
    if args.smoke:
        tasks = tasks[:1]
        horizons = [1]

    summary_rows: List[Dict] = []
    for task in tasks:
        task_name = task["task_name"]
        task_mode = task["task_mode"]
        target_col = task["target_col"]
        decision_time_policy = task.get("decision_time_policy", fe.get("decision_time_policy", "asof_timestamp"))
        task_dir = out_root / task_name
        feat_dir = task_dir / "features"
        model_dir = task_dir / "models"
        feat_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        feat_paths: List[str] = []
        for h in horizons:
            feat_path = feat_dir / f"features_h{h}.parquet"
            meta_path = feat_dir / f"features_h{h}.meta.json"
            cmd = [
                py,
                str(repo_root / "scripts" / "feature_engineering.py"),
                "--input-parquet",
                input_parquet,
                "--output-parquet",
                str(feat_path),
                "--output-meta",
                str(meta_path),
                "--target-col",
                target_col,
                "--task-mode",
                task_mode,
                "--registry-json",
                registry,
                "--freq",
                fe["freq"],
                "--horizon",
                str(h),
                "--lags",
                fe["lags"],
                "--roll-windows",
                fe["roll_windows"],
                "--max-missing-rate",
                str(fe["max_missing_rate"]),
                "--decision-time-policy",
                decision_time_policy,
            ]
            if fe.get("allow_unknown_features", False):
                cmd.append("--allow-unknown-features")
            if fe.get("allow_current_target_feature", False):
                cmd.append("--allow-current-target-feature")
            if fe.get("disable_target_history", False):
                cmd.append("--disable-target-history")
            _run(cmd)
            feat_paths.append(str(feat_path))

        metrics_json = task_dir / f"metrics_{task_name}.json"
        summary_csv = task_dir / "backtest_summary.csv"
        train_cmd = [
            py,
            str(repo_root / "scripts" / "train_baseline.py"),
            "--features-parquets",
            ",".join(feat_paths),
            "--model-dir",
            str(model_dir),
            "--target-col",
            tr["target_col"],
            "--train-frac",
            str(tr["train_frac"]),
            "--val-frac",
            str(tr["val_frac"]),
            "--objective",
            tr["objective"],
            "--quantile-alphas",
            ",".join(str(x) for x in tr.get("quantile_alphas", [0.1, 0.5, 0.9])),
            "--n-estimators",
            str(tr.get("n_estimators", 800)),
            "--metrics-json",
            str(metrics_json),
            "--summary-csv",
            str(summary_csv),
        ]
        if tr.get("rolling_backtest", False):
            train_cmd.append("--rolling-backtest")
            train_cmd.extend(["--rolling-step-size", str(tr.get("rolling_step_size", 0))])
        _run(train_cmd)

        task_summary = pd.read_csv(summary_csv)
        task_summary["task_name"] = task_name
        task_summary["task_mode"] = task_mode
        task_summary["target_col"] = target_col
        summary_rows.append(task_summary)

    if summary_rows:
        all_summary = pd.concat(summary_rows, ignore_index=True)
        all_summary_path = out_root / "backtest_summary.csv"
        all_summary.to_csv(all_summary_path, index=False, encoding="utf-8-sig")
        print(f"[OK] wrote summary: {all_summary_path}")

    # V2 扩展占位：输出配置快照，供下一轮日内96点训练调用
    v2_recipe = out_root / "v2_recipe.json"
    v2_cfg = {
        "description": "V2 direct multi-horizon & quantile recipe",
        "horizons": plan.get("v2", {}).get("horizons", [1, 4, 8, 16, 32, 64, 96]),
        "objective": plan.get("v2", {}).get("objective", "quantile"),
        "quantile_alphas": plan.get("v2", {}).get("quantile_alphas", [0.1, 0.5, 0.9]),
    }
    v2_recipe.write_text(json.dumps(v2_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote v2 recipe: {v2_recipe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

