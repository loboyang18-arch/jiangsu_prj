from __future__ import annotations

"""
基于 DWD 与维表的简单可视化脚本。

功能（初版）：
- 单特征时间序列折线图（15 分钟级别，可选日聚合）
- 缺失情况热力图（日期 × 96 个 15 分钟网格）

使用示例：

python scripts/visualize_features.py \
  --metric-id receive_actual_huadong \
  --region 华东 \
  --out-dir report/figures \
  --heatmap \
  --daily
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 基本字体设置：优先使用常见中文字体，避免图例/标题乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "Heiti TC", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dwd(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # 规范 timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def load_dim_map(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df


def build_human_label(metric_id: str, region_id: str, dim_map: pd.DataFrame) -> str:
    """
    尝试从 dim_source_field_map 中构造“文件主题 + 列名原文”的人类可读标签。
    """
    if dim_map.empty:
        return metric_id

    sub = dim_map[dim_map["metric_id"] == metric_id]
    if region_id:
        sub = sub[sub["region_id"] == region_id]
    if sub.empty:
        return metric_id

    row = sub.iloc[0]
    source_file = str(row.get("source_file", "") or "")
    source_column = str(row.get("source_column", "") or "")

    # 从路径中取最后一段文件名（不含目录）
    file_name = os.path.basename(source_file)
    # 去掉扩展名
    if file_name.lower().endswith(".xlsx") or file_name.lower().endswith(".xls"):
        file_name = os.path.splitext(file_name)[0]

    if source_column:
        return f"{file_name} · {source_column}"
    return file_name or metric_id


def plot_timeseries(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_id: str,
    region_id: Optional[str],
    label: str,
    daily: bool = False,
) -> None:
    sub = df[df["metric_id"] == metric_id].copy()
    if region_id is not None:
        sub = sub[sub["region_id"] == region_id]
    if sub.empty:
        print(f"[warn] metric_id={metric_id}, region_id={region_id} 无数据，跳过时序图。")
        return

    sub = sub.sort_values("timestamp")

    if daily:
        # 日聚合（取均值）
        sub["date"] = sub["timestamp"].dt.date
        g = sub.groupby("date")["value"].mean().reset_index()
        x = pd.to_datetime(g["date"])
        y = g["value"]
        title = f"{metric_id} ({region_id or 'ALL'}) - 日均值\n{label}"
    else:
        x = sub["timestamp"]
        y = sub["value"]
        title = f"{metric_id} ({region_id or 'ALL'}) - 15min\n{label}"
    ax.plot(x, y, linewidth=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time")
    ax.set_ylabel("value")


def plot_heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_id: str,
    region_id: Optional[str],
    label: str,
    last_days: Optional[int] = None,
) -> None:
    sub = df[df["metric_id"] == metric_id].copy()
    if region_id is not None:
        sub = sub[sub["region_id"] == region_id]
    if sub.empty:
        print(f"[warn] metric_id={metric_id}, region_id={region_id} 无数据，跳过热力图。")
        return

    sub = sub.sort_values("timestamp")
    sub["date"] = sub["timestamp"].dt.date
    # 15 分钟序号：0..95
    sub["slot"] = (sub["timestamp"].dt.hour * 60 + sub["timestamp"].dt.minute) // 15

    # 最近 N 天
    if last_days is not None and last_days > 0:
        max_date = sub["date"].max()
        min_date = max_date - pd.Timedelta(days=last_days - 1)
        sub = sub[(sub["date"] >= min_date) & (sub["date"] <= max_date)]

    dates = sorted(sub["date"].unique())
    if not dates:
        print(f"[warn] metric_id={metric_id}, region_id={region_id} 过滤后无日期，跳过热力图。")
        return

    slots = list(range(96))
    mat = np.zeros((len(dates), len(slots)), dtype=int)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    for _, row in sub.iterrows():
        d = row["date"]
        s = int(row["slot"])
        if 0 <= s < 96:
            mat[date_to_idx[d], s] = 1

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.colorbar(im, ax=ax, label="是否有值 (1=有,0=无)")
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels([str(d) for d in dates], fontsize=6)
    ax.set_xticks(
        range(0, 96, 8),
        [f"{h:02d}:00" for h in range(0, 24, 2)],
        rotation=45,
        fontsize=8,
    )
    title = f"{metric_id} ({region_id or 'ALL'}) 缺失热力图\n{label}"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("时间槽位（15min）")
    ax.set_ylabel("日期")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DWD 特征可视化（总表入库结果）")
    ap.add_argument(
        "--dwd-parquet",
        default="warehouse/dwd/dwd_timeseries_15m.parquet",
        help="DWD 长表 parquet 路径",
    )
    ap.add_argument(
        "--dim-source-map",
        default="warehouse/dim/dim_source_field_map.parquet",
        help="dim_source_field_map parquet 路径，用于生成业务标签",
    )
    ap.add_argument(
        "--metric-id",
        nargs="+",
        required=True,
        help="要可视化的 metric_id 列表（一个或多个）",
    )
    ap.add_argument(
        "--region",
        default=None,
        help="region_id（如 汇总 / 华东 / 江南 / 江北），为空则不过滤",
    )
    ap.add_argument(
        "--out-dir",
        default="report/figures",
        help="输出图片目录",
    )
    ap.add_argument(
        "--no-daily",
        action="store_true",
        help="不生成日聚合折线图（默认会生成）",
    )
    ap.add_argument(
        "--no-heatmap",
        action="store_true",
        help="不生成缺失热力图（默认会生成）",
    )
    ap.add_argument(
        "--heatmap-last-days",
        type=int,
        default=60,
        help="缺失热力图仅展示最近 N 天（默认 60，<=0 表示全期）",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    dwd_path = args.dwd_parquet
    dim_map_path = args.dim_source_map
    out_dir = args.out_dir

    if not os.path.exists(dwd_path):
        raise SystemExit(f"DWD parquet 不存在: {dwd_path}")

    print(f"[info] 读取 DWD: {dwd_path}")
    df = load_dwd(dwd_path)

    print(f"[info] 读取维表: {dim_map_path}")
    dim_map = load_dim_map(dim_map_path)

    region = args.region
    for metric_id in args.metric_id:
        label = build_human_label(metric_id, region, dim_map)
        print(f"[info] metric_id={metric_id}, region={region or 'ALL'}, label={label}")

        # 为当前 metric 建一个统一窗口，将多个子图放在同一 Figure 中
        panels = []
        panels.append(("15min", True))  # (kind, enabled)
        if not args.no_daily:
            panels.append(("daily", True))
        if not args.no_heatmap:
            panels.append(("heatmap", True))

        n = len(panels)
        fig, axes = plt.subplots(n, 1, figsize=(11, 3 * n), squeeze=False)
        axes = axes.flatten()

        ax_idx = 0
        # 15min 折线图
        plot_timeseries(axes[ax_idx], df, metric_id, region, label, daily=False)
        ax_idx += 1

        # 日聚合折线图
        if not args.no_daily:
            plot_timeseries(axes[ax_idx], df, metric_id, region, label, daily=True)
            ax_idx += 1

        # 缺失热力图
        if not args.no_heatmap:
            last_days = args.heatmap_last_days
            if last_days is not None and last_days <= 0:
                last_days = None
            plot_heatmap(axes[ax_idx], df, metric_id, region, label, last_days=last_days)

        fig.tight_layout()
        print(f"[show] 展示窗口：metric_id={metric_id}, region={region or 'ALL'}")
        plt.show()

    print("[done] 可视化完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

