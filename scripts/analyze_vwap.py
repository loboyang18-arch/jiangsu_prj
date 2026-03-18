"""
电量加权价格分析 (VWAP Analysis)

从电量维度审视日前/实时出清价格, 回答:
  1. 简单均价 vs 电量加权均价 (VWAP) 差异
  2. 地板价的 时段占比 / 电量占比 / 收入占比 三重对比
  3. 净负荷分位与价格的联合分布
  4. 日内各时段块的 电量份额 vs 收入份额 ("收入转移"效应)
  5. 分电源类型的隐含收入估算

用法:
  python scripts/analyze_vwap.py \
    --input-parquet warehouse/feature_ready/V0/power_market_feature_ready_wide.parquet \
    --output-dir report/vwap_analysis
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.sans-serif"] = ["SimHei", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

FLOOR_PRICE = 50.0
SPIKE_PRICE = 400.0
INTERVAL_HOURS = 0.25  # 15 min


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["month"] = df["trade_date"].dt.month
    df["month_label"] = df["trade_date"].dt.strftime("%Y-%m")
    df = df.sort_values(["trade_date", "hh_index"]).reset_index(drop=True)

    df["re_total"] = (
        df.get("wind_actual_江北", 0) + df.get("wind_actual_江南", 0)
        + df.get("pv_actual_江北", 0) + df.get("pv_actual_江南", 0)
    )
    load_col = "load_actual_total_汇总"
    df["net_load"] = df[load_col] - df["re_total"]
    df["energy_mwh"] = df[load_col] * INTERVAL_HOURS
    return df


def analysis_1_vwap_by_month(df: pd.DataFrame, out_dir: str) -> dict:
    """简单均价 vs VWAP, 按月统计"""
    results = {}
    regions = [
        ("JN", "price_dayahead_jn_江南", "price_realtime_jn_final_江南"),
        ("JB", "price_dayahead_jb_江北", "price_realtime_jb_final_江北"),
    ]
    months = sorted(df["month_label"].unique())

    for region, da_col, rt_col in regions:
        rows = []
        for ml in months:
            mdf = df[df["month_label"] == ml]
            e = mdf["energy_mwh"].values
            da = mdf[da_col].values
            rt = mdf[rt_col].values

            row = {
                "month": ml,
                "days": mdf["trade_date"].nunique(),
                "total_gwh": e.sum() / 1000,
                "da_simple_avg": np.nanmean(da),
                "da_vwap": np.nansum(da * e) / np.nansum(e),
                "rt_simple_avg": np.nanmean(rt),
                "rt_vwap": np.nansum(rt * e) / np.nansum(e),
                "da_total_revenue_m": np.nansum(da * e) / 1e6,
                "rt_total_revenue_m": np.nansum(rt * e) / 1e6,
            }
            rows.append(row)
        results[region] = rows

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (region, da_col, rt_col) in enumerate(regions):
        ax = axes[idx]
        rows = results[region]
        x = np.arange(len(rows))
        w = 0.18
        ax.bar(x - 1.5 * w, [r["da_simple_avg"] for r in rows], w, label="DA 简单均价", color="#1f77b4", alpha=0.7)
        ax.bar(x - 0.5 * w, [r["da_vwap"] for r in rows], w, label="DA VWAP", color="#1f77b4")
        ax.bar(x + 0.5 * w, [r["rt_simple_avg"] for r in rows], w, label="RT 简单均价", color="#d62728", alpha=0.7)
        ax.bar(x + 1.5 * w, [r["rt_vwap"] for r in rows], w, label="RT VWAP", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels([r["month"] for r in rows])
        ax.set_ylabel("元/MWh")
        ax.set_title(f"{region} 简单均价 vs VWAP")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "1_vwap_vs_simple_avg.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1] {path}")
    return results


def analysis_2_floor_shares(df: pd.DataFrame, out_dir: str) -> dict:
    """地板价三重占比: 时段/电量/收入"""
    regions = [
        ("JN", "price_dayahead_jn_江南", "price_realtime_jn_final_江南"),
        ("JB", "price_dayahead_jb_江北", "price_realtime_jb_final_江北"),
    ]
    months = sorted(df["month_label"].unique())
    results = {}

    for region, da_col, rt_col in regions:
        rows = []
        for ml in months + ["全量"]:
            mdf = df[df["month_label"] == ml] if ml != "全量" else df
            e = mdf["energy_mwh"].values
            total_e = e.sum()

            for price_type, pcol in [("DA", da_col), ("RT", rt_col)]:
                p = mdf[pcol].values
                revenue = p * e

                floor_mask = p <= FLOOR_PRICE
                spike_mask = p >= SPIKE_PRICE

                n = len(p)
                row = {
                    "month": ml,
                    "type": price_type,
                    "time_share_floor": floor_mask.sum() / n,
                    "energy_share_floor": e[floor_mask].sum() / total_e if total_e > 0 else 0,
                    "revenue_share_floor": revenue[floor_mask].sum() / revenue.sum() if revenue.sum() > 0 else 0,
                    "time_share_spike": spike_mask.sum() / n,
                    "energy_share_spike": e[spike_mask].sum() / total_e if total_e > 0 else 0,
                    "revenue_share_spike": revenue[spike_mask].sum() / revenue.sum() if revenue.sum() > 0 else 0,
                    "floor_avg_load_gw": e[floor_mask].sum() / (INTERVAL_HOURS * floor_mask.sum()) / 1000 if floor_mask.sum() > 0 else 0,
                    "normal_avg_load_gw": e[~floor_mask & ~spike_mask].sum() / (INTERVAL_HOURS * (~floor_mask & ~spike_mask).sum()) / 1000 if (~floor_mask & ~spike_mask).sum() > 0 else 0,
                }
                rows.append(row)
        results[region] = rows

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (region, da_col, rt_col) in enumerate(regions):
        rows_region = [r for r in results[region] if r["month"] != "全量"]
        for pt_idx, price_type in enumerate(["DA", "RT"]):
            ax = axes[pt_idx][idx]
            pt_rows = [r for r in rows_region if r["type"] == price_type]
            x = np.arange(len(pt_rows))
            w = 0.25
            ax.bar(x - w, [r["time_share_floor"] * 100 for r in pt_rows], w,
                   label="时段占比", color="#ff7f0e")
            ax.bar(x, [r["energy_share_floor"] * 100 for r in pt_rows], w,
                   label="电量占比", color="#2ca02c")
            ax.bar(x + w, [r["revenue_share_floor"] * 100 for r in pt_rows], w,
                   label="收入占比", color="#9467bd")
            ax.set_xticks(x)
            ax.set_xticklabels([r["month"] for r in pt_rows])
            ax.set_ylabel("%")
            ax.set_title(f"{region} {price_type} 地板价三重占比")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "2_floor_triple_shares.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2] {path}")
    return results


def analysis_3_netload_conditional(df: pd.DataFrame, out_dir: str) -> dict:
    """净负荷分位的条件 VWAP"""
    regions = [
        ("JN", "price_dayahead_jn_江南", "price_realtime_jn_final_江南"),
        ("JB", "price_dayahead_jb_江北", "price_realtime_jb_final_江北"),
    ]
    nl = df["net_load"].values
    quantiles = [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    q_vals = np.nanquantile(nl, quantiles)
    results = {}

    for region, da_col, rt_col in regions:
        rows = []
        for i in range(len(quantiles) - 1):
            mask = (nl >= q_vals[i]) & (nl < q_vals[i + 1])
            if i == len(quantiles) - 2:
                mask = (nl >= q_vals[i]) & (nl <= q_vals[i + 1])
            sub = df[mask]
            e = sub["energy_mwh"].values
            da = sub[da_col].values
            rt = sub[rt_col].values
            row = {
                "bin": f"Q{int(quantiles[i]*100)}-Q{int(quantiles[i+1]*100)}",
                "nl_range_gw": f"{q_vals[i]/1000:.1f}~{q_vals[i+1]/1000:.1f}",
                "samples": len(sub),
                "da_vwap": np.nansum(da * e) / np.nansum(e) if np.nansum(e) > 0 else 0,
                "rt_vwap": np.nansum(rt * e) / np.nansum(e) if np.nansum(e) > 0 else 0,
                "da_floor_rate": (da <= FLOOR_PRICE).mean(),
                "rt_floor_rate": (rt <= FLOOR_PRICE).mean(),
            }
            rows.append(row)
        results[region] = rows

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (region, _, _) in enumerate(regions):
        ax = axes[idx]
        rows = results[region]
        x = np.arange(len(rows))
        w = 0.3
        ax.bar(x - w / 2, [r["da_vwap"] for r in rows], w, label="DA VWAP", color="#1f77b4")
        ax.bar(x + w / 2, [r["rt_vwap"] for r in rows], w, label="RT VWAP", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels([r["bin"] for r in rows], fontsize=8)
        ax.set_ylabel("元/MWh")
        ax.set_title(f"{region} 净负荷分位 vs VWAP")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(x, [r["rt_floor_rate"] * 100 for r in rows], "s--", color="#ff7f0e",
                 label="RT地板率", markersize=5)
        ax2.set_ylabel("地板率 (%)", color="#ff7f0e")
        ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "3_netload_conditional_vwap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3] {path}")
    return results


def analysis_4_intraday_revenue_transfer(df: pd.DataFrame, out_dir: str) -> dict:
    """日内时段块: 电量份额 vs 收入份额"""
    blocks = [
        ("00-06", 1, 24),
        ("06-10", 25, 40),
        ("10-14", 41, 56),
        ("14-18", 57, 72),
        ("18-22", 73, 88),
        ("22-24", 89, 96),
    ]
    regions = [
        ("JN", "price_dayahead_jn_江南", "price_realtime_jn_final_江南"),
        ("JB", "price_dayahead_jb_江北", "price_realtime_jb_final_江北"),
    ]
    months = sorted(df["month_label"].unique())
    results = {}

    for region, da_col, rt_col in regions:
        all_rows = []
        for ml in months:
            mdf = df[df["month_label"] == ml]
            total_e = mdf["energy_mwh"].sum()
            total_rev_da = (mdf[da_col] * mdf["energy_mwh"]).sum()
            total_rev_rt = (mdf[rt_col] * mdf["energy_mwh"]).sum()

            for bname, hh_lo, hh_hi in blocks:
                bmask = (mdf["hh_index"] >= hh_lo) & (mdf["hh_index"] <= hh_hi)
                bdf = mdf[bmask]
                be = bdf["energy_mwh"].sum()
                brev_da = (bdf[da_col] * bdf["energy_mwh"]).sum()
                brev_rt = (bdf[rt_col] * bdf["energy_mwh"]).sum()
                all_rows.append({
                    "month": ml,
                    "block": bname,
                    "energy_share": be / total_e if total_e > 0 else 0,
                    "da_revenue_share": brev_da / total_rev_da if total_rev_da > 0 else 0,
                    "rt_revenue_share": brev_rt / total_rev_rt if total_rev_rt > 0 else 0,
                    "da_vwap": brev_da / be if be > 0 else 0,
                    "rt_vwap": brev_rt / be if be > 0 else 0,
                })
        results[region] = all_rows

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for idx, (region, _, _) in enumerate(regions):
        all_rows = results[region]
        for m_idx, ml in enumerate(months):
            ax = axes[m_idx // 2][idx + (m_idx % 2) * 0 if len(months) <= 2 else idx]
            if len(months) > 2:
                ax = axes[m_idx // 2][(m_idx % 2) * 1 if idx == 0 else (m_idx % 2) * 1]

    plt.close(fig)

    fig, axes = plt.subplots(len(months), 2, figsize=(14, 4 * len(months)))
    if len(months) == 1:
        axes = axes.reshape(1, -1)
    for idx, (region, _, _) in enumerate(regions):
        all_rows = results[region]
        for m_idx, ml in enumerate(months):
            ax = axes[m_idx][idx]
            m_rows = [r for r in all_rows if r["month"] == ml]
            x = np.arange(len(m_rows))
            w = 0.25
            ax.bar(x - w, [r["energy_share"] * 100 for r in m_rows], w,
                   label="电量份额", color="#2ca02c")
            ax.bar(x, [r["da_revenue_share"] * 100 for r in m_rows], w,
                   label="DA收入份额", color="#1f77b4")
            ax.bar(x + w, [r["rt_revenue_share"] * 100 for r in m_rows], w,
                   label="RT收入份额", color="#d62728")
            ax.set_xticks(x)
            ax.set_xticklabels([r["block"] for r in m_rows], fontsize=8)
            ax.set_ylabel("%")
            ax.set_title(f"{region} {ml} 日内收入转移")
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "4_intraday_revenue_transfer.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4] {path}")
    return results


def analysis_5_source_implied_revenue(df: pd.DataFrame, out_dir: str) -> dict:
    """分电源类型隐含收入估算"""
    regions = [
        ("JN", "price_dayahead_jn_江南", "price_realtime_jn_final_江南"),
        ("JB", "price_dayahead_jb_江北", "price_realtime_jb_final_江北"),
    ]
    months = sorted(df["month_label"].unique())
    results = {}

    for region, da_col, rt_col in regions:
        rows = []
        for ml in months:
            mdf = df[df["month_label"] == ml]
            rt = mdf[rt_col].values
            load = mdf["load_actual_total_汇总"].values
            re = mdf["re_total"].values
            receive = mdf.get("receive_actual_huadong_华东", pd.Series(0)).values
            gas_jn = mdf.get("gas_actual_江南", pd.Series(0)).values
            gas_jb = mdf.get("gas_actual_江北", pd.Series(0)).values
            gas = gas_jn + gas_jb

            thermal_residual = np.maximum(load - re - receive, 0)

            re_energy = re * INTERVAL_HOURS
            gas_energy = gas * INTERVAL_HOURS
            thermal_energy = thermal_residual * INTERVAL_HOURS
            receive_energy = receive * INTERVAL_HOURS

            re_revenue = (re * INTERVAL_HOURS * rt).sum() / 1e6
            gas_revenue = (gas * INTERVAL_HOURS * rt).sum() / 1e6
            receive_revenue = (receive * INTERVAL_HOURS * rt).sum() / 1e6
            total_revenue = (load * INTERVAL_HOURS * rt).sum() / 1e6

            re_vwap = re_revenue * 1e6 / re_energy.sum() if re_energy.sum() > 0 else 0
            gas_vwap = gas_revenue * 1e6 / gas_energy.sum() if gas_energy.sum() > 0 else 0

            rows.append({
                "month": ml,
                "re_gwh": re_energy.sum() / 1000,
                "gas_gwh": gas_energy.sum() / 1000,
                "receive_gwh": receive_energy.sum() / 1000,
                "thermal_residual_gwh": thermal_energy.sum() / 1000,
                "re_revenue_m": re_revenue,
                "gas_revenue_m": gas_revenue,
                "receive_revenue_m": receive_revenue,
                "total_revenue_m": total_revenue,
                "re_vwap": re_vwap,
                "gas_vwap": gas_vwap,
                "re_revenue_share": re_revenue / total_revenue if total_revenue > 0 else 0,
                "re_energy_share": re_energy.sum() / (load * INTERVAL_HOURS).sum() if (load * INTERVAL_HOURS).sum() > 0 else 0,
            })
        results[region] = rows

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (region, _, _) in enumerate(regions):
        ax = axes[idx]
        rows = results[region]
        x = np.arange(len(rows))
        w = 0.3
        ax.bar(x - w / 2, [r["re_vwap"] for r in rows], w, label="新能源 VWAP", color="#2ca02c")
        ax.bar(x + w / 2, [r["gas_vwap"] for r in rows], w, label="气电 VWAP", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels([r["month"] for r in rows])
        ax.set_ylabel("元/MWh")
        ax.set_title(f"{region} 分电源隐含 VWAP (按实时价)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "5_source_implied_vwap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5] {path}")
    return results


def analysis_6_da_floor_rt_conditional(df: pd.DataFrame, out_dir: str) -> dict:
    """当日前=地板价时, 同一天实时价的条件分布"""
    regions = [
        ("JN", "price_dayahead_jn_江南", "price_realtime_jn_final_江南"),
        ("JB", "price_dayahead_jb_江北", "price_realtime_jb_final_江北"),
    ]
    results = {}

    for region, da_col, rt_col in regions:
        da = df[da_col].values
        rt = df[rt_col].values
        e = df["energy_mwh"].values

        da_floor = da <= FLOOR_PRICE
        da_normal = (da > FLOOR_PRICE) & (da < SPIKE_PRICE)
        da_spike = da >= SPIKE_PRICE

        row = {
            "da_floor_count": int(da_floor.sum()),
            "da_floor_pct": float(da_floor.mean()),
            "da_floor_rt_mean": float(rt[da_floor].mean()) if da_floor.sum() > 0 else 0,
            "da_floor_rt_median": float(np.median(rt[da_floor])) if da_floor.sum() > 0 else 0,
            "da_floor_rt_also_floor": float((rt[da_floor] <= FLOOR_PRICE).mean()) if da_floor.sum() > 0 else 0,
            "da_floor_rt_over200": float((rt[da_floor] > 200).mean()) if da_floor.sum() > 0 else 0,
            "da_floor_rt_over400": float((rt[da_floor] >= SPIKE_PRICE).mean()) if da_floor.sum() > 0 else 0,
            "da_floor_rt_da_spread_mean": float((rt[da_floor] - da[da_floor]).mean()) if da_floor.sum() > 0 else 0,
            "da_floor_rt_vwap": float(np.nansum(rt[da_floor] * e[da_floor]) / np.nansum(e[da_floor])) if da_floor.sum() > 0 else 0,
            "da_floor_energy_share": float(e[da_floor].sum() / e.sum()),
            "da_floor_rt_revenue_share": float((rt[da_floor] * e[da_floor]).sum() / (rt * e).sum()),
            "da_normal_rt_mean": float(rt[da_normal].mean()) if da_normal.sum() > 0 else 0,
            "da_normal_rt_vwap": float(np.nansum(rt[da_normal] * e[da_normal]) / np.nansum(e[da_normal])) if da_normal.sum() > 0 else 0,
            "da_spike_rt_mean": float(rt[da_spike].mean()) if da_spike.sum() > 0 else 0,
        }
        results[region] = row

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (region, da_col, rt_col) in enumerate(regions):
        ax = axes[idx]
        da = df[da_col].values
        rt = df[rt_col].values
        da_floor = da <= FLOOR_PRICE

        if da_floor.sum() > 0:
            rt_at_floor = rt[da_floor]
            bins = [0, 50, 100, 200, 300, 400, 600, 1000, 1500]
            ax.hist(rt_at_floor, bins=bins, color="#d62728", alpha=0.7, edgecolor="black")
            ax.axvline(np.median(rt_at_floor), color="blue", linestyle="--",
                       label=f"中位数={np.median(rt_at_floor):.0f}")
            ax.axvline(np.mean(rt_at_floor), color="green", linestyle="--",
                       label=f"均值={np.mean(rt_at_floor):.0f}")
        ax.set_xlabel("实时价 (元/MWh)")
        ax.set_ylabel("样本数")
        ax.set_title(f"{region} 当日前=地板价时的实时价分布 (n={da_floor.sum()})")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "6_da_floor_rt_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [6] {path}")
    return results


def print_summary(r1, r2, r3, r4, r5, r6):
    print("\n" + "=" * 70)
    print("  电量加权价格分析 (VWAP) 结果摘要")
    print("=" * 70)

    print("\n--- 1. 简单均价 vs VWAP (按月) ---")
    for region, rows in r1.items():
        print(f"\n  {region}:")
        print(f"  {'月份':<10s} {'DA简均':>8s} {'DA-VWAP':>8s} {'RT简均':>8s} {'RT-VWAP':>8s} {'日均GWh':>8s}")
        for r in rows:
            print(f"  {r['month']:<10s} {r['da_simple_avg']:>8.1f} {r['da_vwap']:>8.1f} "
                  f"{r['rt_simple_avg']:>8.1f} {r['rt_vwap']:>8.1f} {r['total_gwh']/r['days']:>8.1f}")

    print("\n--- 2. 地板价三重占比 ---")
    for region, rows in r2.items():
        print(f"\n  {region}:")
        print(f"  {'月份':<10s} {'类型':>4s} {'时段占比':>8s} {'电量占比':>8s} {'收入占比':>8s} {'地板均荷GW':>10s}")
        for r in rows:
            print(f"  {r['month']:<10s} {r['type']:>4s} {r['time_share_floor']*100:>7.1f}% "
                  f"{r['energy_share_floor']*100:>7.1f}% {r['revenue_share_floor']*100:>7.1f}% "
                  f"{r['floor_avg_load_gw']:>10.1f}")

    print("\n--- 3. 净负荷条件 VWAP ---")
    for region, rows in r3.items():
        print(f"\n  {region}:")
        print(f"  {'分位':>10s} {'净负荷GW':>14s} {'DA-VWAP':>8s} {'RT-VWAP':>8s} {'RT地板率':>8s}")
        for r in rows:
            print(f"  {r['bin']:>10s} {r['nl_range_gw']:>14s} {r['da_vwap']:>8.1f} "
                  f"{r['rt_vwap']:>8.1f} {r['rt_floor_rate']*100:>7.1f}%")

    print("\n--- 6. 日前地板价时段的实时价条件分析 ---")
    for region, r in r6.items():
        print(f"\n  {region}:")
        print(f"  日前地板价样本: {r['da_floor_count']} ({r['da_floor_pct']*100:.1f}%)")
        print(f"  对应实时价均值: {r['da_floor_rt_mean']:.1f}  中位数: {r['da_floor_rt_median']:.1f}")
        print(f"  实时也是地板: {r['da_floor_rt_also_floor']*100:.1f}%")
        print(f"  实时>200: {r['da_floor_rt_over200']*100:.1f}%")
        print(f"  实时>400: {r['da_floor_rt_over400']*100:.1f}%")
        print(f"  实时-日前价差均值: {r['da_floor_rt_da_spread_mean']:.1f}")
        print(f"  地板时段电量占比: {r['da_floor_energy_share']*100:.1f}%")
        print(f"  地板时段RT收入占比: {r['da_floor_rt_revenue_share']*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-parquet", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/7] Loading data ...")
    df = load_data(args.input_parquet)
    print(f"  {len(df)} rows, {df['trade_date'].nunique()} days")

    print("[2/7] VWAP vs 简单均价 ...")
    r1 = analysis_1_vwap_by_month(df, args.output_dir)

    print("[3/7] 地板价三重占比 ...")
    r2 = analysis_2_floor_shares(df, args.output_dir)

    print("[4/7] 净负荷条件 VWAP ...")
    r3 = analysis_3_netload_conditional(df, args.output_dir)

    print("[5/7] 日内收入转移 ...")
    r4 = analysis_4_intraday_revenue_transfer(df, args.output_dir)

    print("[6/7] 分电源隐含收入 ...")
    r5 = analysis_5_source_implied_revenue(df, args.output_dir)

    print("[7/7] 日前地板→实时条件分析 ...")
    r6 = analysis_6_da_floor_rt_conditional(df, args.output_dir)

    print_summary(r1, r2, r3, r4, r5, r6)

    summary = {"vwap_monthly": r1, "floor_shares": r2, "netload_cond": r3,
               "source_revenue": r5, "da_floor_rt": r6}
    with open(os.path.join(args.output_dir, "vwap_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Summary JSON -> {args.output_dir}/vwap_analysis_summary.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
