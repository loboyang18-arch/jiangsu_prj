"""
基于统一频率时间序列 parquet 做特征工程，输出带 y 与滞后/滚动/日历特征的特征表。

- 必须提供 --freq，与 build_dataset 输出频率一致；未提供时拒绝运行，避免“按行位移”的歧义。
- 标签 y = target(t + horizon_steps)，horizon 可为步数或时间跨度（如 1 或 15min/1H）。
- 支持 --include-regex / --exclude-regex / --max-missing-rate 控制参与特征构建的列。
- 支持 --task-mode + feature_registry 白黑名单过滤，减少价格任务泄漏风险。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

import numpy as np
import pandas as pd


PRICE_HINT_RE = re.compile(r"(电价|价格|均价|LMP|出清价)", re.IGNORECASE)
SYSTEM_PASSTHROUGH_FEATURES = {"hh_index"}


# -----------------------------------------------------------------------------
# 目标列选取、滞后/滚动特征、频率与 horizon 解析、特征列筛选
# -----------------------------------------------------------------------------

def pick_target_column(df: pd.DataFrame, target: Optional[str]) -> str:
    """若指定 target 则校验并返回；否则优先选名称匹配 PRICE_HINT_RE 的列，再退化为首列数值列。"""
    if target:
        if target not in df.columns:
            raise ValueError(f"target_col 不存在：{target}")
        return target

    candidates = [c for c in df.columns if c != "timestamp" and PRICE_HINT_RE.search(str(c))]
    if candidates:
        return candidates[0]

    # fallback: first numeric col
    num = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        raise ValueError("找不到可用的数值列作为目标（请在 build_dataset 输出中检查列名）。")
    return num[0]


def make_lag_features(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    """为 cols 中每列按 lags 中每个步数生成 shift(lag) 列，命名 {col}__lag{lag}。"""
    out = df.copy()
    for c in cols:
        for lag in lags:
            out[f"{c}__lag{lag}"] = out[c].shift(lag)
    return out


def make_rolling_features(df: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
    """为 cols 中每列按 windows 中每个窗口生成 roll_mean 与 roll_std 列。"""
    out = df.copy()
    for c in cols:
        s = out[c]
        for w in windows:
            out[f"{c}__roll_mean{w}"] = s.rolling(w, min_periods=max(2, w // 3)).mean()
            out[f"{c}__roll_std{w}"] = s.rolling(w, min_periods=max(2, w // 3)).std()
    return out


def _parse_freq(freq: str) -> str:
    """校验并返回合法 pandas 频率字符串。"""
    try:
        pd.tseries.frequencies.to_offset(freq)
        return freq
    except Exception as e:
        raise ValueError(f"非法 freq：{freq}（示例：15min / 1H / 1D）") from e


def _parse_horizon(horizon: str, freq: str) -> int:
    """将 horizon 解析为步数：纯数字即步数；否则按时间跨度（如 15min/1H）除以 freq 得到步数。"""
    h = horizon.strip()
    if re.fullmatch(r"\d+", h):
        return int(h)
    try:
        td = pd.to_timedelta(h)
    except Exception as e:
        raise ValueError(f"非法 horizon：{h}（示例：1 或 15min / 1H / 1D）") from e
    off = pd.tseries.frequencies.to_offset(freq)
    # convert offset to timedelta best-effort
    base = pd.to_timedelta(off.n, unit=off.name) if hasattr(off, "name") and off.name else pd.to_timedelta(off.delta)
    steps = int(round(td / base))
    if steps <= 0:
        raise ValueError("horizon 必须 > 0")
    return steps


def _compile_optional_regex(p: str) -> Optional[Pattern[str]]:
    """空字符串返回 None，否则返回编译后的正则。"""
    p = (p or "").strip()
    if not p:
        return None
    return re.compile(p)


def _select_feature_cols(
    df: pd.DataFrame,
    include_re: Optional[Pattern[str]],
    exclude_re: Optional[Pattern[str]],
    max_missing_rate: float,
) -> List[str]:
    """在数值列中按 include/exclude 正则与缺失率上限筛选，返回列名列表。"""
    cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
    out: List[str] = []
    for c in cols:
        name = str(c)
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        miss = float(df[c].isna().mean())
        if miss > max_missing_rate:
            continue
        out.append(c)
    return out


def _load_registry(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_feature_key(col: str, registry_keys: List[str]) -> Optional[str]:
    # 优先最长匹配，避免同前缀冲突
    for k in sorted(registry_keys, key=len, reverse=True):
        if col == k or col.startswith(f"{k}_"):
            return k
    return None


def _apply_registry_policy(
    candidate_cols: List[str],
    target_col: str,
    task_mode: str,
    registry: Dict[str, Any],
    allow_unknown_features: bool,
) -> Tuple[List[str], Dict[str, List[str]]]:
    features = registry.get("features", {})
    policies = registry.get("forecast_task_policies", {})
    task_policy = policies.get(task_mode, {})
    forbidden_prefixes = set(task_policy.get("forbidden_feature_prefixes", []))
    drop_unknown = bool(task_policy.get("drop_unknown_features", False))
    if allow_unknown_features:
        drop_unknown = False

    keys = list(features.keys())
    kept: List[str] = []
    dropped = {"unknown": [], "not_whitelisted": [], "forbidden": []}
    for c in candidate_cols:
        if c == target_col:
            kept.append(c)
            continue
        if c in SYSTEM_PASSTHROUGH_FEATURES:
            kept.append(c)
            continue
        k = _extract_feature_key(str(c), keys)
        if k is None:
            if drop_unknown:
                dropped["unknown"].append(str(c))
                continue
            kept.append(c)
            continue

        entry = features.get(k, {})
        whitelist = entry.get("task_whitelist", ["dayahead", "realtime"])
        if task_mode not in whitelist and "all" not in whitelist:
            dropped["not_whitelisted"].append(str(c))
            continue

        if any(k == fp or k.startswith(f"{fp}_") for fp in forbidden_prefixes):
            dropped["forbidden"].append(str(c))
            continue
        kept.append(c)

    return kept, dropped


def _assert_no_future_leakage(columns: List[str], target_col: str) -> None:
    # 特征名中出现 lead/future 视为硬错误
    bad = [c for c in columns if re.search(r"(lead|future|t\+)", str(c), re.IGNORECASE)]
    if bad:
        raise ValueError(f"检测到潜在未来信息列，拒绝继续：{bad[:10]}")
    if any(c == target_col for c in columns):
        raise ValueError("检测到当前时点目标列直接进入特征，存在泄漏风险。")


def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", (hhmm or "").strip())
    if not m:
        raise ValueError(f"非法时刻字符串：{hhmm}，应为 HH:MM")
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        raise ValueError(f"非法时刻字符串：{hhmm}，小时需在 0-23，分钟需在 0-59")
    return hh, mm


def _build_decision_ts(ts: pd.Series, policy: str) -> pd.Series:
    p = (policy or "asof_timestamp").strip()
    if p == "asof_timestamp":
        return ts
    if p in {"dayahead_dminus1_0930", "dayahead_dminus1_1700"}:
        hhmm = "09:30" if p.endswith("0930") else "17:00"
        hh, mm = _parse_hhmm(hhmm)
        # 以 value 的交付日（trade_date）为锚：该日对应 D 日，决策时点为 D-1 固定时刻
        return ts.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=hh, minutes=mm)
    raise ValueError(f"未知 decision-time-policy：{p}")


def _resolve_release_rule(metric_key: str, registry: Dict[str, Any]) -> Optional[str]:
    features = registry.get("features", {})
    entry = features.get(metric_key, {})
    rule = entry.get("availability_rule")
    if rule:
        return str(rule)
    by_prefix = registry.get("availability_by_metric_prefix", {})
    for p in sorted(by_prefix.keys(), key=len, reverse=True):
        if metric_key == p or metric_key.startswith(f"{p}_"):
            return str(by_prefix[p])
    return None


def _calc_publish_ts_for_rule(ts: pd.Series, rule_id: str, registry: Dict[str, Any]) -> pd.Series:
    rules = registry.get("availability_rules", {})
    rule = rules.get(rule_id, {})
    if not rule:
        # 未配置规则时，默认视作“与值时刻同时可得”
        return ts
    t = str(rule.get("type", "")).strip()
    if t == "fixed_day_offset":
        day_offset = int(rule.get("day_offset", 0))
        hh, mm = _parse_hhmm(str(rule.get("time_hhmm", "00:00")))
        return ts.dt.normalize() + pd.Timedelta(days=day_offset) + pd.Timedelta(hours=hh, minutes=mm)
    if t == "delay_from_value_ts":
        delay_minutes = int(rule.get("delay_minutes", 0))
        return ts + pd.Timedelta(minutes=delay_minutes)
    raise ValueError(f"未知 availability rule type: {t} (rule_id={rule_id})")


def _build_publish_ts_map(
    feature_cols: List[str],
    ts: pd.Series,
    registry: Dict[str, Any],
) -> Dict[str, pd.Series]:
    keys = list(registry.get("features", {}).keys())
    out: Dict[str, pd.Series] = {}
    for c in feature_cols:
        k = _extract_feature_key(str(c), keys)
        if k is None:
            out[c] = ts
            continue
        rule_id = _resolve_release_rule(k, registry)
        out[c] = _calc_publish_ts_for_rule(ts, rule_id, registry) if rule_id else ts
    return out


# -----------------------------------------------------------------------------
# 入口：读 parquet、重采样、选目标与特征列、构建 y/lag/rolling/日历、写 parquet 与 meta
# -----------------------------------------------------------------------------

def main() -> int:
    """解析参数、生成特征表与可选 meta JSON，返回 0。"""
    ap = argparse.ArgumentParser(description="Feature engineering for unified time-series parquet.")
    ap.add_argument("--input-parquet", required=True, help="Input parquet from build_dataset.py")
    ap.add_argument("--output-parquet", required=True, help="Output parquet for features table")
    ap.add_argument("--output-meta", default="", help="Optional meta json output")
    ap.add_argument("--target-col", default="", help="Target column name (default: auto)")
    ap.add_argument("--freq", default="", help="统一频率（必须与 build_dataset 的 --freq 一致，例如 15min/1H/1D）")
    ap.add_argument("--horizon", default="1", help="预测步长：'1' 表示 1 步；或 '15min/1H/1D' 这种时间跨度（需要 --freq）")
    ap.add_argument("--lags", default="1,2,4,8", help="Comma-separated lags (steps) for lag features")
    ap.add_argument("--roll-windows", default="4,8,24", help="Comma-separated rolling windows (steps)")
    ap.add_argument("--include-regex", default="", help="只保留匹配该正则的特征列名（可选）")
    ap.add_argument("--exclude-regex", default="", help="排除匹配该正则的特征列名（可选）")
    ap.add_argument("--max-missing-rate", type=float, default=0.95, help="特征列缺失率上限（默认 0.95）")
    ap.add_argument("--task-mode", choices=["generic", "dayahead", "realtime"], default="generic", help="任务模式：generic/dayahead/realtime")
    ap.add_argument("--registry-json", default="", help="feature_registry.json 路径（task-mode 非 generic 时建议提供）")
    ap.add_argument("--allow-unknown-features", action="store_true", help="task-mode 下是否允许 registry 未登记特征")
    ap.add_argument("--allow-current-target-feature", action="store_true", help="允许把 target(t) 直接作为特征（默认关闭，防泄漏）")
    ap.add_argument("--disable-target-history", action="store_true", help="禁用 target 历史滞后/滚动特征")
    ap.add_argument(
        "--decision-time-policy",
        choices=["asof_timestamp", "dayahead_dminus1_0930", "dayahead_dminus1_1700"],
        default="asof_timestamp",
        help="样本决策时点策略：asof_timestamp(默认) / 日前D-1 09:30 / 日前D-1 17:00",
    )
    ap.add_argument("--drop-na", action="store_true", help="Drop rows with any NA in X/y")
    args = ap.parse_args()

    inp = os.path.abspath(args.input_parquet)
    outp = os.path.abspath(args.output_parquet)
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    df = pd.read_parquet(inp)
    if "timestamp" not in df.columns:
        # backward compatibility: older build_dataset versions used __timestamp
        if "__timestamp" in df.columns:
            df = df.rename(columns={"__timestamp": "timestamp"})
        else:
            raise ValueError("输入数据缺少 timestamp 列。")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if args.freq:
        freq = _parse_freq(args.freq)
        df = df.set_index("timestamp").resample(freq).mean().asfreq(freq).reset_index()

    target = pick_target_column(df, args.target_col or None)

    # ensure numeric target
    if not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = pd.to_numeric(df[target], errors="coerce")

    include_re = _compile_optional_regex(args.include_regex)
    exclude_re = _compile_optional_regex(args.exclude_regex)
    feature_base_cols = _select_feature_cols(
        df,
        include_re=include_re,
        exclude_re=exclude_re,
        max_missing_rate=float(args.max_missing_rate),
    )
    if target not in feature_base_cols:
        feature_base_cols.append(target)

    registry_path = args.registry_json.strip() or os.path.join(os.path.dirname(__file__), "feature_registry.json")
    registry = _load_registry(registry_path) if args.task_mode != "generic" else {}
    registry_drop_report: Dict[str, List[str]] = {"unknown": [], "not_whitelisted": [], "forbidden": []}
    if args.task_mode != "generic":
        feature_base_cols, registry_drop_report = _apply_registry_policy(
            candidate_cols=feature_base_cols,
            target_col=target,
            task_mode=args.task_mode,
            registry=registry,
            allow_unknown_features=bool(args.allow_unknown_features),
        )
        if target not in feature_base_cols:
            feature_base_cols.append(target)

    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    wins = [int(x) for x in args.roll_windows.split(",") if x.strip()]

    feat = df[["timestamp"] + feature_base_cols].copy()
    decision_ts = _build_decision_ts(feat["timestamp"], args.decision_time_policy)

    if args.freq:
        horizon_steps = _parse_horizon(args.horizon, args.freq)
        label_desc = f"y = target(t+{horizon_steps} step @ {args.freq})"
    else:
        # Without a declared uniform frequency, refuse to pretend shift(-1) is a time-aware horizon.
        raise ValueError("必须提供 --freq（先把时间频率固化），才能定义严格的 horizon/lag/rolling。")

    feat["y"] = feat[target].shift(-horizon_steps)

    # 默认保留 target 历史特征（lag/rolling），但不直接把 target(t) 作为模型输入，避免泄漏。
    x_cols = list(feature_base_cols)
    if args.disable_target_history and target in x_cols:
        x_cols.remove(target)

    # as-of 可得性过滤（规范化）：按 registry 发布时间规则裁剪 base 与 lag 特征。
    # 说明：rolling 特征在“已裁剪的时序”上计算，确保不会引入明确晚于决策时点的值。
    publish_ts_map = _build_publish_ts_map(x_cols, feat["timestamp"], registry if args.task_mode != "generic" else {})
    for c in x_cols:
        pub = publish_ts_map.get(c, feat["timestamp"])
        feat[c] = feat[c].where(pub <= decision_ts)

    feat = make_lag_features(feat, x_cols, lags=lags)
    for c in x_cols:
        pub = publish_ts_map.get(c, feat["timestamp"])
        for lag in lags:
            lag_name = f"{c}__lag{lag}"
            lag_pub = pub.shift(lag)
            feat[lag_name] = feat[lag_name].where(lag_pub <= decision_ts)
    feat = make_rolling_features(feat, x_cols, windows=wins)

    if not args.allow_current_target_feature and target in feat.columns:
        feat = feat.drop(columns=[target])

    # calendar features
    ts = feat["timestamp"]
    feat["dow"] = ts.dt.dayofweek.astype("int16")
    feat["hour"] = ts.dt.hour.astype("int16")
    feat["month"] = ts.dt.month.astype("int16")

    # keep columns order: timestamp, y, X...
    keep = ["timestamp", "y"] + [c for c in feat.columns if c not in {"timestamp", "y"}]
    feat = feat[keep]
    _assert_no_future_leakage([c for c in feat.columns if c not in {"timestamp", "y"}], target_col=target)

    if args.drop_na:
        feat = feat.dropna().reset_index(drop=True)

    feat.to_parquet(outp, index=False)

    if args.output_meta:
        meta_path = os.path.abspath(args.output_meta)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        meta = {
            "input_parquet": inp,
            "output_parquet": outp,
            "target_col_t": target,
            "label": label_desc,
            "freq": args.freq,
            "horizon_steps": horizon_steps,
            "lags": lags,
            "rolling_windows": wins,
            "include_regex": args.include_regex,
            "exclude_regex": args.exclude_regex,
            "max_missing_rate": float(args.max_missing_rate),
            "task_mode": args.task_mode,
            "registry_json": registry_path if args.task_mode != "generic" else "",
            "registry_dropped_unknown_count": len(registry_drop_report.get("unknown", [])),
            "registry_dropped_not_whitelisted_count": len(registry_drop_report.get("not_whitelisted", [])),
            "registry_dropped_forbidden_count": len(registry_drop_report.get("forbidden", [])),
            "allow_current_target_feature": bool(args.allow_current_target_feature),
            "disable_target_history": bool(args.disable_target_history),
            "decision_time_policy": args.decision_time_policy,
            "rows": int(feat.shape[0]),
            "cols": int(feat.shape[1]),
        }
        with open(meta_path, "w", encoding="utf-8") as w:
            json.dump(meta, w, ensure_ascii=False, indent=2)

    print(f"Target(t): {target}")
    if args.task_mode != "generic":
        print(
            "Registry filter drop counts:",
            {k: len(v) for k, v in registry_drop_report.items()},
        )
    print(f"Wrote features: {outp} ({feat.shape[0]} rows, {feat.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

