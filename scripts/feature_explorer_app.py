from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st


DEFAULT_DATA_DIR = Path("warehouse/feature_ready/V0")
KEY_COLS = {"timestamp", "trade_date", "hh_index"}


@st.cache_data(show_spinner=False)
def list_parquet_files(data_dir: str) -> List[str]:
    p = Path(data_dir)
    if not p.exists():
        return []
    return sorted([str(x) for x in p.glob("*.parquet")])


@st.cache_data(show_spinner=True)
def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def split_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    qflag_cols = [c for c in df.columns if str(c).endswith("_qflag")]
    value_cols = [c for c in df.columns if c not in KEY_COLS and c not in qflag_cols]
    return value_cols, qflag_cols


def build_missing_stats(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for c in value_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        miss = int(s.isna().sum())
        rows.append(
            {
                "feature": c,
                "missing_count": miss,
                "missing_ratio": (miss / total) if total else 0.0,
                "non_null_count": int(total - miss),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["missing_ratio", "feature"], ascending=[False, True]).reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Parquet 特征查看器", layout="wide")
    st.title("Parquet 入库查看与特征可视化")
    st.caption("支持 V0 宽表浏览、关键统计、时段筛选、多特征曲线绘制与导出。")

    with st.sidebar:
        st.header("数据源")
        data_dir = st.text_input("Parquet 目录", str(DEFAULT_DATA_DIR))
        files = list_parquet_files(data_dir)
        if not files:
            st.warning("目录下未找到 parquet 文件。")
            st.stop()
        selected_file = st.selectbox("选择 parquet 文件", files, index=0)
        st.markdown("---")
        st.header("图表选项")
        agg_mode = st.selectbox("时间聚合", ["原始15分钟", "按日均值", "按日最大值", "按日最小值"], index=0)
        plot_mode = st.radio("绘图方式", ["同图多曲线", "按特征分图"], index=0)

    df = load_parquet(selected_file)
    if df.empty:
        st.error("文件为空，无法展示。")
        st.stop()
    if "timestamp" not in df.columns:
        st.error("缺少 `timestamp` 列，当前工具无法进行时序可视化。")
        st.stop()

    value_cols, qflag_cols = split_feature_columns(df)
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总行数", f"{len(df):,}")
    c2.metric("总列数", f"{len(df.columns):,}")
    c3.metric("特征列", f"{len(value_cols):,}")
    c4.metric("QFlag列", f"{len(qflag_cols):,}")

    st.write(f"时间范围：`{ts_min}`  到  `{ts_max}`")

    with st.expander("缺失率概览（Top 20）", expanded=False):
        miss_df = build_missing_stats(df, value_cols)
        if miss_df.empty:
            st.info("无可统计特征列。")
        else:
            st.dataframe(miss_df.head(20), use_container_width=True)

    st.subheader("特征曲线")
    default_features = value_cols[: min(3, len(value_cols))]
    selected_features = st.multiselect("选择一条或多条特征", value_cols, default=default_features)
    if not selected_features:
        st.info("请先选择特征。")
        st.stop()

    date_range = st.slider(
        "选择时间范围",
        min_value=ts_min.to_pydatetime(),
        max_value=ts_max.to_pydatetime(),
        value=(ts_min.to_pydatetime(), ts_max.to_pydatetime()),
        format="YYYY-MM-DD HH:mm",
    )
    start_ts, end_ts = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])

    sub = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()
    if sub.empty:
        st.warning("该时间范围无数据。")
        st.stop()

    if agg_mode != "原始15分钟":
        sub["trade_date"] = pd.to_datetime(sub["timestamp"], errors="coerce").dt.normalize()
        agg_func = {
            "按日均值": "mean",
            "按日最大值": "max",
            "按日最小值": "min",
        }[agg_mode]
        grp = sub.groupby("trade_date")[selected_features].agg(agg_func).reset_index()
        grp = grp.rename(columns={"trade_date": "timestamp"})
        plot_df = grp
    else:
        plot_df = sub[["timestamp"] + selected_features].copy()

    if plot_mode == "同图多曲线":
        st.line_chart(plot_df.set_index("timestamp"), height=420)
    else:
        for feat in selected_features:
            st.markdown(f"**{feat}**")
            st.line_chart(plot_df.set_index("timestamp")[[feat]], height=220)

    with st.expander("筛选结果预览与导出", expanded=True):
        preview_cols = ["timestamp"] + selected_features
        include_qflag = st.checkbox("包含对应 qflag 列", value=True)
        if include_qflag:
            for feat in selected_features:
                q = f"{feat}_qflag"
                if q in df.columns and q not in preview_cols:
                    preview_cols.append(q)

        export_df = sub[preview_cols].copy()
        st.dataframe(export_df.head(200), use_container_width=True)

        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载当前筛选结果 CSV",
            data=io.BytesIO(csv_bytes),
            file_name="feature_selection_export.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

