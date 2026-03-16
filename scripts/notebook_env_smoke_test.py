#!/usr/bin/env python3
"""Notebook 环境快速自检：版本 + Parquet 读取。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parquet_path = (
        project_root
        / "warehouse"
        / "feature_ready"
        / "V0"
        / "power_market_feature_ready_wide.parquet"
    )

    print(f"python={sys.version.split()[0]}")
    print(f"numpy={np.__version__}")
    print(f"pandas={pd.__version__}")
    print(f"pyarrow={pa.__version__}")

    if not parquet_path.exists():
        print(f"[WARN] parquet not found: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    print(f"[OK] read_parquet: shape={df.shape}")
    print(f"[OK] timestamp column: {'timestamp' in df.columns}")


if __name__ == "__main__":
    main()

