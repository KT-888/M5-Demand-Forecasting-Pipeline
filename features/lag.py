"""
features/lag.py
===============
ラグ特徴量の生成
 
設計方針:
    - shift(lag) で未来リークを完全排除
    - lag_X_avail でデータ有無をマスク変数として記録
      → モデルが「新商品でデータなし」を明示的に学習できる
"""
 
import numpy as np
import pandas as pd
from config import LAGS
 
 
def create_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    sales と onpromotion（存在する場合）のラグ特徴量を生成。
 
    Returns
    -------
    lag_1, lag_7, lag_14              : 過去N日の売上
    lag_1_avail, ...                  : ラグデータが存在するか（1/0）
    promo_lag_1, ... promo_lag_14     : プロモのラグ（onpromotion がある場合）
    promo_lag_1_avail, ...
    """
    df = df.sort_values(["store", "product", "date"]).copy()
    g  = df.groupby(["store", "product"])
 
    for lag in LAGS:
        col = f"lag_{lag}"
        df[col] = g["sales"].shift(lag)
        # ★ 修正: astype(int)（64bit）→ astype(np.int8)（8bit）に統一
        #   feature_pipeline.py の avail フラグと型を揃える。
        #   0 か 1 しか入らない列に 64bit は不要でメモリの無駄。
        df[f"{col}_avail"] = df[col].notna().astype(np.int8)
 
    if "onpromotion" in df.columns:
        for lag in LAGS:
            col = f"promo_lag_{lag}"
            df[col] = g["onpromotion"].shift(lag)
            df[f"{col}_avail"] = df[col].notna().astype(np.int8)  # ★ 同上
 
    return df