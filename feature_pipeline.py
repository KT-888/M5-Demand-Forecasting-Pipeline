"""
pipeline/feature_pipeline.py
=============================
特徴量生成パイプライン（学習・推論共通）。

【修正内容 (v2)】
・・・
"""

import pandas as pd

from config import LAGS, ROLLING_WINDOWS
from features.calendar import add_calendar_features
from features.lag      import create_lags
from features.rolling  import create_rolling_features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    【学習時に使う】全データの特徴量をまとめて一括生成する。

    ① 日付順に並び替える
    ② カレンダー特徴量（曜日・祝日・月末フラグなど）を追加  → features/calendar.py
    ③ ラグ特徴量（N日前の売上）を追加                     → features/lag.py
    ④ ローリング特徴量（移動平均・標準偏差）を追加          → features/rolling.py
       price / stock の欠損補完もここで行う
    """
    df = df.copy()

    df = (
        df.sort_values(["store", "product", "date"])
          .reset_index(drop=True)
    )

    df = add_calendar_features(df)
    df = create_lags(df)
    df = create_rolling_features(df)

    return df








def get_feature_col_names(df: pd.DataFrame) -> list:
    """
    モデルへの入力として使う列名リストを返す。

    除外する列: store / product / date / sales
    残りの列がすべてモデルへの入力特徴量として使われる。
    """
    exclude_cols = {"store", "product", "date", "sales"}

    return [
        col
        for col in df.columns
        if col not in exclude_cols
    ]
