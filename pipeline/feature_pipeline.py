"""
pipeline/feature_pipeline.py
=============================
特徴量生成パイプライン（学習・推論共通）。

【修正内容 (v2)】
    ① update_features_for_one_day() の二重管理を解消。
       ロジックの実体を features/ に一本化し、build_features() を呼ぶだけにした。

    ② price/stock の中央値補完を全履歴 df から計算するよう修正。
       旧実装は df 全体の median を使っていたが、build_features() 経由にすると
       lookback 期間（49日分）の median になってしまうため、
       事前に全履歴 df から median を計算して temp_df の欠損を埋めてから
       build_features() に渡す方式に変更した。
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


def update_features_for_one_day(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    【予測時に使う】target_date の1日分だけ特徴量を計算して更新する。

    速度の工夫（lookback_days）:
        ラグ・ローリングに必要な過去データは有限なので、必要期間だけ切り出して
        build_features() を呼ぶ。全履歴を毎回再計算しなくてよい。

    price/stock の中央値補完について:
        rolling.py の _fill_price_and_stock() は渡された DataFrame の中央値を使う。
        lookback 期間（49日分）だけを渡すと、全履歴より中央値がずれる可能性がある。
        → ★ 事前に全履歴 df から計算した中央値で temp_df の欠損を補完してから
            build_features() に渡す。これにより旧実装と同じ精度を保てる。
    """

    # ① 日付順に並び替える
    df = (
        df.sort_values(["store", "product", "date"])
          .reset_index(drop=True)
    )

    target_mask = (df["date"] == target_date)
    if not target_mask.any():
        raise ValueError(f"{target_date} が DataFrame に存在しません")

    # ② lookback 期間だけ切り出す
    lookback_days = max(LAGS) + max(ROLLING_WINDOWS) + 7
    start_date    = target_date - pd.Timedelta(days=lookback_days)
    temp_df = df[
        (df["date"] >= start_date) & (df["date"] <= target_date)
    ].copy()

    # ③ ★ price/stock の欠損を全履歴 df の中央値で事前補完する
    #    rolling.py に渡す前に埋めておくことで、
    #    lookback 期間が短くても正確な補完値が使われる
    for col in ["price", "stock"]:
        if col not in df.columns:
            continue
        global_median = float(df[col].median()) if not df[col].isna().all() else 0.0
        temp_df[col]  = temp_df[col].fillna(global_median)

    # ④ build_features() でカレンダー・ラグ・ローリングを一括計算
    #    ロジックの実体は features/ に一本化されているため、
    #    学習時とまったく同じ計算式が使われる（一貫性の保証）
    temp_df = build_features(temp_df)

    # ⑤ 計算した特徴量を元の df の target_date 行に上書きする
    updated_rows = temp_df[temp_df["date"] == target_date]
    df.loc[target_mask, updated_rows.columns] = updated_rows.values

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
