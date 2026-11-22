"""
model/predictor.py
==================
未来 N 日間の逐次予測を担当するモジュール。

【修正内容 (v2)】
    concat をループ外の1回に削減（メモリ膨張防止）。

【修正内容 (v3)】
    log1p / expm1 対応。
    trainer.py で log1p 変換して学習したモデルは
    log スケールの予測値を返すため、expm1 で元スケールに戻す必要がある。
    use_log_transform=True（デフォルト）のとき expm1 変換を適用する。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import FORECAST_DAYS
from pipeline.feature_pipeline import update_features_for_one_day


def predict_future(
    model:             lgb.Booster,
    df:                pd.DataFrame,
    feat_cols:         list,
    forecast_days:     int  = FORECAST_DAYS,
    future_promo:      pd.DataFrame = None,
    use_log_transform: bool = True,
) -> pd.DataFrame:
    """
    未来 forecast_days 日間を1日ずつ逐次予測する。

    Parameters
    ----------
    model             : 学習済み LightGBM モデル
    df                : 特徴量生成済みの全履歴データ
    feat_cols         : モデルの入力特徴量リスト
    forecast_days     : 予測日数（デフォルト: config.FORECAST_DAYS = 7）
    future_promo      : プロモ計画（store / product / date / onpromotion 列）
                        None の場合は全日プロモなし（0）として扱う
    use_log_transform : True のとき予測値に expm1 を適用して元スケールに戻す。
                        trainer.py の train_on_full_data() は log1p 変換して学習するため
                        デフォルト True。log 変換なしのモデルを使う場合は False を指定。

    Returns
    -------
    forecast_df : store / product / date / sales（予測値・元スケール）の DataFrame
    """
    df        = df.sort_values(["store", "product", "date"]).reset_index(drop=True).copy()
    last_date = df["date"].max()

    unique_pairs = df[["store", "product"]].drop_duplicates()

    # 全予測日分のプレースホルダ行を先に一括作成（concat は1回だけ）
    all_future_rows = []
    for i in range(1, forecast_days + 1):
        target_date = last_date + pd.Timedelta(days=i)
        future_rows = unique_pairs.copy()
        future_rows["date"]  = target_date
        future_rows["sales"] = np.float32(np.nan)

        if future_promo is not None:
            promo_day   = future_promo[future_promo["date"] == target_date]
            future_rows = future_rows.merge(
                promo_day[["store", "product", "onpromotion"]],
                on=["store", "product"],
                how="left",
            )
            future_rows["onpromotion"] = future_rows["onpromotion"].fillna(0).astype(int)
        else:
            future_rows["onpromotion"] = 0

        all_future_rows.append(future_rows)

    df = pd.concat([df, *all_future_rows], ignore_index=True)

    results = []

    for i in range(1, forecast_days + 1):
        target_date = last_date + pd.Timedelta(days=i)

        df = update_features_for_one_day(df, target_date)

        today = df[df["date"] == target_date].copy()
        for col in feat_cols:
            if col not in today.columns:
                today[col] = 0.0

        raw_preds = model.predict(today[feat_cols])

        # ★ log1p 変換モデルの場合は expm1 で元スケールに戻す
        if use_log_transform:
            preds = np.expm1(raw_preds).clip(min=0)
        else:
            preds = np.asarray(raw_preds).clip(min=0)

        # 翌日以降のラグ特徴量のために元スケールの値を書き戻す
        # ★ float32 列への代入は明示的にキャストして FutureWarning を回避
        df.loc[df["date"] == target_date, "sales"] = preds.astype(np.float32)

        result = df[df["date"] == target_date][["store", "product", "date", "sales"]].copy()
        results.append(result)

    forecast_df = pd.concat(results, ignore_index=True)
    return forecast_df
