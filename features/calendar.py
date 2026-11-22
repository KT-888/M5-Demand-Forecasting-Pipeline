"""
features/calendar.py
====================
日付から「いつか」に関する特徴量を生成するモジュール。

【食品加工会社向けに残した特徴量】
    月          : 1〜12 の数値。季節性を捉える
    曜日        : 0（月）〜6（日）の数値。週次の発注パターンを捉える
    土日か      : 土日=1、平日=0
    祝日か      : 祝日=1、平日=0
    月末フラグ  : 月の最終3日間=1。スーパーが棚卸前に大量発注する
    期末フラグ  : 3・6・9・12月の月末3日間=1。決算期の駆け込み発注

【削除した特徴量】
    日（1〜31）       : 月末フラグで代替できるため削除
    伊勢神宮初詣      : 三重県固有。他社クライアントには意味がない
    鈴鹿F1月          : 同上
    wdm_* 84列        : 「曜日」と「月」の数値2列で代替可能。
                        config.py から WEEKDAY_MONTH / MONTHS / WEEKDAYS ごと削除済み
"""

import pandas as pd
from config import JP_HOLIDAYS


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    date 列からカレンダー系の特徴量を生成して返す。

    Parameters
    ----------
    df : date 列（datetime 型）を含む DataFrame

    Returns
    -------
    特徴量列が追加された DataFrame
    """
    df = df.copy()

    # ---- 基本のカレンダー情報 ----
    df["月"]   = df["date"].dt.month        # 1〜12
    df["曜日"] = df["date"].dt.dayofweek    # 0（月曜）〜6（日曜）

    # ---- フラグ系（0 か 1 の二値）----
    df["土日か"] = (df["曜日"] >= 5).astype(int)
    df["祝日か"] = df["date"].map(lambda x: x in JP_HOLIDAYS).astype(int)

    # ---- 食品加工業向け：月末・期末フラグ ----
    df = _add_month_end_flags(df)

    return df


# =========================================================
# 内部関数（このファイルの中だけで使う。外から呼ばない。）
# =========================================================

def _add_month_end_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    月末フラグと期末フラグを追加する。

    なぜこの2つが食品加工業で重要か:
        食品メーカーの主な出荷先（スーパー・卸売業者）は
        月末に棚卸をするため、その直前に大量発注してくる傾向がある。
        特に3・6・9・12月の期末は決算の影響でさらに発注が増える。

    月末フラグの定義:
        その月の最終日から3日以内 = 1、それ以外 = 0
        例: 31日ある月なら 29・30・31日が 1

    期末フラグの定義:
        3・6・9・12月 かつ 月末3日以内 = 1、それ以外 = 0
    """
    df = df.copy()

    # その月の最終日を取得する
    # pd.offsets.MonthEnd(0) = 「その月の末日に移動する」pandas の機能
    # 例: 2024-01-15 → 2024-01-31
    month_end     = df["date"] + pd.offsets.MonthEnd(0)
    month_end_day = month_end.dt.day       # 末日の「日」番号（28〜31）
    current_day   = df["date"].dt.day      # 今日の「日」番号

    # 末日から何日前かを計算（0=末日, 1=前日, 2=前々日）
    days_to_month_end = month_end_day - current_day

    # 末日から2日以内（0・1・2日前）なら月末フラグ=1
    df["月末フラグ"] = (days_to_month_end <= 2).astype(int)

    # 期末月（3・6・9・12月）かつ月末フラグが立っている行が期末フラグ=1
    is_quarter_end_month = df["月"].isin([3, 6, 9, 12])
    df["期末フラグ"] = (is_quarter_end_month & (df["月末フラグ"] == 1)).astype(int)

    return df
