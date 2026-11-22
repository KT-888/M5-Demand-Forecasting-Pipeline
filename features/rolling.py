"""
features/rolling.py
===================
「直近N日間の集計値」を特徴量として生成するモジュール。
 
Rolling（ローリング）とは:
    日付を1日ずつ進めながら「直近N日間」の窓（ウィンドウ）で
    平均・標準偏差などを計算すること。
 
【食品加工会社向けに残した特徴量】
    ma7          : 直近7日間の売上平均（週次トレンドを捉える）
    ma28         : 直近28日間の売上平均（月次トレンドを捉える）
    std7         : 直近7日間の売上の標準偏差（需要のばらつき = 安全在庫計算に必要）
    std7_valid   : std7 が計算できたか（データ不足なら 0）
    price        : 売価（欠損を前方補完）
    stock        : 在庫数（欠損を前方補完）
 
【削除した特徴量】
    ma14              : ma7 と ma28 で代替できるため削除
    momentum1         : lag_1 に前日比の情報が含まれているため重複
    popularity        : rank 系と重複。rank 系も削除したためまとめて削除
    rank7/14/30       : ma 系で代替可能なため削除
    promo_rolling_7   : B2B 食品加工では不要なケースが多い
    config.py から RANK_WINDOWS ごと削除済み
 
未来リーク防止の鉄則:
    必ず shift(1) してから rolling/expanding を適用する。
    「当日の売上」が特徴量に混入しないようにするため。
"""
 
import pandas as pd
from config import ROLLING_WINDOWS
 
 
def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ローリング系特徴量をまとめて生成して返す。
    呼び出し順序を固定してバグを防ぐ。
    """
    df = df.sort_values(["store", "product", "date"]).copy()
 
    df = _add_moving_averages(df)    # 移動平均（ma7, ma28）
    df = _add_demand_volatility(df)  # 需要のばらつき（std7, std7_valid）
    df = _fill_price_and_stock(df)   # 価格・在庫の欠損補完
 
    return df
 
 
# =========================================================
# 内部関数（このファイルの中だけで使う。外から呼ばない。）
# =========================================================
 
def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    ROLLING_WINDOWS で指定した日数の移動平均を計算する。
 
    config.py で ROLLING_WINDOWS = [7, 28] に設定しているため
    ma7（7日平均）と ma28（28日平均）の2列が生成される。
 
    shift(1) の理由:
        今日のデータが特徴量に混入する「未来リーク」を防ぐため。
        shift(1) = 1日ずらすことで「昨日までのデータ」だけを使う。
 
    min_periods=1 の理由:
        データが window 日分に満たなくても計算する（新商品の初日から使える）。
    """
    g = df.groupby(["store", "product"])
 
    for window in ROLLING_WINDOWS:
        df[f"ma{window}"] = g["sales"].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
        )
 
    return df
 
 
def _add_demand_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    需要のばらつき（標準偏差）を7日間で計算する。
 
    なぜ食品加工の在庫管理で重要か:
        安全在庫の計算式 = Z × σ × √リードタイム
        σ（シグマ）= 需要の標準偏差。ここで計算する std7 がその近似値。
        ばらつきが大きい商品ほど安全在庫を多く持つ必要がある。
 
    std7_valid フラグの意味:
        標準偏差は最低2行のデータがないと計算できない（NaN になる）。
        ★ 修正: 旧コメントは「std7_valid=0」と書いていたが、
          実装は「NaN かどうか（notna）」で判定している。
          → NaN でなければ 1（データがあって計算できた）
          → NaN なら 0（データ不足で計算できなかった）
          売上が毎日同じ値（例: 毎日50個）の場合は std=0 になるが、
          NaN ではないので std7_valid=1（正しく計算できた）になる。
    """
    g = df.groupby(["store", "product"])
 
    # shift(1) してから直近7日間の標準偏差を計算
    # min_periods=2: 最低2行ないと NaN になる（標準偏差は2点以上必要）
    df["std7"] = g["sales"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=2).std()
    )
 
    # std7_valid: NaN かどうかで「計算できたか」を判定する
    # notna() = NaN でなければ True（→ 1）、NaN なら False（→ 0）
    # ※ std7 > 0 ではなく notna() で判定する理由:
    #   std7 > 0 だと、売上が毎日同じ値のとき std=0 になり
    #   データがあるのに valid=0 と誤判定してしまう
    df["std7_valid"] = df["std7"].notna().astype(int)
 
    # NaN を 0 に置き換える（モデルは NaN を扱えないため）
    # ★ valid の記録が終わってから NaN を埋める（順番が重要）
    df["std7"] = g["std7"].transform(lambda x: x.ffill()).fillna(0)
 
    return df
 
 
def _fill_price_and_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    price（売価）と stock（在庫数）の欠損値を補完する。
 
    補完の順番:
        1. ffill()（前方補完）: 同じ店舗・商品の直前の既知値で埋める
        2. median()（中央値） : 前方補完でも埋まらない先頭の欠損を
                                全データの中央値で埋める（新商品の初日など）
 
    なぜ平均値でなく中央値か:
        価格に外れ値（セール品など極端に安い日）があると平均値が歪む。
        中央値は外れ値の影響を受けにくいため、より安定した補完ができる。
    """
    for col in ["price", "stock"]:
        if col not in df.columns:
            continue  # その列がデータに存在しない場合はスキップ
 
        # 店舗・商品ごとに前方補完
        df[col] = df.groupby(["store", "product"])[col].transform("ffill")
 
        # それでも残った欠損は全体中央値で埋める
        df[col] = df[col].fillna(df[col].median())
 
    return df