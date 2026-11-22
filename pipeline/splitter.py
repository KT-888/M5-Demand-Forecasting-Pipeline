"""
pipeline/splitter.py
====================
学習データと検証データを「日付で」分割するモジュール。

なぜランダム分割ではダメか:
    売上予測は「過去のデータで学習 → 未来を予測」が本来の使い方。
    ランダムに分割すると「未来のデータで学習して過去を予測する」という
    時間の逆転が起き、現実では通用しない甘いモデルができあがる。
    → 必ず「古い日付 = 学習、新しい日付 = 検証」と日付で切ること。

用意している関数:
    split_train_valid_test_by_date() : 3分割（本番推奨）← メインで使う
    generate_walkforward_cv_folds()  : Walkforward CV 用の複数 fold 生成

【修正内容 (v3)】
    ★ SKU ごとに販売開始日が異なるデータへの対応。

    旧実装の問題:
        境界日を df["date"].min() / max() のグローバル値で決めていた。
        新商品が途中から追加されたデータでは、一部 SKU の販売開始日が
        test 期間の開始日より後になり、その SKU の train/valid データが
        ゼロ行になる問題があった。

    修正後:
        「全 SKU に共通して存在する最古・最新日付」を使って境界を決める。
        具体的には SKU ごとの min/max の中央値（median）を採用。
        これにより外れ値的な SKU に引きずられず、大多数の SKU が
        適切に分割されるようになった。
        ※ 販売期間が極端に短い SKU は warmup 除外の段階で対処済み。
"""

import pandas as pd
from typing import Iterator, Tuple


# =========================================================
# 外部から呼び出す関数
# =========================================================

def split_train_valid_test_by_date(
    df:          pd.DataFrame,
    valid_ratio: float = 0.2,
    test_ratio:  float = 0.2,
):
    """
    全データを train / valid / test の3つに日付順で分割する。【推奨】

    各パートの役割:
        train : パラメータ探索と最終モデルの学習（train + valid）に使う
        valid : Early Stopping 判定と MAE 評価に使う
        test  : 最終モデルの精度を公正に評価するためだけに使う
                → 学習には一切使わない。ここでの MAE が本当の精度。

    境界日の決め方:
        SKU ごとの最古・最新日付の中央値を使う。
        グローバルな min/max を使うと、販売開始が遅い新商品 SKU に
        引きずられて境界日がずれる問題を防ぐ。

    Parameters
    ----------
    df          : store / product / date / sales などを含む DataFrame
    valid_ratio : valid として使う期間の割合（デフォルト: 全期間の 20%）
    test_ratio  : test として使う期間の割合（デフォルト: 全期間の 20%）

    Returns
    -------
    train_df, valid_df, test_df の3つの DataFrame
    """
    df = df.sort_values(["store", "product", "date"]).copy()

    # ★ SKU ごとの min/max の中央値で境界日を決める
    oldest_date, newest_date = _get_representative_date_range(df)
    total_days = (newest_date - oldest_date).days

    # 新しい方から test → valid → train の順に境界日を決める
    test_days   = round(total_days * test_ratio)
    valid_days  = round(total_days * valid_ratio)
    test_start  = newest_date - pd.Timedelta(days=test_days)
    valid_start = test_start  - pd.Timedelta(days=valid_days)

    train_df = df[df["date"] <  valid_start].copy()
    valid_df = df[(df["date"] >= valid_start) & (df["date"] < test_start)].copy()
    test_df  = df[df["date"] >= test_start].copy()

    print(f"📅 全期間    : {df['date'].min().date()} ～ {df['date'].max().date()}")
    print(f"✂️  学習期間  : {train_df['date'].min().date()} ～ {train_df['date'].max().date()}  ({len(train_df):,} 行)")
    print(f"🧪 検証期間  : {valid_df['date'].min().date()} ～ {valid_df['date'].max().date()}  ({len(valid_df):,} 行)")
    print(f"🎯 テスト期間: {test_df['date'].min().date()}  ～ {test_df['date'].max().date()}   ({len(test_df):,} 行)")

    return train_df, valid_df, test_df


def generate_walkforward_cv_folds(
    df:              pd.DataFrame,
    n_splits:        int   = 3,
    valid_ratio:     float = 0.2,
    min_train_ratio: float = 0.4,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    時系列 Cross Validation（交差検証）用の fold を生成する。

    Expanding window 方式:
        fold が進むほど学習データを増やしていく。
        [学習:1〜3月 | 検証:4月]
        [学習:1〜4月 | 検証:5月]  ← 学習データが1か月増えた
        [学習:1〜5月 | 検証:6月]  ← さらに増えた

    Parameters
    ----------
    df              : store / product / date / sales などを含む DataFrame
    n_splits        : fold の数（何回に分けて検証するか）
    valid_ratio     : 検証期間の全体に対する割合
    min_train_ratio : 最初の fold の学習期間が全体の何割以上必要か

    Yields
    ------
    (train_df, valid_df) のタプルを n_splits 回
    """
    df = df.sort_values(["store", "product", "date"]).copy()

    # ★ SKU ごとの min/max の中央値で境界日を決める（split と統一）
    oldest_date, newest_date = _get_representative_date_range(df)
    total_days = (newest_date - oldest_date).days
    valid_days = max(1, round(total_days * valid_ratio / n_splits))

    first_valid_start_day = total_days - valid_days * n_splits
    if first_valid_start_day < total_days * min_train_ratio:
        raise ValueError(
            f"n_splits={n_splits} / valid_ratio={valid_ratio} の組み合わせでは\n"
            f"最初の fold の学習期間が全体の {min_train_ratio:.0%} を下回ります。\n"
            f"n_splits を減らすか valid_ratio を小さくしてください。"
        )

    print(f"\n🔀 Walkforward CV: {n_splits} folds / 1fold あたり検証 {valid_days} 日")
    print(f"📅 全期間: {df['date'].min().date()} ～ {df['date'].max().date()}")

    for fold_index in range(n_splits):
        valid_end_day   = total_days - valid_days * (n_splits - fold_index - 1)
        valid_start_day = valid_end_day - valid_days

        valid_start = oldest_date + pd.Timedelta(days=valid_start_day)
        valid_end   = oldest_date + pd.Timedelta(days=valid_end_day)

        train_df     = df[df["date"] < valid_start].copy()
        is_last_fold = (fold_index == n_splits - 1)

        if is_last_fold:
            valid_df = df[df["date"] >= valid_start].copy()
        else:
            valid_df = df[(df["date"] >= valid_start) & (df["date"] < valid_end)].copy()

        print(
            f"  Fold {fold_index + 1}: "
            f"学習=[{train_df['date'].min().date()}, {train_df['date'].max().date()}] "
            f"({len(train_df):,}行) / "
            f"検証=[{valid_df['date'].min().date()}, {valid_df['date'].max().date()}] "
            f"({len(valid_df):,}行)"
        )

        yield train_df, valid_df


# =========================================================
# 内部ヘルパー関数
# =========================================================

def _get_representative_date_range(
    df: pd.DataFrame,
) -> tuple:
    """
    SKU ごとの最古・最新日付の中央値を返す。

    なぜ中央値を使うか:
        新商品が途中から追加されたデータでは、販売開始が遅い SKU の
        min 日付がグローバル min より大幅に新しい。
        全体の min/max を使うと、その SKU の train データがゼロ行になる
        「学習データ不足 SKU」が発生する。
        中央値を使うことで外れ値 SKU に引きずられない境界日を決められる。

    Returns
    -------
    (oldest_date, newest_date) : 代表的な最古・最新日付のタプル
    """
    sku_date_ranges = df.groupby(["store", "product"])["date"].agg(["min", "max"])

    # 中央値を Timestamp に変換
    oldest_date = pd.Timestamp(sku_date_ranges["min"].quantile(0.5, interpolation="midpoint"))
    newest_date = pd.Timestamp(sku_date_ranges["max"].quantile(0.5, interpolation="midpoint"))

    return oldest_date, newest_date
