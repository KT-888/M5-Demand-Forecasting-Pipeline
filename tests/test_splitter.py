"""
tests/test_splitter.py
=======================
pipeline/splitter.py の分割ロジックが正しいかをテストする。

テストの方針:
    「境界日が正しく計算されるか」
    「train/valid/test に日付の重複・欠落がないか」
    「SKU が途中から追加された不均一データでも正しく分割されるか」
    を確認する。

    splitter が壊れると train/valid/test の境界がずれ、
    全ての ML 結果が無意味になるため最重要のテストのひとつ。
"""

import pandas as pd
import numpy as np
import pytest

from pipeline.splitter import (
    split_train_valid_test_by_date,
    generate_walkforward_cv_folds,
    _get_representative_date_range,
)


# =========================================================
# テスト用データ生成ヘルパー
# =========================================================

def _make_df(
    n_days:   int = 100,
    n_stores: int = 2,
    n_products: int = 2,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """均一な日付範囲を持つテスト用 DataFrame。"""
    dates    = pd.date_range(start, periods=n_days)
    stores   = [f"S{i:02d}" for i in range(1, n_stores + 1)]
    products = [f"P{i:03d}" for i in range(1, n_products + 1)]
    records  = [
        {"store": s, "product": p, "date": d, "sales": 10.0}
        for s in stores for p in products for d in dates
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_uneven_df() -> pd.DataFrame:
    """
    SKU ごとに販売開始日が異なる不均一データ。
    S01/P001 は 180日分、S01/P002 は途中から 60日分のみ。
    """
    dates_long  = pd.date_range("2024-01-01", periods=180)
    dates_short = pd.date_range("2024-04-11", periods=60)  # 途中から開始

    records = (
        [{"store": "S01", "product": "P001", "date": d, "sales": 10.0}
         for d in dates_long]
        + [{"store": "S01", "product": "P002", "date": d, "sales": 10.0}
           for d in dates_short]
    )
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# =========================================================
# _get_representative_date_range のテスト
# =========================================================

class TestGetRepresentativeDateRange:

    def test_均一データは全体のmin_maxと一致する(self):
        df = _make_df(n_days=100)
        oldest, newest = _get_representative_date_range(df)
        assert oldest == df["date"].min()
        assert newest == df["date"].max()

    def test_不均一データでも極端な外れ値に引きずられない(self):
        df = _make_uneven_df()
        oldest, newest = _get_representative_date_range(df)

        # 中央値なので P002 の開始日（2024-04-11）に引きずられず
        # 大多数（P001）の開始日 2024-01-01 に近いはず
        global_min = df["date"].min()  # 2024-01-01
        p002_start = pd.Timestamp("2024-04-11")

        # oldest は global_min 〜 p002_start の間に収まる
        assert global_min <= oldest <= p002_start


# =========================================================
# split_train_valid_test_by_date のテスト
# =========================================================

class TestSplitTrainValidTestByDate:

    def test_3分割の行数の合計が元データと一致する(self):
        df = _make_df(n_days=100)
        train, valid, test = split_train_valid_test_by_date(df)
        assert len(train) + len(valid) + len(test) == len(df)

    def test_日付の順序が正しい_train_valid_test(self):
        df = _make_df(n_days=100)
        train, valid, test = split_train_valid_test_by_date(df)
        assert train["date"].max() < valid["date"].min()
        assert valid["date"].max() < test["date"].min()

    def test_各パートに重複行がない(self):
        df = _make_df(n_days=100)
        train, valid, test = split_train_valid_test_by_date(df)
        for part in [train, valid, test]:
            assert not part.duplicated(subset=["store", "product", "date"]).any()

    def test_valid_ratioが反映される(self):
        df = _make_df(n_days=200, n_stores=1, n_products=1)
        train, valid, test = split_train_valid_test_by_date(
            df, valid_ratio=0.2, test_ratio=0.2
        )
        total_days = (df["date"].max() - df["date"].min()).days
        valid_days = (valid["date"].max() - valid["date"].min()).days
        # valid は全体の約 20% になるはず（± 5日の誤差を許容）
        assert abs(valid_days - total_days * 0.2) <= 5

    def test_test_ratioが反映される(self):
        df = _make_df(n_days=200, n_stores=1, n_products=1)
        train, valid, test = split_train_valid_test_by_date(
            df, valid_ratio=0.2, test_ratio=0.2
        )
        total_days = (df["date"].max() - df["date"].min()).days
        test_days  = (test["date"].max() - test["date"].min()).days
        assert abs(test_days - total_days * 0.2) <= 5

    def test_不均一SKUデータでも全パートが空にならない(self):
        df = _make_uneven_df()
        train, valid, test = split_train_valid_test_by_date(df)
        assert len(train) > 0, "train が空"
        assert len(valid) > 0, "valid が空"
        assert len(test)  > 0, "test が空"

    def test_不均一SKUでtrainに全SKUが含まれる(self):
        """新商品（途中追加 SKU）も train に少なくとも数行含まれることを確認。"""
        df = _make_uneven_df()
        train, valid, test = split_train_valid_test_by_date(df)
        train_products = set(train["product"].unique())
        all_products   = set(df["product"].unique())
        # 全 SKU が train に存在する（中央値ベース境界日の恩恵）
        assert train_products == all_products, \
            f"train に含まれない SKU があります: {all_products - train_products}"


# =========================================================
# generate_walkforward_cv_folds のテスト
# =========================================================

class TestGenerateWalkforwardCvFolds:

    def test_指定したfold数だけ生成される(self):
        df    = _make_df(n_days=100)
        folds = list(generate_walkforward_cv_folds(df, n_splits=3))
        assert len(folds) == 3

    def test_各foldでtrainとvalidに重複がない(self):
        df    = _make_df(n_days=100)
        folds = list(generate_walkforward_cv_folds(df, n_splits=3))
        for train, valid in folds:
            assert train["date"].max() < valid["date"].min()

    def test_fold間でtrainが拡大している_expanding_window(self):
        df    = _make_df(n_days=100)
        folds = list(generate_walkforward_cv_folds(df, n_splits=3))
        train_sizes = [len(f[0]) for f in folds]
        # Expanding window: fold が進むほど train が増える
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]

    def test_学習期間不足でValueErrorが発生する(self):
        df = _make_df(n_days=100)
        # n_splits=10, valid_ratio=0.9 → 最初の fold の学習期間が 40% を下回る
        with pytest.raises(ValueError, match="を下回ります"):
            list(generate_walkforward_cv_folds(
                df, n_splits=10, valid_ratio=0.9
            ))

    def test_最終foldのvalidは末尾まで含む(self):
        df    = _make_df(n_days=100, n_stores=1, n_products=1)
        folds = list(generate_walkforward_cv_folds(df, n_splits=3))
        last_train, last_valid = folds[-1]
        # 最終 fold の valid の最終日はデータ全体の最終日と一致するはず
        assert last_valid["date"].max() == df["date"].max()
