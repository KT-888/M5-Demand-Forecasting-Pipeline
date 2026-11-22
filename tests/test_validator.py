"""
tests/test_validator.py
========================
utils/validator.py の全関数をテストする。

テストの方針:
    「正常データは通す・異常データは検知する」の両方を確認する。
    各チェック項目（必須列欠落・負値・重複・dtype不一致など）を
    1テスト1観点で独立して書く。
    → どのチェックが壊れたかが一目でわかる。
"""

import numpy as np
import pandas as pd
import pytest

from utils.validator import (
    validate_raw_input,
    validate_feature_schema,
    validate_features,
    validate_predict_input,
)


# =========================================================
# テスト用データ生成ヘルパー
# =========================================================

def _make_raw_df(
    n_days: int = 30,
    stores: list = None,
    products: list = None,
    sales_val: float = 50.0,
    include_optional: bool = True,
) -> pd.DataFrame:
    """正常な生データを生成するヘルパー。"""
    stores   = stores   or ["S01"]
    products = products or ["P001"]
    dates    = pd.date_range("2024-01-01", periods=n_days)

    records = [
        {
            "store":   s, "product": p, "date": d,
            "sales":   sales_val,
            **({"price": 1000, "stock": 100, "onpromotion": 0}
               if include_optional else {}),
        }
        for s in stores for p in products for d in dates
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_feature_df() -> pd.DataFrame:
    """build_features() 済みの正常な特徴量 DataFrame を生成するヘルパー。"""
    from pipeline.feature_pipeline import build_features

    df = _make_raw_df(n_days=60, include_optional=True)
    df["onpromotion"] = 0
    return build_features(df)


# =========================================================
# ① validate_raw_input のテスト
# =========================================================

class TestValidateRawInput:

    def test_正常データは通る(self):
        df = _make_raw_df()
        assert validate_raw_input(df, raise_error=True) is True

    def test_任意列がなくても通る(self):
        df = _make_raw_df(include_optional=False)
        assert validate_raw_input(df, raise_error=True) is True

    def test_必須列_store_欠落でエラー(self):
        df = _make_raw_df().drop(columns=["store"])
        with pytest.raises(ValueError, match="必須列"):
            validate_raw_input(df, raise_error=True)

    def test_必須列_sales_欠落でエラー(self):
        df = _make_raw_df().drop(columns=["sales"])
        with pytest.raises(ValueError, match="必須列"):
            validate_raw_input(df, raise_error=True)

    def test_sales_負値でエラー(self):
        df = _make_raw_df()
        df.loc[0, "sales"] = -1.0
        with pytest.raises(ValueError, match="最小値を下回る"):
            validate_raw_input(df, raise_error=True)

    def test_sales_ゼロは許容される(self):
        df = _make_raw_df(sales_val=0.0)
        assert validate_raw_input(df, raise_error=True) is True

    def test_空DataFrame_でエラー(self):
        df = pd.DataFrame(columns=["store", "product", "date", "sales"])
        with pytest.raises(ValueError, match="0行"):
            validate_raw_input(df, raise_error=True)

    def test_store_product_date_重複でエラー(self):
        df = _make_raw_df(n_days=5)
        # 同じ行を追加して重複を作る
        df = pd.concat([df, df.head(1)], ignore_index=True)
        with pytest.raises(ValueError, match="重複"):
            validate_raw_input(df, raise_error=True)

    def test_raise_error_False_でエラー時にFalseを返す(self):
        df = _make_raw_df()
        df.loc[0, "sales"] = -1.0
        result = validate_raw_input(df, raise_error=False)
        assert result is False

    def test_onpromotion_が0_1以外でエラー(self):
        df = _make_raw_df()
        df["onpromotion"] = 2  # 0/1 以外
        with pytest.raises(ValueError, match="最大値を超える"):
            validate_raw_input(df, raise_error=True)


# =========================================================
# ② validate_feature_schema のテスト
# =========================================================

class TestValidateFeatureSchema:

    def test_正常な特徴量DataFrameは通る(self):
        df = _make_feature_df()
        assert validate_feature_schema(df, raise_error=True) is True

    def test_期待列が欠落するとエラー(self):
        df = _make_feature_df()
        df = df.drop(columns=["lag_1"])
        with pytest.raises(ValueError, match="期待する特徴量列"):
            validate_feature_schema(df, raise_error=True)

    def test_avail列に0_1以外の値でエラー(self):
        df = _make_feature_df()
        df["lag_1_avail"] = 2  # 不正値
        with pytest.raises(ValueError, match="avail フラグ"):
            validate_feature_schema(df, raise_error=True)

    def test_inf混入でエラー(self):
        df = _make_feature_df()
        # lag_1 を float64 にしてから inf を入れる
        df["lag_1"] = df["lag_1"].astype(float)
        df.loc[df.index[0], "lag_1"] = np.inf
        with pytest.raises(ValueError, match="inf"):
            validate_feature_schema(df, raise_error=True)

    def test_未知列は警告のみでエラーにならない(self):
        df = _make_feature_df()
        df["謎の列"] = 0  # 期待スキーマにない列
        # raise_error=True でも警告どまりでエラーにはならない
        result = validate_feature_schema(df, raise_error=True)
        assert result is True


# =========================================================
# ③ validate_features のテスト
# =========================================================

class TestValidateFeatures:

    def _make_X(self, cols: list) -> pd.DataFrame:
        return pd.DataFrame({c: [1.0] for c in cols})

    def test_同一列名_同一dtype_は通る(self):
        X = self._make_X(["a", "b", "c"])
        # エラーが出ないことを確認
        validate_features(X, X.copy(), raise_error=True)

    def test_train_にのみ存在する列でエラー(self):
        X_train = self._make_X(["a", "b", "extra"])
        X_test  = self._make_X(["a", "b"])
        with pytest.raises(ValueError, match="Train にのみ存在"):
            validate_features(X_train, X_test, raise_error=True)

    def test_test_にのみ存在する列でエラー(self):
        X_train = self._make_X(["a", "b"])
        X_test  = self._make_X(["a", "b", "extra"])
        with pytest.raises(ValueError, match="Test にのみ存在"):
            validate_features(X_train, X_test, raise_error=True)

    def test_dtype不一致は警告のみ_エラーにならない(self):
        X_train = pd.DataFrame({"a": np.array([1.0], dtype=np.float32)})
        X_test  = pd.DataFrame({"a": np.array([1.0], dtype=np.float64)})
        # dtype 不一致は警告どまり（LightGBM は許容するため）
        validate_features(X_train, X_test, raise_error=False)


# =========================================================
# ④ validate_predict_input のテスト
# =========================================================

class TestValidatePredictInput:

    def test_正常データは通る(self):
        df        = _make_feature_df()
        feat_cols = [c for c in df.columns
                     if c not in {"store", "product", "date", "sales"}]
        assert validate_predict_input(df, feat_cols, raise_error=True) is True

    def test_feat_cols_に存在しない列でエラー(self):
        df        = _make_feature_df()
        feat_cols = ["lag_1", "存在しない列"]
        with pytest.raises(ValueError, match="特徴量列がありません"):
            validate_predict_input(df, feat_cols, raise_error=True)

    def test_全行NaN列でエラー(self):
        df        = _make_feature_df()
        feat_cols = [c for c in df.columns
                     if c not in {"store", "product", "date", "sales"}]
        df["lag_1"] = np.nan  # 全行 NaN
        with pytest.raises(ValueError, match="全行 NaN"):
            validate_predict_input(df, feat_cols, raise_error=True)

    def test_inf混入でエラー(self):
        df        = _make_feature_df()
        feat_cols = [c for c in df.columns
                     if c not in {"store", "product", "date", "sales"}]
        df["lag_1"] = df["lag_1"].astype(float)
        df.loc[df.index[0], "lag_1"] = np.inf
        with pytest.raises(ValueError, match="inf"):
            validate_predict_input(df, feat_cols, raise_error=True)
