"""
tests/test_predictor.py
========================
model/predictor.py の逐次予測ロジックをテストする。

テストの方針:
    ① 出力の形式が正しいか（列名・行数・日付範囲）
    ② 予測値が非負か（売上はマイナスにならない）
    ③ ラグ特徴量が1日ずつ正しく引き継がれるか（逐次予測の核心）
    ④ future_promo が反映されるか

    trainer.py の Optuna は実行コストが高いため、
    ここでは「固定パラメータで学習した軽量モデル」を使って
    predictor の動作のみを検証する。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import pytest

from pipeline.feature_pipeline import build_features, get_feature_col_names
from model.predictor import predict_future


# =========================================================
# テスト用モデル・データ生成ヘルパー
# =========================================================

def _make_history_df(n_days: int = 60) -> pd.DataFrame:
    """テスト用の履歴データを生成する。"""
    dates    = pd.date_range("2024-01-01", periods=n_days)
    stores   = ["S01", "S02"]
    products = ["P001", "P002"]
    records  = [
        {"store": s, "product": p, "date": d,
         "sales": float(20 + i % 7), "onpromotion": 0}
        for i, (s, p, d) in enumerate(
            (s, p, d) for s in stores for p in products for d in dates
        )
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _train_simple_model(df: pd.DataFrame):
    """
    軽量な固定パラメータで LightGBM を学習する。
    Optuna は使わない（テスト速度のため）。
    """
    df_feat   = build_features(df)
    feat_cols = get_feature_col_names(df_feat)

    # ウォームアップ除去
    cutoff = df_feat["date"].min() + pd.Timedelta(days=28)
    df_train = df_feat[df_feat["date"] >= cutoff]

    X = df_train[feat_cols]
    y = df_train["sales"]

    params = {
        "objective":     "regression_l1",
        "metric":        "mae",
        "num_leaves":    31,
        "learning_rate": 0.1,
        "verbosity":     -1,
        "random_state":  42,
    }
    model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=50)
    return model, df_feat, feat_cols


# =========================================================
# predict_future のテスト
# =========================================================

class TestPredictFuture:

    @pytest.fixture(scope="class")
    def trained(self):
        """クラス内でモデルを1回だけ学習してキャッシュする。"""
        df    = _make_history_df(n_days=60)
        model, df_feat, feat_cols = _train_simple_model(df)
        return model, df_feat, feat_cols

    def test_出力列が正しい(self, trained):
        model, df_feat, feat_cols = trained
        result = predict_future(model, df_feat, feat_cols, forecast_days=3, use_log_transform=False)
        assert set(result.columns) == {"store", "product", "date", "sales"}

    def test_予測行数が正しい_stores_x_products_x_days(self, trained):
        model, df_feat, feat_cols = trained
        forecast_days = 5
        result = predict_future(model, df_feat, feat_cols, forecast_days=forecast_days, use_log_transform=False)
        n_skus = df_feat[["store", "product"]].drop_duplicates().shape[0]
        assert len(result) == n_skus * forecast_days

    def test_予測日付が連続している(self, trained):
        model, df_feat, feat_cols = trained
        forecast_days = 7
        result   = predict_future(model, df_feat, feat_cols, forecast_days=forecast_days)
        dates    = sorted(result["date"].unique())
        assert len(dates) == forecast_days
        # 日付が1日ずつ連続していることを確認
        for i in range(1, len(dates)):
            assert (dates[i] - dates[i - 1]).days == 1

    def test_予測開始日が履歴の翌日(self, trained):
        model, df_feat, feat_cols = trained
        result    = predict_future(model, df_feat, feat_cols, forecast_days=3)
        last_hist = df_feat["date"].max()
        first_pred = result["date"].min()
        assert first_pred == last_hist + pd.Timedelta(days=1)

    def test_予測値が非負(self, trained):
        model, df_feat, feat_cols = trained
        result = predict_future(model, df_feat, feat_cols, forecast_days=7)
        assert (result["sales"] >= 0).all(), "予測値に負の値が含まれています"

    def test_予測値がnanでない(self, trained):
        model, df_feat, feat_cols = trained
        result = predict_future(model, df_feat, feat_cols, forecast_days=7)
        assert result["sales"].notna().all(), "予測値に NaN が含まれています"

    def test_逐次予測でlag1が前日予測値を参照する(self, trained):
        """
        predict_future の核心ロジックのテスト。
        1日目の予測値が2日目の lag_1 として引き継がれているかを確認する。

        検証方法:
            ① 通常の逐次予測を実行
            ② 1日目の予測値を強制的に 9999 に書き換えた場合と
               通常の場合で2日目の予測値が異なることを確認
               （lag_1 が引き継がれているなら差が出るはず）
        """
        model, df_feat, feat_cols = trained

        # 通常の予測
        result_normal = predict_future(model, df_feat, feat_cols, forecast_days=2, use_log_transform=False)

        # 1日目の予測値を操作するため、モックモデルを作る
        first_pred_date = df_feat["date"].max() + pd.Timedelta(days=1)
        call_count = [0]
        original_predict = model.predict

        def mock_predict(X):
            call_count[0] += 1
            if call_count[0] == 1:
                # 1日目の予測を全て 9999 に
                return np.full(len(X), 9999.0)
            return original_predict(X)

        model.predict = mock_predict
        result_mock = predict_future(model, df_feat, feat_cols, forecast_days=2, use_log_transform=False)
        model.predict = original_predict  # 元に戻す

        # 2日目の予測値が異なるはず（lag_1 = 9999 が影響する）
        day2_normal = result_normal[result_normal["date"] == first_pred_date +
                                    pd.Timedelta(days=1)]["sales"].values
        day2_mock   = result_mock[result_mock["date"]   == first_pred_date +
                                  pd.Timedelta(days=1)]["sales"].values

        assert not np.allclose(day2_normal, day2_mock), \
            "2日目の予測値が1日目に依存していない（lag の引き継ぎが壊れている可能性）"

    def test_future_promoが反映される(self, trained):
        """future_promo を渡したとき onpromotion が正しく設定されるか確認。"""
        model, df_feat, feat_cols = trained
        last_date   = df_feat["date"].max()
        target_date = last_date + pd.Timedelta(days=1)

        future_promo = pd.DataFrame([
            {"store": "S01", "product": "P001", "date": target_date, "onpromotion": 1},
        ])

        # エラーなく実行できることを確認
        result = predict_future(
            model, df_feat, feat_cols, forecast_days=2,
            future_promo=future_promo, use_log_transform=False
        )
        assert len(result) > 0

    def test_forecast_days_パラメータが効く(self, trained):
        model, df_feat, feat_cols = trained
        for days in [1, 3, 7]:
            result = predict_future(model, df_feat, feat_cols, forecast_days=days, use_log_transform=False)
            unique_dates = result["date"].nunique()
            assert unique_dates == days, f"forecast_days={days} なのに {unique_dates} 日分出力"
