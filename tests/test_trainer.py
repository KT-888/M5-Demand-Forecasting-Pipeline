"""
tests/test_trainer.py
======================
model/trainer.py の学習ロジックをテストする。

テストの方針:
    Optuna の 150 試行をテストで走らせると数分かかるため、
    ここでは n_trials=3 / num_boost_round=50 の軽量設定で
    「関数が正しく動くか・出力の型・形式が正しいか」を確認する。

    テストするのはモデルの「精度」ではなく「動作の正しさ」。
    精度テストは CI ではなく定期的な本番評価で行う。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import pytest

from model.trainer import train_on_full_data, search_best_params
from pipeline.feature_pipeline import build_features, get_feature_col_names


# =========================================================
# テスト用データ生成ヘルパー
# =========================================================

def _make_train_valid(n_days: int = 80):
    """train / valid 用の特徴量 DataFrame を生成する。"""
    dates    = pd.date_range("2024-01-01", periods=n_days)
    records  = [
        {"store": "S01", "product": "P001", "date": d,
         "sales": float(20 + i % 7), "onpromotion": 0}
        for i, d in enumerate(dates)
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df_feat   = build_features(df)
    feat_cols = get_feature_col_names(df_feat)

    cutoff   = df_feat["date"].min() + pd.Timedelta(days=28)
    df_model = df_feat[df_feat["date"] >= cutoff]

    split    = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split]
    valid_df = df_model.iloc[split:]

    X_train = train_df[feat_cols]
    y_train = train_df["sales"]
    X_valid = valid_df[feat_cols]
    y_valid = valid_df["sales"]

    return X_train, y_train, X_valid, y_valid


# =========================================================
# train_on_full_data のテスト
# =========================================================

class TestTrainOnFullData:

    def test_戻り値がlgb_Boosterである(self):
        X_train, y_train, X_valid, y_valid = _make_train_valid()
        X = pd.concat([X_train, X_valid])
        y = pd.concat([y_train, y_valid])
        best_params = {"best_iteration": 50, "num_leaves": 31, "learning_rate": 0.1}
        model = train_on_full_data(X, y, best_params)
        assert isinstance(model, lgb.Booster)

    def test_学習済みモデルで予測できる(self):
        X_train, y_train, X_valid, y_valid = _make_train_valid()
        X = pd.concat([X_train, X_valid])
        y = pd.concat([y_train, y_valid])
        best_params = {"best_iteration": 50, "num_leaves": 31, "learning_rate": 0.1}
        model  = train_on_full_data(X, y, best_params)
        preds  = model.predict(X_valid)
        assert len(preds) == len(X_valid)
        assert not np.isnan(preds).any()

    def test_予測値が有限値である(self):
        X_train, y_train, X_valid, y_valid = _make_train_valid()
        X = pd.concat([X_train, X_valid])
        y = pd.concat([y_train, y_valid])
        best_params = {"best_iteration": 50, "num_leaves": 31, "learning_rate": 0.1}
        model = train_on_full_data(X, y, best_params)
        preds = model.predict(X_valid)
        assert np.isfinite(preds).all()

    def test_best_iteration_キーが除外されてエラーにならない(self):
        """best_iteration は lgb.train に渡せないキー。除外されないとエラーになる。"""
        X_train, y_train, X_valid, y_valid = _make_train_valid()
        X = pd.concat([X_train, X_valid])
        y = pd.concat([y_train, y_valid])
        # best_iteration だけ持つ params（他のパラメータはデフォルト）
        best_params = {"best_iteration": 30}
        # エラーなく動けばOK
        model = train_on_full_data(X, y, best_params)
        assert isinstance(model, lgb.Booster)

    def test_num_roundsがFINAL_ROUND_SCALEで増加する(self, capsys):
        """best_iteration × 1.1 の回数で学習されることをログから確認。"""
        X_train, y_train, X_valid, y_valid = _make_train_valid()
        X = pd.concat([X_train, X_valid])
        y = pd.concat([y_train, y_valid])
        best_params = {"best_iteration": 100, "num_leaves": 31, "learning_rate": 0.1}
        train_on_full_data(X, y, best_params)
        captured = capsys.readouterr()
        # 110 回（100 × 1.1）が出力に含まれるはず
        assert "110" in captured.out


# =========================================================
# search_best_params のテスト（軽量: n_trials=3）
# =========================================================

class TestSearchBestParams:

    @pytest.fixture(scope="class")
    def best_params(self):
        """クラス内で1回だけ Optuna を走らせてキャッシュ。"""
        X_train, y_train, X_valid, y_valid = _make_train_valid()
        return search_best_params(
            X_train, y_train, X_valid, y_valid,
            n_trials=3,  # テスト用に最小回数
        )

    def test_戻り値が辞書である(self, best_params):
        assert isinstance(best_params, dict)

    def test_best_iterationキーが含まれる(self, best_params):
        assert "best_iteration" in best_params

    def test_best_iterationが正の整数(self, best_params):
        assert isinstance(best_params["best_iteration"], int)
        assert best_params["best_iteration"] > 0

    def test_必須パラメータが含まれる(self, best_params):
        required = ["learning_rate", "num_leaves", "max_depth",
                    "feature_fraction", "bagging_fraction"]
        for key in required:
            assert key in best_params, f"'{key}' が best_params に含まれていない"

    def test_learning_rateが範囲内(self, best_params):
        assert 0.005 <= best_params["learning_rate"] <= 0.15

    def test_num_leavesが範囲内(self, best_params):
        assert 31 <= best_params["num_leaves"] <= 255

    def test_best_paramsでモデルが学習できる(self, best_params):
        """search_best_params の結果を train_on_full_data に渡して動くことを確認。"""
        X_train, y_train, X_valid, y_valid = _make_train_valid()
        X = pd.concat([X_train, X_valid])
        y = pd.concat([y_train, y_valid])
        model = train_on_full_data(X, y, best_params)
        assert isinstance(model, lgb.Booster)
