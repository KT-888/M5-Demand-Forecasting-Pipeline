"""
tests/test_evaluator.py
========================
model/evaluator.py の評価指標計算が数学的に正しいかをテストする。

テストの方針:
    「既知の実績・予測を入れて計算結果を手計算で検証する」。
    特に RMSSE は計算式が複雑なため、手計算値との一致を必ず確認する。
    また「ナイーブ予測より良い場合 <1.0、悪い場合 >1.0」という
    RMSSE の本質的な性質もテストする。
"""

import numpy as np
import pandas as pd
import pytest

from model.evaluator import (
    _calc_mape_nonzero_only,
    _calc_smape,
    _calc_rmsse,
)


# =========================================================
# MAPE のテスト
# =========================================================

class TestCalcMapeNonzeroOnly:

    def test_完全一致はゼロ(self):
        y    = np.array([10.0, 20.0, 30.0])
        pred = np.array([10.0, 20.0, 30.0])
        assert _calc_mape_nonzero_only(y, pred) == pytest.approx(0.0)

    def test_手計算値と一致する(self):
        # |10-12|/10 = 0.2, |20-18|/20 = 0.1 → 平均 0.15
        y    = np.array([10.0, 20.0])
        pred = np.array([12.0, 18.0])
        assert _calc_mape_nonzero_only(y, pred) == pytest.approx(0.15)

    def test_実績ゼロ行を除外して計算する(self):
        # y=0 の行は除外 → |10-12|/10 = 0.2 のみ
        y    = np.array([10.0, 0.0])
        pred = np.array([12.0, 5.0])
        assert _calc_mape_nonzero_only(y, pred) == pytest.approx(0.2)

    def test_全行ゼロはnanを返す(self):
        y    = np.array([0.0, 0.0])
        pred = np.array([1.0, 2.0])
        assert np.isnan(_calc_mape_nonzero_only(y, pred))


# =========================================================
# sMAPE のテスト
# =========================================================

class TestCalcSmape:

    def test_完全一致はゼロ(self):
        y    = np.array([10.0, 20.0])
        pred = np.array([10.0, 20.0])
        assert _calc_smape(y, pred) == pytest.approx(0.0)

    def test_手計算値と一致する(self):
        # 2*|10-12|/(10+12) = 4/22 ≈ 0.1818
        # 2*|20-18|/(20+18) = 4/38 ≈ 0.1053
        # 平均 ≈ 0.1435
        y    = np.array([10.0, 20.0])
        pred = np.array([12.0, 18.0])
        expected = np.mean([4/22, 4/38])
        assert _calc_smape(y, pred) == pytest.approx(expected, rel=1e-5)

    def test_実績ゼロでもnanにならない(self):
        # ゼロ安全が sMAPE の利点
        y    = np.array([0.0, 10.0])
        pred = np.array([5.0, 10.0])
        result = _calc_smape(y, pred)
        assert not np.isnan(result)
        assert result >= 0.0

    def test_両方ゼロでもnanにならない(self):
        y    = np.array([0.0])
        pred = np.array([0.0])
        result = _calc_smape(y, pred)
        assert not np.isnan(result)
        assert result == pytest.approx(0.0, abs=1e-6)


# =========================================================
# RMSSE のテスト（最重要）
# =========================================================

class TestCalcRmsse:

    def test_完全一致は0(self):
        train_y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y       = np.array([60.0, 70.0])
        pred    = np.array([60.0, 70.0])  # 完全一致
        assert _calc_rmsse(y, pred, train_y) == pytest.approx(0.0)

    def test_手計算値と一致する(self):
        """
        手計算:
            train_y = [10, 20, 30]
            ナイーブ差分 = [20-10, 30-20] = [10, 10]
            分母 = mean([100, 100]) = 100

            y    = [40, 50]
            pred = [44, 46]
            誤差 = [40-44, 50-46] = [-4, 4]
            分子 = mean([16, 16]) = 16

            RMSSE = sqrt(16 / 100) = sqrt(0.16) = 0.4
        """
        train_y = np.array([10.0, 20.0, 30.0])
        y       = np.array([40.0, 50.0])
        pred    = np.array([44.0, 46.0])
        assert _calc_rmsse(y, pred, train_y) == pytest.approx(0.4, rel=1e-5)

    def test_ナイーブ予測より良い場合は1未満(self):
        """
        モデルがナイーブ予測（前日値そのまま）より良ければ RMSSE < 1.0。
        """
        train_y = np.array([10.0, 20.0, 15.0, 25.0, 18.0])
        y       = np.array([22.0, 19.0, 24.0])

        # ナイーブ予測: 前日値をそのまま使う
        naive_pred = np.array([18.0, 22.0, 19.0])
        # 完璧な予測
        perfect_pred = y.copy()

        rmsse_naive   = _calc_rmsse(y, naive_pred,   train_y)
        rmsse_perfect = _calc_rmsse(y, perfect_pred, train_y)

        assert rmsse_perfect < rmsse_naive, \
            "完璧な予測の RMSSE がナイーブ予測より大きい（計算式に誤りの疑い）"
        assert rmsse_perfect < 1.0, "完璧な予測の RMSSE が 1.0 以上（計算式に誤りの疑い）"

    def test_ナイーブ予測と同等なら1付近(self):
        """
        ナイーブ予測（前日値）を pred として渡すと RMSSE ≈ 1.0 になる。
        """
        train_y = np.array([10.0, 12.0, 11.0, 13.0, 10.0,
                            12.0, 14.0, 11.0, 13.0, 12.0])
        # 検証期間もナイーブ予測
        y    = np.array([11.0, 13.0, 12.0, 14.0])
        pred = np.array([12.0, 11.0, 13.0, 12.0])  # 1日前の y をそのまま

        rmsse = _calc_rmsse(y, pred, train_y)
        # ナイーブ予測なので 1.0 付近になるはず（厳密に 1.0 ではないがおおよそ）
        assert 0.5 < rmsse < 2.0, f"ナイーブ予測の RMSSE が期待範囲外: {rmsse}"

    def test_train_y_が1件以下はnanを返す(self):
        # 差分が計算できない
        assert np.isnan(_calc_rmsse(np.array([10.0]), np.array([10.0]), np.array([5.0])))

    def test_train_y_が全て同値はnanを返す(self):
        # 分母がゼロになる
        train_y = np.array([10.0, 10.0, 10.0])
        y       = np.array([10.0])
        pred    = np.array([10.0])
        assert np.isnan(_calc_rmsse(y, pred, train_y))

    def test_負の予測値でも計算できる(self):
        # np.maximum(0, pred) は evaluator 側でやるが、
        # _calc_rmsse 自体は負値を受け取っても壊れないことを確認
        train_y = np.array([10.0, 20.0, 30.0])
        y       = np.array([40.0])
        pred    = np.array([-5.0])
        result  = _calc_rmsse(y, pred, train_y)
        assert not np.isnan(result)
        assert result > 0


# =========================================================
# evaluate() の統合テスト（モデルなしで指標の整合性を確認）
# =========================================================

class TestEvaluateIntegration:

    def _make_booster_mock(self, preds: np.ndarray):
        """lgb.Booster の predict() と feature_importance() だけ実装したモック。"""
        class MockBooster:
            def __init__(self, p):
                self._preds = p
            def predict(self, X):
                return self._preds
            def feature_importance(self, importance_type="gain"):
                return np.ones(len(X_cols))

        X_cols = ["feat_a", "feat_b"]

        mock = MockBooster(preds)
        mock._feat_cols = X_cols
        return mock, X_cols

    def test_overall_mae_が手計算と一致する(self):
        from model.evaluator import evaluate

        # 実績と予測を作る
        stores   = ["S01", "S01"]
        products = ["P001", "P001"]
        dates    = pd.to_datetime(["2024-06-01", "2024-06-02"])
        y_true   = np.array([10.0, 20.0])
        y_pred   = np.array([12.0, 18.0])  # 誤差 2, 2 → MAE = 2.0

        valid_df = pd.DataFrame({
            "store": stores, "product": products,
            "date": dates,   "sales": y_true,
            "feat_a": [1.0, 1.0], "feat_b": [2.0, 2.0],
        })
        train_df = pd.DataFrame({
            "store": ["S01"] * 10, "product": ["P001"] * 10,
            "date": pd.date_range("2024-01-01", periods=10),
            "sales": np.arange(10, 20, dtype=float),
        })

        class MockBooster:
            def predict(self, X):
                return y_pred
            def feature_importance(self, importance_type="gain"):
                return np.array([1.0, 1.0])

        feat_cols = ["feat_a", "feat_b"]
        result    = evaluate(MockBooster(), valid_df, feat_cols, train_df, use_log_transform=False)

        assert result["overall"]["mae"] == pytest.approx(2.0)

    def test_overall_rmsse_が正の値で返る(self):
        from model.evaluator import evaluate

        n_valid = 5
        valid_df = pd.DataFrame({
            "store":   ["S01"] * n_valid,
            "product": ["P001"] * n_valid,
            "date":    pd.date_range("2024-04-01", periods=n_valid),
            "sales":   np.array([10.0, 12.0, 11.0, 13.0, 10.0]),
            "feat_a":  np.ones(n_valid),
            "feat_b":  np.ones(n_valid),
        })
        train_df = pd.DataFrame({
            "store":   ["S01"] * 20,
            "product": ["P001"] * 20,
            "date":    pd.date_range("2024-01-01", periods=20),
            "sales":   np.random.default_rng(42).uniform(8, 15, 20),
        })

        class MockBooster:
            def predict(self, X):
                return np.array([10.5, 11.5, 11.0, 12.5, 10.5])
            def feature_importance(self, importance_type="gain"):
                return np.array([1.0, 1.0])

        feat_cols = ["feat_a", "feat_b"]
        result    = evaluate(MockBooster(), valid_df, feat_cols, train_df, use_log_transform=False)

        rmsse = result["overall"]["rmsse"]
        assert not np.isnan(rmsse)
        assert rmsse > 0

    def test_by_store_にrmsse列が存在する(self):
        from model.evaluator import evaluate

        n = 6
        valid_df = pd.DataFrame({
            "store":   ["S01"] * 3 + ["S02"] * 3,
            "product": ["P001"] * n,
            "date":    pd.date_range("2024-04-01", periods=n),
            "sales":   np.ones(n) * 10,
            "feat_a":  np.ones(n),
        })
        train_df = pd.DataFrame({
            "store":   ["S01"] * 10 + ["S02"] * 10,
            "product": ["P001"] * 20,
            "date":    list(pd.date_range("2024-01-01", periods=10)) * 2,
            "sales":   np.ones(20) * 10,
        })

        class MockBooster:
            def predict(self, X):
                return np.ones(len(X)) * 10
            def feature_importance(self, importance_type="gain"):
                return np.array([1.0])

        result = evaluate(MockBooster(), valid_df, ["feat_a"], train_df, use_log_transform=False)
        assert "rmsse" in result["by_store"].columns
        assert "rmsse" in result["by_product"].columns
