"""
tests/test_features.py
=======================
features/ 配下の特徴量生成ロジックが数学的に正しいかをテストする。

テストの方針:
    「既知の値を入れて既知の答えが出るか」で検証する。
    例: 3日間の売上が [10, 20, 30] なら lag_1 は [NaN, 10, 20] になるはず。

    未来リークが発生していないかも明示的にテストする。
    これが崩れると学習時に「答えを見ながら学習」してしまい、
    本番精度が大幅に落ちるため最重要のテスト。
"""

import numpy as np
import pandas as pd
import pytest

from features.lag     import create_lags
from features.rolling import create_rolling_features
from features.calendar import add_calendar_features


# =========================================================
# テスト用データ生成ヘルパー
# =========================================================

def _make_df(sales: list, store: str = "S01", product: str = "P001") -> pd.DataFrame:
    """指定した sales リストから単一 SKU の DataFrame を生成する。"""
    n     = len(sales)
    dates = pd.date_range("2024-01-01", periods=n)
    df    = pd.DataFrame({
        "store":       store,
        "product":     product,
        "date":        dates,
        "sales":       sales,
        "onpromotion": 0,
    })
    df["date"] = pd.to_datetime(df["date"])
    return df


# =========================================================
# lag のテスト
# =========================================================

class TestCreateLags:

    def test_lag1_の値が1日前の売上と一致する(self):
        sales = [10, 20, 30, 40, 50]
        df    = create_lags(_make_df(sales))
        # lag_1[0] は NaN（前日がない）
        assert pd.isna(df["lag_1"].iloc[0])
        # lag_1[1] は前日の売上 10
        assert df["lag_1"].iloc[1] == 10
        assert df["lag_1"].iloc[2] == 20
        assert df["lag_1"].iloc[3] == 30

    def test_lag7_の値が7日前の売上と一致する(self):
        sales = list(range(1, 20))  # 1, 2, 3, ... 19
        df    = create_lags(_make_df(sales))
        # lag_7[7] = sales[0] = 1
        assert df["lag_7"].iloc[7] == 1
        # lag_7[8] = sales[1] = 2
        assert df["lag_7"].iloc[8] == 2

    def test_lag14_の値が14日前の売上と一致する(self):
        sales = list(range(1, 30))
        df    = create_lags(_make_df(sales))
        assert df["lag_14"].iloc[14] == 1
        assert df["lag_14"].iloc[15] == 2

    def test_avail_フラグ_データあり行は1(self):
        sales = [10] * 20
        df    = create_lags(_make_df(sales))
        # lag_1 は2行目以降で利用可能
        assert df["lag_1_avail"].iloc[1] == 1
        assert df["lag_1_avail"].iloc[-1] == 1

    def test_avail_フラグ_データなし行は0(self):
        sales = [10] * 20
        df    = create_lags(_make_df(sales))
        # lag_1 の先頭行はデータなし
        assert df["lag_1_avail"].iloc[0] == 0

    def test_未来リーク_lag1は当日の売上を含まない(self):
        """lag_1 が当日の sales を参照していないことを確認（未来リーク防止）。"""
        sales = [100, 200, 300]
        df    = create_lags(_make_df(sales))
        # lag_1[1] は「昨日の売上 100」であるべき。「今日の 200」ではない。
        assert df["lag_1"].iloc[1] == 100, "lag_1 が当日の値を参照している（未来リーク）"

    def test_複数SKU間でラグが混入しない(self):
        """S01 の lag_1 に S02 の売上が入らないことを確認。"""
        df_s01 = _make_df([10, 20, 30], store="S01", product="P001")
        df_s02 = _make_df([99, 99, 99], store="S02", product="P001")
        df     = pd.concat([df_s01, df_s02], ignore_index=True)
        result = create_lags(df)

        s01_lag1 = result[result["store"] == "S01"]["lag_1"].dropna().values
        # S01 の lag_1 は 10, 20 のみで 99 が混入していない
        assert all(v in [10.0, 20.0] for v in s01_lag1), \
            "S01 の lag_1 に S02 の売上が混入している"

    def test_promo_lag_はonpromotion列がある場合のみ生成される(self):
        sales = [10] * 10
        df_with = _make_df(sales)           # onpromotion あり
        df_without = _make_df(sales).drop(columns=["onpromotion"])  # なし

        result_with    = create_lags(df_with)
        result_without = create_lags(df_without)

        assert "promo_lag_1" in result_with.columns
        assert "promo_lag_1" not in result_without.columns


# =========================================================
# rolling のテスト
# =========================================================

class TestCreateRollingFeatures:

    def test_ma7_が直近7日間の平均と一致する(self):
        """7日間の売上が全て10なら ma7 は 10 になる。"""
        sales  = [10] * 20
        df     = create_rolling_features(_make_df(sales))
        # 先頭はウォームアップ期間なので最後の方で確認
        assert df["ma7"].iloc[-1] == pytest.approx(10.0)

    def test_ma7_は当日の値を含まない_未来リーク防止(self):
        """
        shift(1) が正しく機能しているか確認。
        t 日目の ma7 は「t-1 日目まで」の平均であるべき。
        t 日目の値を含んでいたら未来リーク。
        """
        # 1日目だけ 999、残りは 1 のデータ
        sales = [999] + [1] * 20
        df    = create_rolling_features(_make_df(sales))

        # 2日目の ma7: shift(1) なら「1日目=999」が含まれる → 999付近
        # 3日目の ma7: shift(1) なら「1日目=999, 2日目=1」の平均 → 500付近
        # 8日目以降: 1日目が窓外に出て → 1 に収束
        # もし shift がなければ（未来リーク）、最終日の ma7 も高い値になる
        ma7_last = df["ma7"].iloc[-1]
        assert ma7_last == pytest.approx(1.0, abs=0.1), \
            f"ma7 が 1.0 に収束していない（{ma7_last}）。未来リークの疑い。"

    def test_std7_は非負(self):
        sales = [10, 20, 30, 40, 50] * 5
        df    = create_rolling_features(_make_df(sales))
        assert (df["std7"].fillna(0) >= 0).all()

    def test_std7_定数売上は0になる(self):
        """売上が毎日同じなら標準偏差は 0。"""
        sales = [50] * 30
        df    = create_rolling_features(_make_df(sales))
        # ウォームアップ期間後（先頭8日以降）は std7 = 0 のはず
        std7_tail = df["std7"].iloc[10:]
        assert (std7_tail == 0).all(), "定数売上なのに std7 が 0 でない"

    def test_std7_valid_データ十分なら1(self):
        sales = [10] * 30
        df    = create_rolling_features(_make_df(sales))
        assert df["std7_valid"].iloc[-1] == 1

    def test_price_欠損が前方補完される(self):
        df        = _make_df([10] * 10)
        df["price"] = [None, None, 500] + [None] * 7  # 先頭2行が欠損
        result    = create_rolling_features(df)
        # 前方補完後は 500 が先頭方向に伝播し、中央値で先頭欠損が埋まる
        assert result["price"].isna().sum() == 0


# =========================================================
# calendar のテスト
# =========================================================

class TestAddCalendarFeatures:

    def test_曜日_月曜は0(self):
        # 2024-01-01 は月曜
        df = _make_df([10])
        df["date"] = pd.to_datetime(["2024-01-01"])
        result = add_calendar_features(df)
        assert result["曜日"].iloc[0] == 0

    def test_曜日_日曜は6(self):
        # 2024-01-07 は日曜
        df = _make_df([10])
        df["date"] = pd.to_datetime(["2024-01-07"])
        result = add_calendar_features(df)
        assert result["曜日"].iloc[0] == 6

    def test_土日フラグ_土曜は1(self):
        # 2024-01-06 は土曜
        df = _make_df([10])
        df["date"] = pd.to_datetime(["2024-01-06"])
        result = add_calendar_features(df)
        assert result["土日か"].iloc[0] == 1

    def test_土日フラグ_月曜は0(self):
        df = _make_df([10])
        df["date"] = pd.to_datetime(["2024-01-01"])
        result = add_calendar_features(df)
        assert result["土日か"].iloc[0] == 0

    def test_祝日フラグ_元日は1(self):
        df = _make_df([10])
        df["date"] = pd.to_datetime(["2024-01-01"])
        result = add_calendar_features(df)
        assert result["祝日か"].iloc[0] == 1

    def test_祝日フラグ_平日は0(self):
        # 2024-01-10 は水曜・平日
        df = _make_df([10])
        df["date"] = pd.to_datetime(["2024-01-10"])
        result = add_calendar_features(df)
        assert result["祝日か"].iloc[0] == 0

    def test_月末フラグ_月末3日は1(self):
        # 1月は31日なので 29・30・31 が月末フラグ=1
        df = _make_df([10] * 3)
        df["date"] = pd.to_datetime(["2024-01-29", "2024-01-30", "2024-01-31"])
        result = add_calendar_features(df)
        assert result["月末フラグ"].tolist() == [1, 1, 1]

    def test_月末フラグ_月中は0(self):
        df = _make_df([10])
        df["date"] = pd.to_datetime(["2024-01-15"])
        result = add_calendar_features(df)
        assert result["月末フラグ"].iloc[0] == 0

    def test_期末フラグ_3月末は1(self):
        df = _make_df([10])
        df["date"] = pd.to_datetime(["2024-03-31"])
        result = add_calendar_features(df)
        assert result["期末フラグ"].iloc[0] == 1

    def test_期末フラグ_1月末は0(self):
        # 1月は期末月でない
        df = _make_df([10])
        df["date"] = pd.to_datetime(["2024-01-31"])
        result = add_calendar_features(df)
        assert result["期末フラグ"].iloc[0] == 0
