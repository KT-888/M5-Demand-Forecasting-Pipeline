"""
model/evaluator.py
==================
学習済みモデルの予測精度を評価するモジュール。

評価指標:
    MAE   : |実績 - 予測| の平均。単位が売上個数なので直感的に理解しやすい。
    MAPE  : |実績 - 予測| / 実績 の平均。実績ゼロの行は除外して計算。
    sMAPE : 2 × |実績 - 予測| / (|実績| + |予測| + ε) の平均。ゼロ安全な MAPE 改良版。
    RMSSE : Root Mean Squared Scaled Error。M5コンペ標準の誤差指標。
            RMSSE < 1.0 → ナイーブ予測より良い（目標）

【修正内容 (v2)】
    groupby().apply() を pandas 3.0 対応に修正（include_groups=False）。

【修正内容 (v3)】
    RMSSE を追加。evaluate() のシグネチャに train_df を追加。

【修正内容 (v4)】
    groupby+apply → groupby+agg に変更。
    apply はグループごとに Python 関数を1回ずつ呼ぶため、
    店舗数・商品数が増えると線形に遅くなる。
    agg はベクトル化演算を使うため大幅に高速。
    ただし RMSSE は train_df の参照が必要なため apply のまま維持し、
    MAE / MAPE / sMAPE の3指標のみ agg 化した。
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def evaluate(
    model:             lgb.Booster,
    valid_df:          pd.DataFrame,
    feat_cols:         list,
    train_df:          pd.DataFrame,
    use_log_transform: bool = True,
) -> dict:
    """
    検証データで予測精度を評価し、結果を辞書で返す。

    Parameters
    ----------
    model     : train_on_full_data() で作った学習済みモデル
    valid_df  : 検証用 DataFrame（test_df を渡す）
    feat_cols : モデルに渡す特徴量の列名リスト
    train_df          : 学習用 DataFrame。RMSSE の分母計算に使う。
                        必ず train_df を渡すこと（valid/test を渡すとデータリーク）。
    use_log_transform : True のとき model.predict() の結果に expm1 を適用する。
                        train_on_full_data() は log1p 変換して学習するため
                        デフォルト True。log 変換なしのモデルには False を指定。

    Returns
    -------
    {
        "overall":            {"mae", "mape", "smape", "rmsse"},
        "by_store":           DataFrame,
        "by_product":         DataFrame,
        "feature_importance": DataFrame,
    }
    """
    X_valid = valid_df[feat_cols]
    y_valid = valid_df["sales"]

    raw_preds = model.predict(X_valid)
    # ★ log1p モデルの場合は expm1 で元スケールに戻す
    preds = (np.expm1(raw_preds) if use_log_transform else np.asarray(raw_preds)).clip(min=0)
    y_arr = y_valid.values

    # ---- 全体の精度計算 ----
    overall_mae   = mean_absolute_error(y_arr, preds)
    overall_mape  = _calc_mape_nonzero_only(y_arr, preds)
    overall_smape = _calc_smape(y_arr, preds)
    overall_rmsse = _calc_rmsse(y_arr, preds, train_df["sales"].values)

    print(f"\n✅ 検証データの評価結果:")
    print(f"  MAE          : {overall_mae:.4f}  （平均 {overall_mae:.1f} 個のズレ）")
    print(f"  MAPE（y>0）  : {overall_mape:.4%}  ※ 実績ゼロの日を除いた誤差率")
    print(f"  sMAPE（全行）: {overall_smape:.4%}  ※ ゼロ安全な対称MAPE")
    print(f"  RMSSE        : {overall_rmsse:.4f}  ※ <1.0 でナイーブ予測より良い")

    # ---- 店舗別・商品別の精度計算 ----
    result_df = valid_df[["store", "product"]].copy()
    result_df["actual"]    = y_arr
    result_df["predicted"] = preds
    result_df["abs_error"] = np.abs(y_arr - preds)
    result_df["nonzero"]   = (y_arr > 0).astype(int)
    # ★ np.where は両辺を先に計算するため y=0 行で ÷0 RuntimeWarning が出る
    #   safe_denom で y=0 の行の分母を 1.0 に置き換えて警告を抑制する
    # 実績ゼロの行による NumPy の一時的なゼロ除算警告を完全に無視する
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_denom = np.where(y_arr > 0, y_arr, 1.0)
        result_df["mape_num"]  = np.where(y_arr > 0, np.abs(y_arr - preds) / safe_denom, np.nan)
        result_df["smape_num"] = 2.0 * np.abs(y_arr - preds) / (np.abs(y_arr) + np.abs(preds) + 1e-9)

    # ★ MAE / MAPE / sMAPE は agg でベクトル化（apply より高速）
    def _agg_metrics(g):
        mae   = g["abs_error"].mean()
        mape  = g["mape_num"].mean()   # NaN は自動でスキップされる
        smape = g["smape_num"].mean()
        return pd.Series({"mae": mae, "mape": mape, "smape": smape})

    by_store_base = (
        result_df.groupby("store")
        .apply(_agg_metrics, include_groups=False)
        .reset_index()
    )
    by_product_base = (
        result_df.groupby("product")
        .apply(_agg_metrics, include_groups=False)
        .reset_index()
    )

    # ★ RMSSE は train_df の参照が必要なため別途 apply で計算して結合
    store_rmsse = (
        result_df.groupby("store")
        .apply(
            lambda g: pd.Series({"rmsse": _calc_rmsse(
                g["actual"].values, g["predicted"].values,
                train_df[train_df["store"] == g.name]["sales"].values,
            )}),
            include_groups=False,
        )
        .reset_index()
    )
    product_rmsse = (
        result_df.groupby("product")
        .apply(
            lambda g: pd.Series({"rmsse": _calc_rmsse(
                g["actual"].values, g["predicted"].values,
                train_df[train_df["product"] == g.name]["sales"].values,
            )}),
            include_groups=False,
        )
        .reset_index()
    )

    by_store = (
        by_store_base.merge(store_rmsse, on="store")
        .sort_values("mae", ascending=False)
    )
    by_product = (
        by_product_base.merge(product_rmsse, on="product")
        .sort_values("mae", ascending=False)
    )

    feature_importance = (
        pd.DataFrame({
            "feature":    feat_cols,
            "importance": model.feature_importance(importance_type="gain"),
        })
        .sort_values("importance", ascending=False)
    )

    print("\n📊 店舗別評価（MAE 降順）:")
    print(by_store.head(5).to_string(index=False))
    print("\n📊 商品別評価（MAE 降順 上位5商品）:")
    print(by_product.head(5).to_string(index=False))
    print("\n📊 特徴量の重要度 Top 10:")
    print(feature_importance.head(10).to_string(index=False))

    return {
        "overall":            {
            "mae":   overall_mae,
            "mape":  overall_mape,
            "smape": overall_smape,
            "rmsse": overall_rmsse,
        },
        "by_store":           by_store,
        "by_product":         by_product,
        "feature_importance": feature_importance,
    }


# =========================================================
# 内部関数
# =========================================================

def _calc_rmsse(
    y:       np.ndarray,
    pred:    np.ndarray,
    train_y: np.ndarray,
) -> float:
    """
    RMSSE = sqrt( mean((実績 - 予測)²) / mean((train_t - train_{t-1})²) )

    分母がゼロ（学習データが全て同値）または train_y が1件以下の場合は nan を返す。
    """
    if len(train_y) < 2:
        return float("nan")
    numerator    = np.mean((y - pred) ** 2)
    naive_errors = train_y[1:] - train_y[:-1]
    denominator  = np.mean(naive_errors ** 2)
    if denominator == 0:
        return float("nan")
    return float(np.sqrt(numerator / denominator))


def _calc_mape_nonzero_only(y: np.ndarray, pred: np.ndarray) -> float:
    """実績 > 0 の行だけで MAPE を計算する。÷0 防止。"""
    nonzero_mask = y > 0
    if nonzero_mask.sum() == 0:
        return float("nan")
    y_nz    = y[nonzero_mask]
    pred_nz = pred[nonzero_mask]
    return float(np.mean(np.abs((y_nz - pred_nz) / y_nz)))


def _calc_smape(y: np.ndarray, pred: np.ndarray, eps: float = 1e-9) -> float:
    """対称 MAPE。実績ゼロでも nan にならない。"""
    denominator = np.abs(y) + np.abs(pred) + eps
    return float(np.mean(2.0 * np.abs(y - pred) / denominator))
