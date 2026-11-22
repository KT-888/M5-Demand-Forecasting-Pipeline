"""
main.py
=======
時系列需要予測システム - 全体の司令塔（エントリーポイント）

処理の流れ:
    1. 前処理・特徴量生成
    2. Walkforward CV（精度の健全性チェック）
    3. train / valid / test の3分割
    4. Optuna でベストパラメータを探索
    5. train + valid で最終モデルを学習
    6. test データで公正に評価
    7. 未来7日間の売上を予測
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from config                    import RANDOM_STATE, LAGS, ROLLING_WINDOWS, CV_PARAMS
from pipeline.feature_pipeline import build_features, get_feature_col_names
from pipeline.splitter         import (
    split_train_valid_test_by_date,
    generate_walkforward_cv_folds,
)
from model.trainer             import search_best_params, train_on_full_data
from model.predictor           import predict_future
from model.evaluator           import evaluate
from utils.validator           import (
    validate_raw_input,       # ① 生データ検証
    validate_feature_schema,  # ② 特徴量生成後検証
    validate_features,        # ③ train/valid/test 整合性
    validate_predict_input,   # ④ 予測時整合性
)


def main(
    raw_df:      pd.DataFrame,
    cv_n_splits: int   = 3,
    valid_ratio: float = 0.2,
    test_ratio:  float = 0.2,
):
    """
    需要予測の全処理を順番に実行する。

    Parameters
    ----------
    raw_df       : 生データ。必須列: store / product / date / sales
                   任意列: price / stock / onpromotion
    cv_n_splits  : Walkforward CV の fold 数
    valid_ratio  : valid として使う期間の割合
    test_ratio   : test として使う期間の割合

    Returns
    -------
    forecast    : 未来7日間の予測結果 DataFrame
    final_model : 学習済みの LightGBM モデル
    eval_result : 評価結果の辞書
    """

    # =========================================================
    # 1. 前処理・特徴量生成
    # =========================================================
    print("\n" + "=" * 60)
    print("⚙  1. 前処理・特徴量生成")
    print("=" * 60)

    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ★ ① 生データのスキーマ検証（型・欠損・値域・重複）
    print("\n🔍 生データ検証:")
    validate_raw_input(df, raise_error=True)

    # float64 → float32 に変換してメモリを節約
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    if "onpromotion" not in df.columns:
        df["onpromotion"] = 0
    df["onpromotion"] = df["onpromotion"].fillna(0).astype(int)

    df = build_features(df)

    # ★ ② 特徴量生成後のスキーマ検証（期待列・型・inf・avail フラグ）
    print("\n🔍 特徴量スキーマ検証:")
    validate_feature_schema(df, raise_error=True)

    feat_cols = get_feature_col_names(df)
    print(f"📊 生成された特徴量の数: {len(feat_cols)} 個")

    warmup_days  = max(max(LAGS), max(ROLLING_WINDOWS))
    oldest_date  = df["date"].min()
    cutoff_date  = oldest_date + pd.Timedelta(days=warmup_days)
    df_for_model = df[df["date"] >= cutoff_date].copy()
    print(f"📉 先頭 {warmup_days} 日分を除外（ウォームアップ期間）")
    print(f"   除外期間: {oldest_date.date()} ～ {(cutoff_date - pd.Timedelta(days=1)).date()}")
    print(f"   学習開始: {cutoff_date.date()} 以降  {len(df_for_model):,} 行")

    # =========================================================
    # 2. Walkforward CV（データの健全性チェック）
    # =========================================================
    print("\n" + "=" * 60)
    print(f"🔀 2. Walkforward CV（{cv_n_splits} folds・健全性チェック）")
    print("   ※ 軽量固定パラメータで実行。最終精度とは別物。")
    print("=" * 60)

    cv_scores = []

    try:
        cv_fold_list = list(generate_walkforward_cv_folds(
            df_for_model, n_splits=cv_n_splits, valid_ratio=valid_ratio
        ))
    except ValueError as e:
        print(f"⚠️  CV をスキップします: {e}")
        cv_fold_list = []

    for fold_idx, (fold_train, fold_valid) in enumerate(cv_fold_list):
        X_tr = fold_train[feat_cols]
        y_tr = fold_train["sales"]
        X_va = fold_valid[feat_cols]
        y_va = fold_valid["sales"]

        # ★ ③ fold 間の整合性検証
        validate_features(X_tr, X_va, raise_error=False)

        dtrain     = lgb.Dataset(X_tr, y_tr)
        dvalid     = lgb.Dataset(X_va, y_va, reference=dtrain)
        fold_model = lgb.train(
            CV_PARAMS, dtrain,
            valid_sets=[dvalid],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        fold_preds = np.maximum(0, fold_model.predict(X_va))
        fold_mae   = mean_absolute_error(y_va, fold_preds)
        cv_scores.append(fold_mae)
        print(f"  Fold {fold_idx + 1} MAE: {fold_mae:.4f}")

    if cv_scores:
        cv_mean = np.mean(cv_scores)
        cv_std  = np.std(cv_scores)
        print(f"\n📊 CV MAE: {cv_mean:.4f} ± {cv_std:.4f}")
        if cv_std / (cv_mean + 1e-9) > 0.3:
            print("  ⚠️  fold 間のばらつきが大きいです（std/mean > 30%）。")
            print("     データの急激なトレンド変化などを確認してください。")

    # =========================================================
    # 3. train / valid / test の3分割
    # =========================================================
    print("\n" + "=" * 60)
    print("✂️  3. train / valid / test の3分割")
    print("=" * 60)

    train_df, valid_df, test_df = split_train_valid_test_by_date(
        df_for_model,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    X_train = train_df[feat_cols]
    y_train = train_df["sales"]
    X_valid = valid_df[feat_cols]
    y_valid = valid_df["sales"]
    X_test  = test_df[feat_cols]

    # ★ ③ train/valid/test 間の整合性検証
    print("\n🔍 分割データ整合性検証:")
    validate_features(X_train, X_valid, raise_error=False)
    validate_features(X_train, X_test,  raise_error=False)

    # =========================================================
    # 4. Optuna でベストパラメータを探索
    # =========================================================
    print("\n" + "=" * 60)
    print("🎯 4. Optuna ハイパーパラメータ最適化")
    print("=" * 60)

    best_params = search_best_params(X_train, y_train, X_valid, y_valid, n_trials=150)

    # =========================================================
    # 5. train + valid で最終モデルを学習
    # =========================================================
    print("\n" + "=" * 60)
    print("📦 5. 最終モデル学習（train + valid）")
    print("=" * 60)

    X_full = pd.concat([X_train, X_valid])
    y_full = pd.concat([y_train, y_valid])

    final_model = train_on_full_data(X_full, y_full, best_params)

    # =========================================================
    # 6. test データで公正に評価
    # =========================================================
    print("\n" + "=" * 60)
    print("📊 6. モデル評価（test データ・学習に使っていない公正な評価）")
    print("=" * 60)

    eval_result = evaluate(final_model, test_df, feat_cols, train_df, use_log_transform=True)

    if cv_scores:
        cv_mean   = float(np.mean(cv_scores))
        cv_std    = float(np.std(cv_scores))
        final_mae = eval_result["overall"]["mae"]
        eval_result["cv"] = {"mae_mean": cv_mean, "mae_std": cv_std, "scores": cv_scores}

        gap_ratio = abs(final_mae - cv_mean) / (cv_mean + 1e-9)
        if gap_ratio > 0.3:
            print(
                f"\n  ⚠️  CV MAE ({cv_mean:.4f}) と最終 MAE ({final_mae:.4f}) の乖離: {gap_ratio:.0%}\n"
                f"     CV は軽量固定パラメータなので差は想定内ですが、\n"
                f"     極端な乖離の場合はデータの時系列変化を確認してください。"
            )

    # =========================================================
    # 7. 未来7日間の売上を予測
    # =========================================================
    print("\n" + "=" * 60)
    print("🔮 7. 未来 7 日予測")
    print("=" * 60)

    # ★ ④ 予測前に直近データの整合性を検証
    print("\n🔍 予測入力データ検証:")
    validate_predict_input(df, feat_cols, raise_error=True)

    forecast = predict_future(final_model, df, feat_cols)

    print(f"\n📈 予測完了: {len(forecast):,} 行")
    print(f"📅 予測期間: {forecast['date'].min().date()} ～ {forecast['date'].max().date()}")
    print("\n📋 予測結果サンプル（先頭10行）:")
    print(forecast.head(10).to_string(index=False))

    return forecast, final_model, eval_result


# =========================================================
# 実行エントリーポイント
# =========================================================
if __name__ == "__main__":

    print("📦 M5 データを読み込み中...")

    # M5 loader
    from dataload_m5_subset import load_m5_subset

    raw_df = load_m5_subset()

    print(f"✅ M5　データ読み込み完了: {len(raw_df):,} 行")

    # メイン処理実行
    forecast, model, eval_result = main(raw_df)

    print("\n" + "=" * 60)
    print("🎉 すべての処理が完了しました！")
    print("=" * 60)

'''
# =========================================================
# 動作確認用ダミーデータ
# if __name__ == "__main__" とは:
#   このファイルを直接 python main.py で実行したときだけ動く。
#   他のファイルから import されたときは動かない。
# =========================================================
if __name__ == "__main__":
    print("📦 ダミーデータを生成中...")

    dates    = pd.date_range("2024-01-01", periods=180)  # 3分割できるよう180日
    stores   = [f"S{i:02d}" for i in range(1, 4)]        # S01, S02, S03
    products = [f"P{i:03d}" for i in range(1, 6)]        # P001〜P005

    records = [
        {
            "store":       store,
            "product":     product,
            "date":        date,
            "sales":       max(0, np.random.poisson(50) + np.random.randint(-10, 10)),
            "price":       1000 + np.random.randint(-100, 100),
            "stock":       np.random.randint(50, 200),
            "onpromotion": np.random.choice([0, 1], p=[0.9, 0.1]),
        }
        for store   in stores
        for product in products
        for date    in dates
    ]

    from dataload_m5_subset import load_m5_subset
    raw_df = load_m5_subset()
    print(f"✅ M5 loaderからデータ完了: {len(raw_df):,} 行")

    forecast, model, eval_result = main(raw_df)

    print("\n" + "=" * 60)
    print("🎉 すべての処理が完了しました！")
    print("=" * 60)
'''