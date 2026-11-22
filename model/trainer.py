"""
model/trainer.py
================
LightGBM モデルの「パラメータ探索」と「最終学習」を担当するモジュール。

【修正内容 (v2)】
    ★ log1p / expm1 変換の導入
    ★ categorical_feature の明示指定
    ★ Optuna の評価を single-valid → 内部 CV（3fold）に変更
       splitter に依存せず trainer 内部で時系列インデックス分割を実装。
       これにより Optuna の trial ごとに splitter の print が出なくなる。
"""

import numpy as np
import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

from config import RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)

FINAL_ROUND_SCALE = 1.1
MAX_FINAL_ROUNDS  = 5000
OPTUNA_CV_SPLITS  = 3    # Optuna 内部の CV fold 数
OPTUNA_N_TRIALS   = 100  # CV 化で1試行あたりの計算量が増えるため 150→100 に調整


def search_best_params(
    X_train:  pd.DataFrame,
    y_train:  pd.Series,
    X_valid:  pd.DataFrame,
    y_valid:  pd.Series,
    n_trials: int = OPTUNA_N_TRIALS,
) -> dict:
    """
    Optuna + 内部 CV（3fold）で LightGBM の最適パラメータを探索する。

    評価方法:
        train + valid を合わせて時系列順に 3fold CV で MAE を計算。
        単一 valid での評価と違い、どの時期でも安定したパラメータが選ばれる。

    log1p 変換:
        _run_one_optuna_trial_cv() の内部で log1p → 学習 → expm1 → MAE 計算。
        呼び出し元（main.py）は変換を意識しなくてよい。
    """
    X_all = pd.concat([X_train, X_valid]).reset_index(drop=True)
    y_all = pd.concat([y_train, y_valid]).reset_index(drop=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _run_one_optuna_trial_cv(
            trial, X_all, y_all, n_splits=OPTUNA_CV_SPLITS
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params.copy()
    best_params["best_iteration"] = study.best_trial.user_attrs["best_iteration"]

    print("\n🎉 最適パラメータ:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\n📈 Best CV MAE: {study.best_value:.4f}")

    return best_params


def train_on_full_data(
    X:           pd.DataFrame,
    y:           pd.Series,
    best_params: dict,
    cat_cols:    list = None,
) -> lgb.Booster:
    """
    search_best_params() で見つけたパラメータを使い、
    train + valid を合わせた全データで最終モデルを学習する。

    log1p 変換:
        y を log1p 変換してから学習する。
        予測・評価時は expm1 で元スケールに戻す。
        （predictor.py / evaluator.py の use_log_transform=True で対応）

    categorical_feature:
        cat_cols で渡された列を LightGBM にカテゴリ変数として明示指定。
        None の場合は object / category 型の列を自動検出する。
    """
    params = {
        **{k: v for k, v in best_params.items() if k != "best_iteration"},
        "objective":     "regression_l1",
        "metric":        "mae",
        "boosting_type": "gbdt",
        "random_state":  RANDOM_STATE,
        "verbosity":     -1,
    }

    raw_rounds = int(best_params.get("best_iteration", 800))
    # ★ best_iteration が極端に小さい場合（データ不足など）の下限保証
    #   最低 50 回は学習しないとモデルが underfitting になる
    raw_rounds = max(raw_rounds, 50)
    num_rounds = min(int(raw_rounds * FINAL_ROUND_SCALE), MAX_FINAL_ROUNDS)
    print(f"  ℹ️  最終モデル学習回数: {raw_rounds} × {FINAL_ROUND_SCALE} = {num_rounds} 回")

    # ★ log1p 変換
    y_log = np.log1p(y.values.astype(float))

    # ★ categorical_feature 自動検出
    if cat_cols is None:
        cat_cols = [c for c in X.columns if X[c].dtype in ("object", "category")]

    dtrain = lgb.Dataset(
        X, y_log,
        categorical_feature=cat_cols if cat_cols else "auto",
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        callbacks=[lgb.log_evaluation(period=200)],
    )
    return model


def _run_one_optuna_trial_cv(
    trial,
    X_all:    pd.DataFrame,
    y_all:    pd.Series,
    n_splits: int = 3,
) -> float:
    """
    Optuna が1回の試行ごとに呼び出す関数（内部 CV 版）。

    時系列順の expanding window で fold を作る。
    splitter に依存しないためログ出力が発生しない。

    log1p → 学習 → expm1 → MAE 計算のサイクルで評価する。
    """
    params = {
        "objective":        "regression_l1",
        "metric":           "mae",
        "boosting_type":    "gbdt",
        "random_state":     RANDOM_STATE,
        "verbosity":        -1,
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
        "max_depth":        trial.suggest_int("max_depth", 4, 12),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 15, 200),
        "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
        "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
    }

    n_total    = len(X_all)
    fold_size  = n_total // (n_splits + 1)
    fold_maes       = []
    best_iterations = []

    for fold in range(n_splits):
        # expanding window: train は先頭から fold+1 ブロック分、valid は次の1ブロック
        tr_end = fold_size * (fold + 1)
        va_end = fold_size * (fold + 2)
        va_end = min(va_end, n_total)

        if tr_end <= 0 or va_end <= tr_end:
            continue

        X_tr = X_all.iloc[:tr_end]
        y_tr = y_all.iloc[:tr_end]
        X_va = X_all.iloc[tr_end:va_end]
        y_va = y_all.iloc[tr_end:va_end]

        # ★ log1p 変換
        y_log_tr = np.log1p(y_tr.values.astype(float))
        y_log_va = np.log1p(y_va.values.astype(float))

        dtrain = lgb.Dataset(X_tr, y_log_tr, free_raw_data=False)
        dvalid = lgb.Dataset(X_va, y_log_va, reference=dtrain, free_raw_data=False)

        # ★ stopping_rounds をデータ量に応じて動的に計算する。
        #   固定値 200 だとデータが少ない場合（ダミーデータなど）に
        #   3回目で early stopping が発動して best_iteration=3 になる問題を防ぐ。
        #   経験則: max(50, train行数 // 100) が適切な下限。
        #   M5データ(11000行): max(50, 110) = 110
        #   ダミーデータ(453行): max(50, 4) = 50
        stopping_rounds = max(50, len(X_tr) // 100)

        model = lgb.train(
            params, dtrain,
            valid_sets=[dvalid],
            num_boost_round=3000,
            callbacks=[lgb.early_stopping(stopping_rounds, verbose=False)],
        )

        # ★ expm1 で元スケールに戻してから MAE を計算
        preds_orig = np.expm1(model.predict(X_va)).clip(min=0)
        fold_mae   = mean_absolute_error(y_va.values, preds_orig)

        fold_maes.append(fold_mae)
        best_iterations.append(model.best_iteration)

    if not fold_maes:
        trial.set_user_attr("best_iteration", 100)
        return float("inf")

    trial.set_user_attr("best_iteration", int(np.median(best_iterations)))
    return float(np.mean(fold_maes))
