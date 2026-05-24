# M5-Demand-Forecasting-Pipeline
M5 Forecasting データセットを用いた 実務向け需要予測パイプラインです。
LightGBM + Optuna + Walkforward Validation を用いて、
店舗×商品の時系列需要予測を行います。

# Features
LightGBM による需要予測
Walkforward Cross Validation
Optuna によるハイパーパラメータ最適化
leakage を避けた特徴量生成
lag / rolling 特徴量
train / valid / test 分離
future forecasting
feature importance 可視化
M5 dataset 対応 loader

# Dataset
使用データセット:
M5 Forecasting Accuracy

Kaggle:M5 Forecasting Accuracy - Kaggle

必要ファイル:

data/
├── calendar.csv
├── sales_train_validation.csv
└── sell_prices.csv
Project Structure
project/
│
├── data/
│   ├── calendar.csv
│   ├── sales_train_validation.csv
│   └── sell_prices.csv
│
├── features/
│   ├── lag.py
│   ├── rolling.py
│   └── calendar_feature.py
│
├── validation/
│   └── walkforward.py
│
├── training/
│   ├── train.py
│   └── optuna_tuning.py
│
├── inference/
│   └── predict_future.py
│
├── data/load_m5_subset.py
│
└── main.py

# Pipeline
## 1. Data Loading

M5 データセットを long format に変換。

生成列:

store
product
date
sales
price
onpromotion
stock

## 2. Feature Engineering
主な特徴量:
lag_1
lag_7
lag_14
ma7
ma28
std7
曜日
月
価格特徴量

## 3. Validation
Walkforward Cross Validation を採用。
未来データ leakage を防止し、
実運用に近い検証を行います。

## 4. Hyperparameter Optimization
Optuna を用いて LightGBM パラメータを最適化。

最適化対象:
learning_rate
num_leaves
max_depth
feature_fraction
bagging_fraction
regularization

## 5. Forecasting
未来 N 日予測に対応。

Example Result
CV MAE: 7.05 ± 0.50

Test MAE : 7.15
MAPE     : 15.7%
sMAPE    : 14.6%

## Tech Stack
Python
pandas
numpy
LightGBM
Optuna
scikit-learn
Notes

このリポジトリは、

時系列 leakage 回避
実務向け validation
再利用可能 pipeline

を重視しています。

Author

需要予測・在庫最適化向け ML pipeline 開発用プロジェクト。
