# M5-Demand-Forecasting-Pipeline
M5 Forecasting データセットを用いた 実務向け需要予測パイプラインです。<br>
LightGBM + Optuna + Walkforward Validation を用いて、
店舗×商品の時系列需要予測を行います。

# Features
•	LightGBM による需要予測<br>
•	WalkforwardCrossValidation<br>
•	Optuna によるハイパーパラメータ最適化<br>
•	leakage を避けた特徴量生成<br>
•	lag / rolling 特徴量<br>
•	train / valid / test 分離<br>
•	future forecasting<br>
•	feature importance 可視化<br>
•	M5 dataset 対応 loader<br>

# Dataset
使用データセット:<br>
•	M5 Forecasting Accuracy<br>

Kaggle:<br>
M5 Forecasting Accuracy<br>
10store×20product×365日×3年<br>

必要ファイル:<br>
data/<br>
├── calendar.csv<br>
├── sales_train_validation.csv<br>
└── sell_prices.csv<br>

# Project Structure
project/<br>
├── main.py<br>
├── dataload_m5_subset.py  (data_loader)<br>
│<br>
├── data/<br>
│   ├── calendar.csv<br>
│   ├── sales_train_validation.csv<br>
│   └── sell_prices.csv<br>
│<br>
├── pipeline/<br>
│   ├── feature_pipeline.py<br>
│   └── splitter.py<br>
│<br>
├── features/<br>
│   ├── lag.py<br>
│   ├── rolling.py<br>
│   └── calendar.py<br>
│<br>
├── models/<br>
│   ├── train.py<br>
│   ├── predict.py<br>
│   └── evaluator.py<br>
│<br>
└── utils/<br>
    └── validator.py<br>

# Pipeline
## 1. Data Loading

M5 データセットを long format に変換。

生成列:<br>
•	store<br>
•	product<br>
•	date<br>
•	sales<br>
•	price<br>
•	onpromotion<br>
•	stock<br>

## 2. Feature Engineering
主な特徴量:<br>
•	lag_1<br>
•	lag_7<br>
•	lag_14<br>
•	ma7<br>
•	ma28<br>
•	std7<br>
•	曜日<br>
•	月<br>
•	価格特徴量<br>

## 3. Validation
Walkforward Cross Validation を採用。<br>
未来データ leakage を防止し、<br>
実運用に近い検証を行います。

## 4. Hyperparameter Optimization
Optuna を用いて LightGBM パラメータを最適化。

最適化対象:<br>
•	learning_rate<br>
•	num_leaves<br>
•	max_depth<br>
•	feature_fraction<br>
•	bagging_fraction<br>
•	regularization<br>

## 5. Forecasting
未来 N 日予測に対応。<br>

予測結果（例）<br>
-------------<br>
Walkforward CV MAE : 7.81 ± 0.33<br>

テスト MAE         :  7.9044  （平均 7.9 個のズレ）<br>
テスト MAPE        : 40.5570%  ※ 実績ゼロの日を除いた誤差率<br>
テスト sMAPE       : 47.8016%  ※ ゼロ安全な対称MAPE<br>
テスト RMSSE       : 0.5706  ※ <1.0 でナイーブ予測より良い<br>
•	売上ゼロの日が 21〜42% ある（間欠需要）<br>
•	平均の 10〜13 倍売れる突発日がある<br>
•	RMSSE 0.57 = ナイーブ予測より 43% 改善しており、モデルとして十分機能している<br>

特徴<br>
----<br>
- CV と test の整合性が高く、過学習しにくい構成<br>
- データリークを防止した時系列検証<br>
- Walkforward Validation を採用<br>
- 未来 7 日間の需要予測に対応<br>

## Tech Stack
•	Python<br>
•	pandas<br>
•	numpy<br>
•	LightGBM<br>
•	Optuna<br>
•	scikit-learn<br>

## Notes
このリポジトリは、<br>
•	時系列 leakage 回避<br>
•	実務向け validation<br>
•	再利用可能 pipeline<br>
を重視しています。<br>

## Author

需要予測・在庫最適化や使用電力削減向け ML pipeline 開発用プロジェクト。  
電力需要予測・電力コスト削減にも対応可。

