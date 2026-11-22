# M5-Demand-Forecasting-Pipeline
M5 Forecasting データセットを用いた 実務向け需要予測パイプラインです。<br>
LightGBM + Optuna + Walkforward Validation を用いて、
店舗×商品の時系列需要予測を行います。

# 設計方針
実験用ノートブックから一歩進めて、**再現性・保守性・データ品質**を重視し、
モジュール分割した需要予測パイプラインにしました。

1. **時系列データのリーケージ防止**(`pipeline/splitter.py`、`features/`)
   - ランダム分割ではなく、日付の順序を保ったまま train / valid / test に分割します。
   - ラグ特徴量は `shift(lag)`、移動平均・標準偏差は `shift(1)` してから集計し、
     予測時点より未来の情報が入らないようにしています。

2. **入力データの検証を独立させた設計**(`utils/validator.py`)
   - 欠損率・型・重複・日付の連続性・期待する列の過不足を、
     学習の前に検査します。検証ロジックを1か所に集めています。

3. **単体テスト100件**(`tests/`、`pytest`で実行)
   - 特徴量生成・分割・検証・学習・予測・評価の各モジュールをテストしています。

# Key Features
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

使用データセット: M5 Forecasting Accuracy<br>
(Kaggle / ニコシア大学 主催: https://www.kaggle.com/c/m5-forecasting-accuracy)

本パイプラインは、M5 のうち **3店舗×10商品×1,913日**
(2011-01-29〜2016-04-24、57,390行)のサブセットを使用します。
店舗数・商品数は `dataload_m5_subset.py` の `N_STORES` / `N_PRODUCTS` で変更できます。

動作確認用に、`data/` にデータを同梱しています。<br>
出典: Zenodo掲載版(Kaggle提供, CC BY 4.0) https://zenodo.org/records/12636070

## データの展開

```
cd data
unzip sales_train_validation.zip
unzip sell_prices.zip
```

展開後の `data/` は次の構成になります。

```
data/
├── calendar.csv
├── sales_train_validation.csv
└── sell_prices.csv
```

# Project Structure
project/<br>
├── main.py<br>
├── config.py<br>
├── dataload_m5_subset.py  (data_loader)<br>
│<br>
├── data/<br>
│   ├── calendar.csv<br>
│   ├── sales_train_validation.zip (展開すると .csv)<br>
│   └── sell_prices.zip (展開すると .csv)<br>
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
├── model/<br>
│   ├── trainer.py<br>
│   ├── predictor.py<br>
│   └── evaluator.py<br>
├── tests/<br>
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
未来データ leakage を防止し、実運用に近い検証を行います。

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
- CV(軽量モデル)とtestのMAEが近い<br>
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

