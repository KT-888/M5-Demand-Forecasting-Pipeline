"""
M5 loader

M5 Forecasting を
既存 pipeline 用 long format に変換する。
"""

from pathlib import Path

import numpy as np
import pandas as pd


# project/data/
DATA_DIR = Path("data")

N_STORES = 3
N_PRODUCTS = 10


def load_m5_subset():

    print("loading M5...")

    # -----------------------------
    # csv
    # -----------------------------

    calendar = pd.read_csv(
        DATA_DIR / "calendar.csv"
    )

    sales = pd.read_csv(
        DATA_DIR / "sales_train_validation.csv"
    )

    prices = pd.read_csv(
        DATA_DIR / "sell_prices.csv"
    )

    # -----------------------------
    # subset
    # -----------------------------
    # 最初は小さく回す
    # -----------------------------

    stores = (
        sales["store_id"]
        .drop_duplicates()
        .sort_values()
        .head(N_STORES)
    )

    sales = sales[
        sales["store_id"].isin(stores)
    ].copy()

    # -----------------------------
    # d_ columns
    # -----------------------------

    d_cols = [
        c for c in sales.columns
        if c.startswith("d_")
    ]

    # -----------------------------
    # 売上上位商品
    # -----------------------------
    # intermittent demand を避ける
    # -----------------------------

    item_sales = (
        sales[d_cols]
        .sum(axis=1)
    )

    item_sales = (
        pd.DataFrame({
            "item_id": sales["item_id"],
            "sales": item_sales,
        })
        .groupby("item_id")["sales"]
        .sum()
        .sort_values(ascending=False)
    )

    top_items = (
        item_sales
        .head(N_PRODUCTS)
        .index
    )

    sales = sales[
        sales["item_id"].isin(top_items)
    ].copy()

    # -----------------------------
    # wide → long
    # -----------------------------
    # d_1 d_2 ...
    # ↓
    # date / sales
    # -----------------------------

    df = sales.melt(
        id_vars=[
            "store_id",
            "item_id",
            "state_id",
        ],
        value_vars=d_cols,
        var_name="d",
        value_name="sales",
    )

    # -----------------------------
    # calendar merge
    # -----------------------------

    calendar["date"] = pd.to_datetime(
        calendar["date"]
    )

    df = df.merge(
        calendar[
            [
                "d",
                "date",
                "wm_yr_wk",
            ]
        ],
        on="d",
        how="left",
    )

    # -----------------------------
    # price merge
    # -----------------------------

    df = df.merge(
        prices[
            [
                "store_id",
                "item_id",
                "wm_yr_wk",
                "sell_price",
            ]
        ],
        on=[
            "store_id",
            "item_id",
            "wm_yr_wk",
        ],
        how="left",
    )

    # M5 は未販売週に NaN がある
    df["sell_price"] = (
        df["sell_price"]
        .fillna(df["sell_price"].median())
    )

    # -----------------------------
    # pipeline 用列名
    # -----------------------------

    df = df.rename(
        columns={
            "store_id": "store",
            "item_id": "product",
            "sell_price": "price",
        }
    )

    # pipeline 互換 dummy
    df["onpromotion"] = 0
    df["stock"] = 9999

    # -----------------------------
    # keep columns
    # -----------------------------

    df = df[
        [
            "store",
            "product",
            "date",
            "sales",
            "price",
            "onpromotion",
            "stock",
        ]
    ]

    # lag / rolling 用
    df = df.sort_values(
        ["store", "product", "date"]
    ).reset_index(drop=True)

    # 軽量化
    df["sales"] = df["sales"].astype(np.float32)
    df["price"] = df["price"].astype(np.float32)

    print(df.head())
    print(df.shape)
    print(df.isnull().sum())
    print(df.dtypes)

    return df


if __name__ == "__main__":

    df = load_m5_subset()

    print(df.shape)