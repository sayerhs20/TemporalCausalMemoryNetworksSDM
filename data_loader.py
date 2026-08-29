import numpy as np
import pandas as pd
import torch

PRICE_COLS = ["price1", "price2", "price3"]
STATUS_MAP = {"Completed": 0, "Cancelled": 1, "Pending": 2}


def load_orders(path="data/orders_expanded.csv"):
    df = pd.read_csv(path)
    df["country"] = df["country"].fillna("Unknown")
    return df


def build_tensors(df):
    countries = sorted(df["country"].unique())
    country_to_id = {c: i for i, c in enumerate(countries)}

    # normalize prices (per-trajectory z-score) so scale doesn't dominate learning
    prices = df[PRICE_COLS].to_numpy(dtype=np.float32)
    mean = prices.mean(axis=1, keepdims=True)
    std = prices.std(axis=1, keepdims=True) + 1e-6
    prices_norm = (prices - mean) / std

    trajectories = torch.tensor(prices_norm[:, :, None], dtype=torch.float32)   # (N, 3, 1)
    country_ids = torch.tensor(df["country"].map(country_to_id).to_numpy(), dtype=torch.long)
    labels = torch.tensor(df["status"].map(STATUS_MAP).to_numpy(), dtype=torch.long)

    return trajectories, country_ids, labels, country_to_id


if __name__ == "__main__":
    df = load_orders()
    trajectories, country_ids, labels, country_to_id = build_tensors(df)
    print("trajectories:", trajectories.shape)
    print("country_ids:", country_ids.shape, "| vocab:", country_to_id)
    print("labels:", labels.shape, "| class counts:", torch.bincount(labels))
