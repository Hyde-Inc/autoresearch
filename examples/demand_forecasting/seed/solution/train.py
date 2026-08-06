"""Seasonal-naive baseline: the same weekday from the most recent week."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    train = pd.read_parquet(os.environ["AUTORESEARCH_TRAIN_DATA"])
    request = pd.read_parquet(os.environ["AUTORESEARCH_REQUEST"])
    output = Path(os.environ["AUTORESEARCH_OUTPUT"])
    train["date"] = pd.to_datetime(train["date"])
    request["date"] = pd.to_datetime(request["date"])

    history = {
        sku: group.set_index("date")["demand"].sort_index()
        for sku, group in train.groupby("sku", sort=False)
    }
    forecasts = []
    for row in request.itertuples(index=False):
        series = history[row.sku]
        lag_date = row.date
        while lag_date not in series.index:
            lag_date -= pd.Timedelta(days=7)
        value = float(series.loc[lag_date])
        forecasts.append(max(0.0, value))
    result = request[["sku", "date"]].copy()
    result["forecast"] = np.asarray(forecasts)
    result.to_parquet(output, index=False)


if __name__ == "__main__":
    main()
