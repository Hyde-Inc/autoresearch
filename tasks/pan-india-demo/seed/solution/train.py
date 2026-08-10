"""Seasonal-naive baseline: repeat the same weekday from the most recent week."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ID = "sku_name"
DATE = "date"
TARGET = "sales"


def main() -> None:
    train = pd.read_parquet(os.environ["AUTORESEARCH_TRAIN_DATA"])
    request = pd.read_parquet(os.environ["AUTORESEARCH_REQUEST"])
    output = Path(os.environ["AUTORESEARCH_OUTPUT"])
    train[DATE] = pd.to_datetime(train[DATE])
    request[DATE] = pd.to_datetime(request[DATE])

    history = {
        key: group.set_index(DATE)[TARGET].sort_index()
        for key, group in train.groupby(ID, sort=False)
    }
    global_mean = float(train[TARGET].mean())
    means = train.groupby(ID)[TARGET].mean().to_dict()

    forecasts = []
    for row in request.itertuples(index=False):
        key = getattr(row, ID)
        series = history.get(key)
        value = means.get(key, global_mean)
        if series is not None and len(series):
            lag_date = getattr(row, DATE)
            steps = 0
            while lag_date not in series.index and steps < 60:
                lag_date -= pd.Timedelta(days=7)
                steps += 1
            value = float(series.loc[lag_date]) if lag_date in series.index else float(series.iloc[-1])
        forecasts.append(max(0.0, float(value)))

    result = request[[ID, DATE]].copy()
    result["forecast"] = np.asarray(forecasts, dtype=float)
    result.to_parquet(output, index=False)


if __name__ == "__main__":
    main()
