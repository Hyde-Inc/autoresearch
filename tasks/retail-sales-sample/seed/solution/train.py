"""
Per-Item Bias-Calibrated Ensemble with Price Elasticity.

Model A: Seasonal Naive + SES blend (weekly seasonality)
Model B: XGBoost with lag, rolling-mean, calendar, and price-elasticity features
Ensemble: bias-corrected A and B combined (0.4*A + 0.6*B),
          with per-item multiplicative bias calibration on last 14 days of train.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ID = "sku_name"
DATE = "date"
TARGET = "sales"
PRICE = "selling_price"


# ---------------------------------------------------------------------------
# Model A: Seasonal Naive + Simple Exponential Smoothing blend
# ---------------------------------------------------------------------------

def ses_forecast(series: pd.Series, alpha: float = 0.3, steps: int = 1) -> float:
    """Simple exponential smoothing level (last fitted value)."""
    if len(series) == 0:
        return np.nan
    level = float(series.iloc[0])
    for v in series.iloc[1:]:
        level = alpha * float(v) + (1 - alpha) * level
    return level


def model_a_forecast(series: pd.Series, target_date: pd.Timestamp,
                     global_mean: float) -> float:
    """Seasonal naive (7-day) blended 60/40 with SES level."""
    if series is None or len(series) == 0:
        return global_mean

    # Seasonal naive: walk back in 7-day steps
    lag_date = target_date
    seasonal_val = None
    for _ in range(60):
        lag_date -= pd.Timedelta(days=7)
        if lag_date in series.index:
            seasonal_val = float(series.loc[lag_date])
            break

    if seasonal_val is None:
        seasonal_val = float(series.iloc[-1])

    # SES on last 28 days
    recent = series.iloc[-28:]
    ses_level = ses_forecast(recent, alpha=0.3)
    if np.isnan(ses_level):
        ses_level = float(series.mean())

    return 0.6 * seasonal_val + 0.4 * ses_level


# ---------------------------------------------------------------------------
# Feature engineering for XGBoost
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag, rolling, calendar and price features for a single SKU dataframe."""
    df = df.sort_values(DATE).copy()
    df = df.reset_index(drop=True)

    s = df[TARGET].values.astype(float)
    n = len(s)

    lag_days = [1, 7, 14, 21, 28]
    for lag in lag_days:
        arr = np.empty(n)
        arr[:lag] = np.nan
        arr[lag:] = s[:n - lag]
        df[f"lag_{lag}"] = arr

    for window in [7, 14, 28]:
        roll = pd.Series(s).shift(1).rolling(window, min_periods=1).mean().values
        df[f"roll_mean_{window}"] = roll

    df["day_of_week"] = df[DATE].dt.dayofweek
    df["day_of_month"] = df[DATE].dt.day
    df["week_of_year"] = df[DATE].dt.isocalendar().week.astype(int)
    df["month"] = df[DATE].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Price elasticity features
    mean_price = df[PRICE].mean()
    if mean_price > 0:
        df["price_ratio"] = df[PRICE] / mean_price
        df["log_price_ratio"] = np.log(df["price_ratio"].clip(lower=1e-6))
    else:
        df["price_ratio"] = 1.0
        df["log_price_ratio"] = 0.0

    return df


FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28",
    "day_of_week", "day_of_month", "week_of_year", "month", "is_weekend",
    "price_ratio", "log_price_ratio",
    "sku_id",
]


def build_global_features(train: pd.DataFrame):
    """Build feature dataframe from the full training set (all SKUs)."""
    sku_to_id = {s: i for i, s in enumerate(train[ID].unique())}
    frames = []
    for sku, grp in train.groupby(ID, sort=False):
        feat = build_features(grp)
        feat["sku_id"] = sku_to_id[sku]
        frames.append(feat)
    full = pd.concat(frames, ignore_index=True)
    return full, sku_to_id


# ---------------------------------------------------------------------------
# XGBoost model B
# ---------------------------------------------------------------------------

def train_xgboost(train: pd.DataFrame):
    """Train a global XGBoost model on all SKU features using native API."""
    import xgboost as xgb

    full, sku_to_id = build_global_features(train)

    # Drop rows with NaN in lag features
    full = full.dropna(subset=["lag_1", "lag_7", "lag_14", "lag_28"])

    X = full[FEATURE_COLS].values.astype(float)
    y = full[TARGET].values.astype(float)

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "seed": 42,
        "nthread": -1,
        "verbosity": 0,
    }

    model = xgb.train(params, dtrain, num_boost_round=400, verbose_eval=False)
    return model, sku_to_id


def make_future_features_for_sku(
    hist_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    future_prices: pd.Series,
    sku_id: int,
    mean_price: float,
) -> pd.DataFrame:
    """
    Build feature rows for future dates of a single SKU using recursive strategy.
    hist_df must be sorted by date, with columns [date, sales, selling_price].
    """
    # Working copy of known sales
    known = hist_df.set_index(DATE)[TARGET].to_dict()

    rows = []
    for fd in sorted(future_dates):
        price = float(future_prices.get(fd, mean_price))
        price_ratio = price / mean_price if mean_price > 0 else 1.0
        log_price_ratio = float(np.log(max(price_ratio, 1e-6)))

        def get_lag(days):
            d = fd - pd.Timedelta(days=days)
            if d in known:
                return float(known[d])
            return np.nan

        lag_1 = get_lag(1)
        lag_7 = get_lag(7)
        lag_14 = get_lag(14)
        lag_21 = get_lag(21)
        lag_28 = get_lag(28)

        # Rolling means using known history (shift=1 means no leakage)
        def roll_mean(window):
            vals = []
            for d in range(1, window + 1):
                v = known.get(fd - pd.Timedelta(days=d))
                if v is not None:
                    vals.append(float(v))
            return float(np.mean(vals)) if vals else np.nan

        roll7 = roll_mean(7)
        roll14 = roll_mean(14)
        roll28 = roll_mean(28)

        rows.append({
            "date": fd,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_21": lag_21,
            "lag_28": lag_28,
            "roll_mean_7": roll7,
            "roll_mean_14": roll14,
            "roll_mean_28": roll28,
            "day_of_week": fd.dayofweek,
            "day_of_month": fd.day,
            "week_of_year": int(fd.isocalendar()[1]),
            "month": fd.month,
            "is_weekend": int(fd.dayofweek >= 5),
            "price_ratio": price_ratio,
            "log_price_ratio": log_price_ratio,
            "sku_id": sku_id,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-item bias calibration
# ---------------------------------------------------------------------------

def compute_bias_ratio(actuals: pd.Series, preds: pd.Series,
                       clip_lo: float = 0.5, clip_hi: float = 2.0) -> float:
    """Multiplicative bias correction ratio (actual / pred mean), clipped."""
    if len(actuals) == 0 or preds.sum() == 0:
        return 1.0
    ratio = actuals.mean() / (preds.mean() + 1e-9)
    return float(np.clip(ratio, clip_lo, clip_hi))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    train = pd.read_parquet(os.environ["AUTORESEARCH_TRAIN_DATA"])
    request = pd.read_parquet(os.environ["AUTORESEARCH_REQUEST"])
    output = Path(os.environ["AUTORESEARCH_OUTPUT"])

    train[DATE] = pd.to_datetime(train[DATE])
    request[DATE] = pd.to_datetime(request[DATE])
    train = train.sort_values([ID, DATE]).reset_index(drop=True)

    global_mean = float(train[TARGET].mean())

    # -----------------------------------------------------------------------
    # Calibration window: last 14 days of training for bias estimation
    # -----------------------------------------------------------------------
    max_train_date = train[DATE].max()
    calib_start = max_train_date - pd.Timedelta(days=13)
    calib_mask = train[DATE] >= calib_start
    train_fit = train[~calib_mask].copy()
    train_calib = train[calib_mask].copy()

    # -----------------------------------------------------------------------
    # Train XGBoost on train_fit (all data minus last 14 days)
    # -----------------------------------------------------------------------
    print("Training XGBoost model...")
    xgb_model, sku_to_id = train_xgboost(train_fit)

    # -----------------------------------------------------------------------
    # Build history dicts for all data (used for Model A and future feats)
    # -----------------------------------------------------------------------
    history = {
        key: grp.set_index(DATE).sort_index()
        for key, grp in train.groupby(ID, sort=False)
    }

    # -----------------------------------------------------------------------
    # Per-SKU: compute bias ratios from calibration window
    # -----------------------------------------------------------------------
    bias_a = {}
    bias_b = {}

    for sku, calib_grp in train_calib.groupby(ID, sort=False):
        hist_full = history[sku]
        sku_series = hist_full[TARGET]
        sku_hist_fit = train_fit[train_fit[ID] == sku]

        calib_grp = calib_grp.sort_values(DATE)
        calib_dates = calib_grp[DATE].values
        actuals = calib_grp[TARGET].values.astype(float)

        # Model A on calibration window
        preds_a = []
        for i, (cdate, actual) in enumerate(zip(calib_dates, actuals)):
            ts = pd.Timestamp(cdate)
            # Use only history strictly before this date
            s_before = sku_series[sku_series.index < ts]
            pred = model_a_forecast(s_before, ts, global_mean)
            preds_a.append(pred)
        preds_a = np.array(preds_a)

        # Model B on calibration window (using only train_fit features)
        sku_id = sku_to_id.get(sku, -1)
        if sku_id >= 0 and len(sku_hist_fit) >= 28:
            mean_price = float(sku_hist_fit[PRICE].mean())
            # Build price lookup for calibration window
            price_lookup = calib_grp.set_index(DATE)[PRICE]
            fut_feat = make_future_features_for_sku(
                hist_df=sku_hist_fit[[DATE, TARGET, PRICE]],
                future_dates=pd.DatetimeIndex([pd.Timestamp(d) for d in calib_dates]),
                future_prices=price_lookup,
                sku_id=sku_id,
                mean_price=mean_price,
            )
            import xgboost as xgb
            X_calib = fut_feat[FEATURE_COLS].values.astype(float)
            # Fill NaN with column means from training features
            col_means = np.nanmean(X_calib, axis=0)
            inds = np.where(np.isnan(X_calib))
            X_calib[inds] = np.take(col_means, inds[1])
            preds_b = np.maximum(0, xgb_model.predict(xgb.DMatrix(X_calib)))
        else:
            preds_b = preds_a.copy()

        a_ser = pd.Series(preds_a)
        b_ser = pd.Series(preds_b)
        act_ser = pd.Series(actuals)

        bias_a[sku] = compute_bias_ratio(act_ser, a_ser)
        bias_b[sku] = compute_bias_ratio(act_ser, b_ser)

    # -----------------------------------------------------------------------
    # Re-train XGBoost on ALL training data for final forecasting
    # -----------------------------------------------------------------------
    print("Re-training XGBoost on full training data...")
    xgb_model_full, sku_to_id_full = train_xgboost(train)

    # -----------------------------------------------------------------------
    # Generate forecasts for requested dates
    # -----------------------------------------------------------------------
    print("Generating forecasts...")
    forecasts = []

    for sku, req_grp in request.groupby(ID, sort=False):
        req_grp = req_grp.sort_values(DATE)
        future_dates = pd.DatetimeIndex(req_grp[DATE])

        hist_full = history.get(sku)
        sku_series = hist_full[TARGET] if hist_full is not None else None
        sku_train = train[train[ID] == sku]

        # Model A forecasts
        a_preds = []
        for fd in future_dates:
            pred = model_a_forecast(sku_series, fd, global_mean)
            a_preds.append(pred)
        a_preds = np.array(a_preds)

        # Model B forecasts
        sku_id = sku_to_id_full.get(sku, -1)
        if sku_id >= 0 and len(sku_train) >= 28:
            mean_price = float(sku_train[PRICE].mean())
            # Use last known price for future dates (no price data available)
            last_price = float(sku_train[PRICE].iloc[-1])
            price_lookup = pd.Series(last_price, index=future_dates)
            fut_feat = make_future_features_for_sku(
                hist_df=sku_train[[DATE, TARGET, PRICE]],
                future_dates=future_dates,
                future_prices=price_lookup,
                sku_id=sku_id,
                mean_price=mean_price,
            )
            import xgboost as xgb
            X_fut = fut_feat[FEATURE_COLS].values.astype(float)
            col_means = np.nanmean(X_fut, axis=0)
            inds = np.where(np.isnan(X_fut))
            X_fut[inds] = np.take(col_means, inds[1])
            b_preds = np.maximum(0, xgb_model_full.predict(xgb.DMatrix(X_fut)))
        else:
            b_preds = a_preds.copy()

        # Apply bias correction
        ba = bias_a.get(sku, 1.0)
        bb = bias_b.get(sku, 1.0)
        a_corrected = a_preds * ba
        b_corrected = b_preds * bb

        # Short history: fallback to Model A only
        if len(sku_train) < 14:
            ensemble = a_corrected
        else:
            ensemble = 0.4 * a_corrected + 0.6 * b_corrected

        ensemble = np.maximum(0.0, ensemble)

        for fd, fc in zip(future_dates, ensemble):
            forecasts.append({ID: sku, DATE: fd, "forecast": float(fc)})

    result = pd.DataFrame(forecasts)
    result = result[[ID, DATE, "forecast"]]

    # Final safety clip
    result["forecast"] = result["forecast"].clip(lower=0.0)
    result.to_parquet(output, index=False)
    print(f"Wrote {len(result)} forecasts to {output}")


if __name__ == "__main__":
    main()
