import numpy as np
import pandas as pd

from autoresearch import eda


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=140, freq="D")
    day = np.arange(len(dates))
    rows = []
    # Smooth weekly-seasonal item.
    rows.append(
        pd.DataFrame(
            {
                "sku": "smooth",
                "date": dates,
                "demand": 50 + 20 * np.sin(2 * np.pi * day / 7),
                "promo": (day % 10 == 0).astype(int),
                "price": 10.0 - (day % 10 == 0) * 2.0,
            }
        )
    )
    # Intermittent item: mostly zeros.
    demand = np.zeros(len(dates))
    demand[::9] = 3
    rows.append(
        pd.DataFrame(
            {"sku": "sparse", "date": dates, "demand": demand, "promo": 0, "price": 5.0}
        )
    )
    return pd.concat(rows, ignore_index=True)


def test_profile_describes_columns() -> None:
    result = eda.profile(_frame())
    assert result["rows"] == 280
    by_name = {column["name"]: column for column in result["columns"]}
    assert by_name["demand"]["zero_share"] > 0.4
    assert by_name["sku"]["unique"] == 2


def test_seasonality_finds_weekly_cycle() -> None:
    result = eda.seasonality(_frame(), "date", "demand")
    assert result["daily_total_autocorrelation"]["lag_7"] > 0.8
    assert len(result["weekday_profile_vs_mean"]) == 7


def test_intermittency_classifies_patterns() -> None:
    result = eda.intermittency(_frame(), "sku", "date", "demand")
    assert result["items"] == 2
    counts = result["demand_pattern_counts"]
    assert counts["intermittent"] + counts["lumpy"] >= 1
    assert result["item_zero_share_quantiles"]["max"] > 0.8


def test_drivers_reports_promo_lift_and_price_correlation() -> None:
    result = eda.drivers(_frame(), "demand", exclude=["sku", "date"])
    assert "price" in result["correlation_with_target"]
    assert "promo" in result["binary_flag_lift_on_target"]
