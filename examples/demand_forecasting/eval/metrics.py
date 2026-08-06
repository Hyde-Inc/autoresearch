"""Reference metric definitions used by the protected evaluator."""

import numpy as np


def wmape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.abs(forecast - actual).sum() / np.abs(actual).sum())


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = np.abs(actual) > 1e-8
    return float(np.mean(np.abs(forecast[mask] - actual[mask]) / np.abs(actual[mask])))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((forecast - actual) ** 2)))


def bias_pct(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float((forecast - actual).sum() / np.abs(actual).sum() * 100)
