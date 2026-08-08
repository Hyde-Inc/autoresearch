from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TaskConfig
from .guardrails import check_guardrails
from .metrics import compile_metric, load_task_spec


@dataclass
class Evaluation:
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    holdout_metrics: dict[str, float] = field(default_factory=dict)
    guardrail_failures: list[str] = field(default_factory=list)
    error: str | None = None
    duration_s: float = 0
    validation_frame: pd.DataFrame | None = None
    """Merged validation actuals and forecasts, kept for director error analysis."""


def forecasting_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    error = forecast - actual
    abs_error = np.abs(error)
    denominator = float(np.abs(actual).sum())
    wmape = float(abs_error.sum() / denominator) if denominator else float("inf")
    nonzero = np.abs(actual) > 1e-8
    mape = float(np.mean(abs_error[nonzero] / np.abs(actual[nonzero]))) if nonzero.any() else 0.0
    rmse = float(np.sqrt(np.mean(error**2)))
    bias_pct = float(error.sum() / denominator * 100) if denominator else float("inf")
    return {"wmape": wmape, "mape": mape, "rmse": rmse, "bias_pct": bias_pct}


def _validate_forecasts(
    forecast_path: Path, actuals: pd.DataFrame, config: TaskConfig
) -> tuple[dict[str, float], pd.DataFrame]:
    data = config.data
    if not forecast_path.exists():
        raise ValueError("solution did not create forecasts.parquet")
    forecasts = pd.read_parquet(forecast_path)
    required = [data.id_column, data.date_column, "forecast"]
    missing = set(required) - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecast output is missing columns: {sorted(missing)}")
    forecasts[data.date_column] = pd.to_datetime(forecasts[data.date_column])
    actuals = actuals.copy()
    actuals[data.date_column] = pd.to_datetime(actuals[data.date_column])
    keys = [data.id_column, data.date_column]
    if forecasts.duplicated(keys).any():
        raise ValueError("forecast output contains duplicate sku/date rows")
    expected = actuals[keys].sort_values(keys).reset_index(drop=True)
    received = forecasts[keys].sort_values(keys).reset_index(drop=True)
    if not expected.equals(received):
        raise ValueError("forecast output does not exactly cover the requested sku/date rows")
    merged = actuals.merge(forecasts[required], on=keys, validate="one_to_one")
    values = merged["forecast"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("forecast values must all be finite")
    if (values < 0).any():
        raise ValueError("forecast values must be non-negative")
    metrics = forecasting_metrics(merged[data.target_column].to_numpy(dtype=float), values)
    spec = load_task_spec(config)
    if spec is not None:
        frame = merged.rename(columns={data.target_column: "actual"})
        try:
            metrics[spec.name] = compile_metric(spec.code)(frame)
        except Exception as exc:
            raise ValueError(f"custom metric '{spec.name}' failed: {exc}") from exc
    return metrics, merged


async def _evaluate_split(
    worktree: Path,
    actuals_path: Path,
    config: TaskConfig,
    label: str,
) -> tuple[dict[str, float], float, pd.DataFrame]:
    actuals = pd.read_parquet(actuals_path)
    keys = [config.data.id_column, config.data.date_column]
    request = worktree / f".autoresearch-{label}-request.parquet"
    output = worktree / config.data.output
    actuals[keys].to_parquet(request, index=False)
    output.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "AUTORESEARCH_TRAIN_DATA": str(config.resolve(config.data.train)),
            "AUTORESEARCH_REQUEST": str(request),
            "AUTORESEARCH_OUTPUT": str(output),
        }
    )
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "--project",
        str(worktree),
        "python",
        "solution/train.py",
        cwd=worktree,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(), timeout=config.budget.train_timeout_s
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise ValueError(f"training exceeded {config.budget.train_timeout_s}s timeout")
    finally:
        request.unlink(missing_ok=True)
    elapsed = time.monotonic() - started
    if process.returncode:
        raise ValueError(
            f"solution exited with {process.returncode}: {stdout.decode(errors='replace')[-2000:]}"
        )
    metrics, merged = _validate_forecasts(output, actuals, config)
    return metrics, elapsed, merged


async def evaluate(
    worktree: Path,
    config: TaskConfig,
    baseline: dict[str, float] | None = None,
    guardrails: list[str] | None = None,
) -> Evaluation:
    started = time.monotonic()
    try:
        metrics, val_time, validation_frame = await _evaluate_split(
            worktree, config.resolve(config.data.validation_actuals), config, "validation"
        )
        holdout, holdout_time, _ = await _evaluate_split(
            worktree, config.resolve(config.data.holdout_actuals), config, "holdout"
        )
        metrics["runtime_s"] = val_time
        holdout["runtime_s"] = holdout_time
        expressions = guardrails if guardrails is not None else config.guardrails
        checks = check_guardrails(expressions, metrics, baseline or metrics)
        failures = [f"{item.expression}: {item.detail}" for item in checks if not item.passed]
        primary = config.metric.name
        holdout_key = f"holdout_{primary}"
        incumbent_holdout = baseline.get(holdout_key) if baseline else None
        holdout_worse = (
            incumbent_holdout is not None
            and (
                holdout[primary] > incumbent_holdout
                if config.metric.direction == "min"
                else holdout[primary] < incumbent_holdout
            )
        )
        if holdout_worse:
            failures.append(
                f"hidden holdout {primary} {holdout[primary]:.6f} is worse than incumbent "
                f"{incumbent_holdout:.6f}"
            )
        return Evaluation(
            passed=not failures,
            metrics=metrics,
            holdout_metrics=holdout,
            guardrail_failures=failures,
            duration_s=time.monotonic() - started,
            validation_frame=validation_frame,
        )
    except Exception as exc:  # noqa: BLE001 - evaluator failures become scored failures
        return Evaluation(False, error=str(exc), duration_s=time.monotonic() - started)
