"""Pre-merge review gate: a structured, metric-agnostic confidence report.

Before a data scientist accepts an experiment, ``autoresearch review <id>``
breaks the primary metric down across entity groups, data cuts, and evaluation
sub-periods - comparing the candidate against the baseline on each - and rolls
everything into a single readiness score.

Everything here is derived from the data itself and the task config, so the
same gate works for any repo, context, and objective:

* the grouping/cut dimensions are *discovered* from whatever columns the
  customer's dataset carries (numeric covariates become tercile cuts, binary
  flags become on/off cuts, low-cardinality text becomes category groups);
* every comparison uses ``config.metric`` (name + direction), so a min-metric
  and a max-metric are scored the same way without special-casing;
* nothing is hard-coded to demand forecasting beyond the id/date/target trio
  the pipeline already relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TaskConfig
from .harness import forecasting_metrics
from .metrics import compile_metric, load_task_spec

_STANDARD_METRICS = {"wmape", "mape", "rmse", "bias_pct"}

# Score thresholds (0-100) shared by every dimension.
_PASS = 80.0
_WARN = 55.0

# How much a dimension counts toward the aggregate readiness score.
_WEIGHTS = {
    "overall": 0.40,
    "group": 0.20,
    "cut": 0.20,
    "temporal": 0.20,
}

# Keep the report tidy: at most this many discovered dimensions per step.
_MAX_GROUP_COLUMNS = 2
_MAX_CUT_COLUMNS = 3
_MAX_CATEGORY_VALUES = 5


@dataclass
class Segment:
    """One row of a breakdown: the metric for baseline vs candidate on a slice."""

    label: str
    size: int
    baseline: float | None
    candidate: float | None


@dataclass
class Block:
    """A single breakdown dimension (e.g. 'By volume' or 'oos_flag')."""

    label: str
    segments: list[Segment]


@dataclass
class ReviewStep:
    key: str
    title: str
    blocks: list[Block] = field(default_factory=list)

    def segments(self) -> list[Segment]:
        return [seg for block in self.blocks for seg in block.segments]


@dataclass
class Dimension:
    name: str
    score: float
    status: str


@dataclass
class ReviewReport:
    attempt_id: str
    metric: str
    direction: str
    overall_baseline: float | None
    overall_candidate: float | None
    steps: list[ReviewStep]
    dimensions: list[Dimension]
    aggregate_score: float
    verdict: str

    @property
    def overall_improvement(self) -> float | None:
        return improvement(self.overall_baseline, self.overall_candidate, self.direction)


# ---------------------------------------------------------------------------
# Metric + comparison primitives
# ---------------------------------------------------------------------------


def _metric_computer(config: TaskConfig):
    """Return a function that scores any sub-frame with the configured metric.

    Standard forecasting metrics are computed directly; a custom metric spec is
    compiled and run; anything else (e.g. runtime_s, which has no per-slice
    meaning) falls back to WMAPE so the breakdowns still say something useful.
    """
    name = config.metric.name
    target = config.data.target_column
    spec = load_task_spec(config)
    custom = compile_metric(spec.code) if spec is not None and name not in _STANDARD_METRICS else None

    def compute(frame: pd.DataFrame) -> float | None:
        if frame is None or frame.empty:
            return None
        actual = frame[target].to_numpy(dtype=float)
        forecast = frame["forecast"].to_numpy(dtype=float)
        if name in _STANDARD_METRICS:
            return forecasting_metrics(actual, forecast).get(name)
        if custom is not None:
            try:
                return float(custom(frame.rename(columns={target: "actual"})))
            except Exception:  # noqa: BLE001 - a bad slice must not crash the report
                return None
        return forecasting_metrics(actual, forecast).get("wmape")

    return compute


def improvement(
    baseline: float | None, candidate: float | None, direction: str
) -> float | None:
    """Signed relative improvement of candidate over baseline (positive = better)."""
    if baseline is None or candidate is None or baseline == 0:
        return None
    delta = (baseline - candidate) if direction == "min" else (candidate - baseline)
    return delta / abs(baseline)


def _is_worse(seg: Segment, direction: str) -> bool:
    if seg.baseline is None or seg.candidate is None:
        return False
    return seg.candidate > seg.baseline if direction == "min" else seg.candidate < seg.baseline


# ---------------------------------------------------------------------------
# Dimension discovery
# ---------------------------------------------------------------------------


def _discover(frame: pd.DataFrame, config: TaskConfig) -> tuple[list[str], list[tuple[str, str]]]:
    """Split the covariate columns into category groups and numeric/flag cuts.

    Returns ``(group_columns, cut_columns)`` where each cut is ``(kind, name)``
    with kind in {"numeric", "flag"}.
    """
    reserved = {
        config.data.id_column,
        config.data.date_column,
        config.data.target_column,
        "forecast",
    }
    groups: list[str] = []
    cuts: list[tuple[str, str]] = []
    for col in frame.columns:
        if col in reserved:
            continue
        series = frame[col]
        unique = int(series.nunique(dropna=True))
        if unique < 2:
            continue
        if pd.api.types.is_numeric_dtype(series):
            values = set(pd.unique(series.dropna()))
            if values <= {0, 1}:
                cuts.append(("flag", col))
            elif unique >= 6:
                cuts.append(("numeric", col))
            else:
                groups.append(col)
        elif unique <= 2000:
            groups.append(col)
    return groups[:_MAX_GROUP_COLUMNS], cuts[:_MAX_CUT_COLUMNS]


# ---------------------------------------------------------------------------
# Breakdown builders (each returns a Block)
# ---------------------------------------------------------------------------


def _segments_from_labels(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    base_labels: pd.Series,
    cand_labels: pd.Series,
    order: list,
    compute,
) -> list[Segment]:
    segments = []
    for value in order:
        cand_slice = candidate[cand_labels == value]
        base_slice = baseline[base_labels == value]
        if cand_slice.empty:
            continue
        segments.append(
            Segment(
                label=str(value),
                size=len(cand_slice),
                baseline=compute(base_slice),
                candidate=compute(cand_slice),
            )
        )
    return segments


def _volume_block(baseline, candidate, config, compute) -> Block | None:
    id_col = config.data.id_column
    target = config.data.target_column
    totals = candidate.groupby(id_col)[target].sum().sort_values(ascending=False)
    if len(totals) < 3:
        return None
    ids = totals.index.tolist()
    third = max(1, len(ids) // 3)
    tiers = {
        "High volume": set(ids[:third]),
        "Mid volume": set(ids[third : 2 * third]),
        "Low volume": set(ids[2 * third :]),
    }
    segments = []
    for label, members in tiers.items():
        cand_slice = candidate[candidate[id_col].isin(members)]
        base_slice = baseline[baseline[id_col].isin(members)]
        if cand_slice.empty:
            continue
        segments.append(
            Segment(label, len(cand_slice), compute(base_slice), compute(cand_slice))
        )
    return Block(f"By {id_col} volume", segments) if segments else None


def _category_block(baseline, candidate, column, compute) -> Block | None:
    top = candidate[column].value_counts().head(_MAX_CATEGORY_VALUES).index.tolist()
    if not top:
        return None
    cand_labels = candidate[column].where(candidate[column].isin(top), other="Other")
    base_labels = baseline[column].where(baseline[column].isin(top), other="Other")
    order = [*top]
    if not candidate[column].isin(top).all():
        order.append("Other")
    segments = _segments_from_labels(baseline, candidate, base_labels, cand_labels, order, compute)
    return Block(f"By {column}", segments) if segments else None


def _numeric_cut_block(baseline, candidate, column, compute) -> Block | None:
    values = candidate[column].to_numpy(dtype=float)
    edges = np.unique(np.nanquantile(values, [0.0, 1 / 3, 2 / 3, 1.0]))
    if len(edges) < 3:
        return None
    names = ["low", "mid", "high"][: len(edges) - 1]
    labels = [f"{column}: {name}" for name in names]
    cand_labels = pd.cut(candidate[column], bins=edges, labels=labels, include_lowest=True)
    base_labels = pd.cut(baseline[column], bins=edges, labels=labels, include_lowest=True)
    segments = _segments_from_labels(baseline, candidate, base_labels, cand_labels, labels, compute)
    return Block(f"By {column}", segments) if segments else None


def _flag_cut_block(baseline, candidate, column, compute) -> Block | None:
    cand_labels = candidate[column].map({0: f"{column}=0", 1: f"{column}=1"})
    base_labels = baseline[column].map({0: f"{column}=0", 1: f"{column}=1"})
    order = [f"{column}=0", f"{column}=1"]
    segments = _segments_from_labels(baseline, candidate, base_labels, cand_labels, order, compute)
    return Block(f"By {column}", segments) if segments else None


def _weekday_block(baseline, candidate, config, compute) -> Block | None:
    date_col = config.data.date_column
    cand_dt = pd.to_datetime(candidate[date_col])
    base_dt = pd.to_datetime(baseline[date_col])
    cand_labels = np.where(cand_dt.dt.dayofweek >= 5, "Weekend", "Weekday")
    base_labels = np.where(base_dt.dt.dayofweek >= 5, "Weekend", "Weekday")
    segments = _segments_from_labels(
        baseline,
        candidate,
        pd.Series(base_labels, index=baseline.index),
        pd.Series(cand_labels, index=candidate.index),
        ["Weekday", "Weekend"],
        compute,
    )
    return Block("Weekday vs weekend", segments) if len(segments) >= 2 else None


def _temporal_block(baseline, candidate, config, compute, periods: int = 3) -> Block | None:
    date_col = config.data.date_column
    cand_dt = pd.to_datetime(candidate[date_col])
    base_dt = pd.to_datetime(baseline[date_col])
    dates = sorted(cand_dt.dt.normalize().unique())
    if len(dates) < 2:
        return None
    periods = min(periods, len(dates))
    chunks = np.array_split(dates, periods)
    segments = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        lo = pd.Timestamp(chunk[0]).strftime("%m-%d")
        hi = pd.Timestamp(chunk[-1]).strftime("%m-%d")
        label = lo if lo == hi else f"{lo}..{hi}"
        members = set(pd.to_datetime(chunk))
        cand_slice = candidate[cand_dt.dt.normalize().isin(members)]
        base_slice = baseline[base_dt.dt.normalize().isin(members)]
        if cand_slice.empty:
            continue
        segments.append(
            Segment(label, len(cand_slice), compute(base_slice), compute(cand_slice))
        )
    return Block("By evaluation period", segments) if len(segments) >= 2 else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _status(score: float) -> str:
    if score >= _PASS:
        return "Pass"
    if score >= _WARN:
        return "Warn"
    return "Fail"


def _overall_score(rel: float | None) -> float:
    """50 = parity with baseline; ~+10% relative improvement saturates near 90."""
    if rel is None:
        return 50.0
    return float(max(0.0, min(100.0, 50.0 + rel * 400.0)))


def _consistency_score(segments: list[Segment], direction: str) -> float:
    """Size-weighted share of the data where the candidate is not worse."""
    total = 0
    good = 0
    for seg in segments:
        if seg.baseline is None or seg.candidate is None:
            continue
        total += seg.size
        if not _is_worse(seg, direction):
            good += seg.size
    if total == 0:
        return 50.0
    return 100.0 * good / total


def _verdict(aggregate: float, dimensions: list[Dimension]) -> str:
    fails = sum(1 for dim in dimensions if dim.status == "Fail")
    if fails >= 2 or aggregate < 45:
        return "Not recommended"
    if fails >= 1 or aggregate < _PASS:
        return "Review recommended"
    return "Ready to merge"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_review(
    config: TaskConfig,
    baseline_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    attempt_id: str,
) -> ReviewReport:
    compute = _metric_computer(config)
    direction = config.metric.direction
    groups, cuts = _discover(candidate_frame, config)

    # Step 1 - entity groups
    step1 = ReviewStep("group", "Step 1 - metric across entity groups")
    volume = _volume_block(baseline_frame, candidate_frame, config, compute)
    if volume:
        step1.blocks.append(volume)
    for column in groups:
        block = _category_block(baseline_frame, candidate_frame, column, compute)
        if block:
            step1.blocks.append(block)

    # Step 2 - data cuts
    step2 = ReviewStep("cut", "Step 2 - metric across data cuts")
    weekday = _weekday_block(baseline_frame, candidate_frame, config, compute)
    if weekday:
        step2.blocks.append(weekday)
    for kind, column in cuts:
        builder = _numeric_cut_block if kind == "numeric" else _flag_cut_block
        block = builder(baseline_frame, candidate_frame, column, compute)
        if block:
            step2.blocks.append(block)

    # Step 3 - evaluation periods
    step3 = ReviewStep("temporal", "Step 3 - metric across evaluation periods")
    period = _temporal_block(baseline_frame, candidate_frame, config, compute)
    if period:
        step3.blocks.append(period)

    steps = [step for step in (step1, step2, step3) if step.blocks]

    overall_b = compute(baseline_frame)
    overall_c = compute(candidate_frame)
    dimensions = [
        Dimension("Overall improvement", _overall_score(improvement(overall_b, overall_c, direction)), ""),
        Dimension("Group consistency", _consistency_score(step1.segments(), direction), ""),
        Dimension("Cut consistency", _consistency_score(step2.segments(), direction), ""),
        Dimension("Temporal stability", _consistency_score(step3.segments(), direction), ""),
    ]
    # Drop dimensions with no data (empty step) so they neither help nor hurt.
    present = {"group": bool(step1.blocks), "cut": bool(step2.blocks), "temporal": bool(step3.blocks)}
    kept: list[Dimension] = []
    weight_keys = {"Overall improvement": "overall", "Group consistency": "group",
                   "Cut consistency": "cut", "Temporal stability": "temporal"}
    for dim in dimensions:
        key = weight_keys[dim.name]
        if key != "overall" and not present[key]:
            continue
        dim.status = _status(dim.score)
        kept.append(dim)

    total_weight = sum(_WEIGHTS[weight_keys[dim.name]] for dim in kept) or 1.0
    aggregate = sum(_WEIGHTS[weight_keys[dim.name]] * dim.score for dim in kept) / total_weight

    return ReviewReport(
        attempt_id=attempt_id,
        metric=config.metric.name,
        direction=direction,
        overall_baseline=overall_b,
        overall_candidate=overall_c,
        steps=steps,
        dimensions=kept,
        aggregate_score=aggregate,
        verdict=_verdict(aggregate, kept),
    )
