from __future__ import annotations

import re
from dataclasses import dataclass

_COMPARISON = re.compile(
    r"^\s*(?P<metric>[a-zA-Z_][\w]*)\s*(?P<op><=|>=|<|>)\s*"
    r"(?:(?P<baseline>baseline)\s*\*\s*)?(?P<value>-?\d+(?:\.\d+)?)\s*$"
)
_WITHIN = re.compile(
    r"^\s*(?P<metric>[a-zA-Z_][\w]*)\s+within\s+"
    r"(?P<low>-?\d+(?:\.\d+)?)\s*\.\.\s*(?P<high>-?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class GuardrailResult:
    expression: str
    passed: bool
    detail: str


def evaluate_guardrail(
    expression: str,
    metrics: dict[str, float],
    baseline: dict[str, float],
) -> GuardrailResult:
    match = _WITHIN.match(expression)
    if match:
        metric = match["metric"]
        if metric not in metrics:
            return GuardrailResult(expression, False, f"metric {metric!r} is missing")
        low, high, actual = float(match["low"]), float(match["high"]), metrics[metric]
        passed = low <= actual <= high
        return GuardrailResult(expression, passed, f"{actual:.6g} within [{low:.6g}, {high:.6g}]")

    match = _COMPARISON.match(expression)
    if not match:
        return GuardrailResult(expression, False, "invalid expression")
    metric, op = match["metric"], match["op"]
    if metric not in metrics:
        return GuardrailResult(expression, False, f"metric {metric!r} is missing")
    limit = float(match["value"])
    if match["baseline"]:
        if metric not in baseline:
            return GuardrailResult(expression, False, f"baseline metric {metric!r} is missing")
        limit *= baseline[metric]
    actual = metrics[metric]
    passed = {"<": actual < limit, "<=": actual <= limit, ">": actual > limit, ">=": actual >= limit}[op]
    return GuardrailResult(expression, passed, f"{actual:.6g} {op} {limit:.6g}")


def check_guardrails(
    expressions: list[str],
    metrics: dict[str, float],
    baseline: dict[str, float],
) -> list[GuardrailResult]:
    return [evaluate_guardrail(expr, metrics, baseline) for expr in expressions]
