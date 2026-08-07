"""User-defined evaluation metrics: interpret, verify, persist, and load.

The flow: a user describes a metric in plain language, an LLM turns it into a
``MetricSpec`` (precise restatement + hand-worked example + grader code), and the
spec is only accepted if the generated code, executed on the LLM's own worked
example, reproduces the hand-calculated value. The worked example doubles as a
unit test for the grader.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from openai import OpenAI
from pydantic import BaseModel, Field

METRIC_FUNCTION = "metric"
ALLOWED_IMPORTS = {"numpy", "pandas", "math"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
}
DEFINITION_FILENAME = "custom_metric.json"


class WorkedExample(BaseModel):
    rows: list[dict] = Field(min_length=2)
    steps: list[str]
    value: float


class MetricSpec(BaseModel):
    name: str
    direction: Literal["min", "max"]
    description: str
    understanding: str
    example: WorkedExample
    code: str


class MetricCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class MetricValidation(BaseModel):
    passed: bool
    checks: list[MetricCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def failure_summary(self) -> str:
        return "; ".join(f"{c.name}: {c.detail}" for c in self.checks if not c.passed)


def _assert_safe(code: str) -> None:
    """Reject code that reaches outside numpy/pandas/math arithmetic."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ValueError(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"dunder attribute access '{node.attr}' is not allowed")


def compile_metric(code: str) -> Callable[[pd.DataFrame], float]:
    _assert_safe(code)
    namespace: dict = {"np": np, "numpy": np, "pd": pd, "pandas": pd, "math": math}
    exec(compile(code, "<custom_metric>", "exec"), namespace)  # noqa: S102 - AST-sanitized above
    fn = namespace.get(METRIC_FUNCTION)
    if not callable(fn):
        raise ValueError(f"code must define a function named '{METRIC_FUNCTION}'")

    def wrapped(df: pd.DataFrame) -> float:
        return float(fn(df.copy()))

    return wrapped


def _example_frame(spec: MetricSpec) -> pd.DataFrame:
    return pd.DataFrame(spec.example.rows)


def validate_spec(spec: MetricSpec, columns: list[str] | None = None) -> MetricValidation:
    """Machine-check a spec before a human ever confirms it."""
    checks: list[MetricCheck] = []
    warnings: list[str] = []

    try:
        fn = compile_metric(spec.code)
        checks.append(MetricCheck(name="compiles", passed=True, detail="code compiled safely"))
    except Exception as exc:  # noqa: BLE001
        checks.append(MetricCheck(name="compiles", passed=False, detail=str(exc)))
        return MetricValidation(passed=False, checks=checks, warnings=warnings)

    frame = _example_frame(spec)
    missing = {"actual", "forecast"} - set(frame.columns)
    if missing:
        checks.append(
            MetricCheck(
                name="example_columns",
                passed=False,
                detail=f"example rows are missing columns: {sorted(missing)}",
            )
        )
        return MetricValidation(passed=False, checks=checks, warnings=warnings)
    if columns:
        extra = set(frame.columns) - set(columns)
        if extra:
            warnings.append(f"example uses columns not present in eval data: {sorted(extra)}")

    # The decisive check: run the generated grader on the LLM's own hand-worked
    # example and require it to reproduce the hand-calculated value.
    try:
        computed = fn(frame)
    except Exception as exc:  # noqa: BLE001
        checks.append(
            MetricCheck(name="worked_example", passed=False, detail=f"metric raised: {exc}")
        )
        return MetricValidation(passed=False, checks=checks, warnings=warnings)
    expected = spec.example.value
    matches = math.isfinite(computed) and math.isclose(
        computed, expected, rel_tol=5e-3, abs_tol=1e-9
    )
    checks.append(
        MetricCheck(
            name="worked_example",
            passed=matches,
            detail=(
                f"code computed {computed:.6f}, hand calculation said {expected:.6f}"
                + ("" if matches else " (mismatch)")
            ),
        )
    )

    # Robustness: the metric must stay finite on larger, noisier data.
    try:
        rng = np.random.default_rng(7)
        big = pd.concat([frame] * 10, ignore_index=True)
        noise = rng.uniform(0.5, 1.5, len(big))
        big["forecast"] = big["forecast"].astype(float) * noise
        stressed = fn(big)
        finite = math.isfinite(stressed)
        checks.append(
            MetricCheck(
                name="finite_on_noise",
                passed=finite,
                detail=f"returned {stressed:.6f} on noisy 10x data"
                if finite
                else f"returned non-finite value {stressed} on noisy data",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            MetricCheck(name="finite_on_noise", passed=False, detail=f"metric raised: {exc}")
        )

    # Direction sanity: a perfect forecast should not score worse than the
    # example's imperfect one. Advisory only - some metrics (e.g. bias) are not
    # monotone in forecast error.
    try:
        perfect = frame.copy()
        perfect["forecast"] = perfect["actual"]
        v_perfect = fn(perfect)
        consistent = v_perfect <= computed if spec.direction == "min" else v_perfect >= computed
        if not consistent:
            warnings.append(
                f"direction check: perfect forecast scored {v_perfect:.6f} which is worse than "
                f"the example's {computed:.6f} for direction '{spec.direction}'"
            )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"direction check could not run: {exc}")

    return MetricValidation(
        passed=all(c.passed for c in checks), checks=checks, warnings=warnings
    )


def save_spec(spec: MetricSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.model_dump_json(indent=2) + "\n")


def load_spec(path: Path) -> MetricSpec:
    return MetricSpec.model_validate_json(path.read_text())


def load_task_spec(config) -> MetricSpec | None:
    """Load the custom metric spec referenced by a TaskConfig, if any."""
    definition = getattr(config.metric, "definition", None)
    if not definition:
        return None
    return load_spec(config.resolve(definition))


def adopt_spec(config, spec: MetricSpec) -> Path:
    """Persist a confirmed spec and point the task config's metric block at it."""
    definition = Path("eval") / DEFINITION_FILENAME
    target = config.resolve(definition)
    save_spec(spec, target)
    raw = yaml.safe_load(config.config_path.read_text())
    raw["metric"] = {
        "name": spec.name,
        "direction": spec.direction,
        "definition": str(definition),
    }
    config.config_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    return target


def eval_columns(config) -> list[str]:
    """Columns the grader will hand to the metric function for this task."""
    actuals = pd.read_parquet(config.resolve(config.data.validation_actuals))
    columns = [c for c in actuals.columns if c != config.data.target_column]
    return ["actual", "forecast", *columns]


_SYSTEM_PROMPT = """\
You are the metric engineer for an autonomous ML forecasting lab. A user describes an
evaluation metric in plain language. You must produce three things:
1. A precise restatement of the metric so the user can confirm you understood it.
2. A tiny hand-worked example proving the calculation step by step.
3. The Python grader function that will score every experiment.

Grader function contract:
- Signature exactly: def metric(df: pd.DataFrame) -> float
- df has one row per (item, date) pair being scored. Available columns: {columns}.
  'actual' is the true value, 'forecast' is the model's prediction. Use only these columns.
- Only numpy (as np), pandas (as pd), and math may be used. No other imports, no file or
  network access, no exec/eval.
- Must return one finite float. Guard every division that could hit zero.

Return strict JSON with exactly these keys:
{{
  "name": snake_case identifier for the metric (e.g. "weighted_mape"),
  "direction": "min" if lower is better, "max" if higher is better,
  "understanding": 2-4 sentences restating the metric precisely, including the exact formula.
    If the user's description is ambiguous, pick the most standard interpretation and state
    the assumption explicitly.
  "example": {{
    "rows": 4-6 objects, each with EVERY available column listed above (use small round
      numbers; forecasts must differ from actuals),
    "steps": array of short strings walking through the arithmetic by hand,
    "value": the exact numeric result for those rows - compute it carefully to at least
      6 significant digits, it will be checked against your code
  }},
  "code": Python source string defining metric(df)
}}
"""


class MetricInterpreter:
    """Turns a plain-language metric description into a verified MetricSpec."""

    def __init__(self, model: str, temperature: float = 0.2):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/Hyde-Inc/autoresearch",
                "X-Title": "Parallel Autoresearch",
            },
        )
        self.model = model.removeprefix("openrouter/")
        self.temperature = temperature

    def _complete(self, messages: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("metric interpreter returned an empty response")
        match = re.match(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", content, re.DOTALL)
        if match:
            content = match["body"]
        return json.loads(content)

    async def interpret(
        self, description: str, columns: list[str], max_attempts: int = 3
    ) -> tuple[MetricSpec, MetricValidation]:
        """Interpret a description, retrying with validation feedback on failure."""
        system = _SYSTEM_PROMPT.format(columns=", ".join(columns))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Metric description from the user:\n{description}"},
        ]
        last_error = "unknown error"
        for _ in range(max_attempts):
            payload = await asyncio.to_thread(self._complete, messages)
            try:
                spec = MetricSpec.model_validate({**payload, "description": description})
            except Exception as exc:  # noqa: BLE001
                last_error = f"response did not match the required JSON shape: {exc}"
                messages.append({"role": "assistant", "content": json.dumps(payload)})
                messages.append(
                    {"role": "user", "content": f"That failed validation: {last_error}. Fix it."}
                )
                continue
            validation = validate_spec(spec, columns)
            if validation.passed:
                return spec, validation
            last_error = validation.failure_summary()
            messages.append({"role": "assistant", "content": json.dumps(payload)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your grader code was executed against your own worked example and "
                        f"failed these checks: {last_error}. Reconcile the code, the hand "
                        "calculation, and the value, then return the corrected full JSON."
                    ),
                }
            )
        raise RuntimeError(f"could not produce a verified metric after {max_attempts} attempts: {last_error}")
