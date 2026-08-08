from __future__ import annotations

import json
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from .config import TaskConfig


class ResearchBrief(BaseModel):
    goal: str
    context: str
    guardrails: list[str] = Field(default_factory=list)
    idea_hints: list[str] = Field(default_factory=list)
    metric_name: str = "wmape"
    metric_description: str = ""


class InterviewTurn(BaseModel):
    message: str
    done: bool = False
    brief: ResearchBrief | None = None


def data_profile(config: TaskConfig) -> dict:
    frame = pd.read_parquet(config.resolve(config.data.train))
    target = pd.to_numeric(frame[config.data.target_column], errors="coerce")
    dates = pd.to_datetime(frame[config.data.date_column])
    return {
        "rows": len(frame),
        "items": int(frame[config.data.id_column].nunique()),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "distinct_dates": int(dates.nunique()),
        "zero_share": round(float((target.fillna(0) == 0).mean()), 4),
        "columns": list(frame.columns),
    }


class ResearchInterview:
    def __init__(
        self,
        config: TaskConfig,
        complete: Callable[[str, str], Awaitable[dict]],
    ):
        self.config = config
        self.complete = complete
        self.profile = data_profile(config)

    async def next_turn(
        self, conversation: list[dict[str, str]], force_finish: bool = False
    ) -> InterviewTurn:
        system = (
            "You are a demand forecasting Research Director helping a team define an autonomous "
            "research assignment. Ask one concise question at a time. Resolve the objective, "
            "metric, asymmetric business costs, constraints, guardrails, and useful model families. "
            "Do not ask for facts already supplied. Once the assignment is actionable, set done "
            "to true and provide a complete brief. Return strict JSON with keys message, done, "
            "brief. brief is null until done; then it has exactly goal, context, guardrails, "
            "idea_hints, metric_name, metric_description. Leave metric_description empty for "
            "standard WMAPE, MAPE, RMSE, or bias. Use it only when custom grader logic is needed. "
            "Guardrails use expressions such as "
            "'rmse<=baseline*1.10', 'bias_pct within -8..8', or 'runtime_s<=600'."
        )
        finish = (
            "\nThe user requested completion. Infer reasonable defaults and return the final brief."
            if force_finish
            else ""
        )
        user = (
            f"Current task:\n{self.config.model_dump_json(exclude={'config_path'})}\n"
            f"Training data profile:\n{json.dumps(self.profile)}\n"
            f"Conversation:\n{json.dumps(conversation)}{finish}"
        )
        payload = await self.complete(system, user)
        turn = InterviewTurn.model_validate(payload)
        if turn.done and turn.brief is None:
            raise ValueError("interview marked done without a research brief")
        return turn


def apply_brief(config: TaskConfig, brief: ResearchBrief) -> None:
    assert config.config_path is not None
    raw = yaml.safe_load(config.config_path.read_text())
    raw["goal"] = brief.goal
    raw["context"] = brief.context
    raw["guardrails"] = brief.guardrails
    raw["idea_hints"] = brief.idea_hints
    if brief.metric_name and not brief.metric_description:
        raw["metric"] = {
            "name": brief.metric_name,
            "direction": "min",
        }
    config.config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    )


def replace_baseline(config: TaskConfig, source: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"baseline script not found: {source}")
    destination = config.resolve(config.workspace.seed) / "solution" / "train.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def append_context(config: TaskConfig, text: str) -> None:
    assert config.config_path is not None
    raw = yaml.safe_load(config.config_path.read_text())
    existing = str(raw.get("context", "")).strip()
    raw["context"] = "\n".join(part for part in (existing, text.strip()) if part)
    config.config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    )
