from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class MetricConfig(BaseModel):
    name: str = "wmape"
    direction: Literal["min", "max"] = "min"


class ModelConfig(BaseModel):
    model: str
    temperature: float = 0.2


class AgentConfig(ModelConfig):
    count: int = Field(default=3, ge=1)
    timeout_s: int = Field(default=900, ge=1)


class BudgetConfig(BaseModel):
    max_experiments: int = Field(default=12, ge=1)
    train_timeout_s: int = Field(default=600, ge=1)
    rounds: int = Field(default=4, ge=1)


class DataConfig(BaseModel):
    train: Path = Path("seed/data/train.parquet")
    validation_actuals: Path = Path("private/validation.parquet")
    holdout_actuals: Path = Path("private/holdout.parquet")
    output: Path = Path("forecasts.parquet")
    id_column: str = "sku"
    date_column: str = "date"
    target_column: str = "demand"


class WorkspaceConfig(BaseModel):
    seed: Path = Path("seed")
    runs: Path = Path("../../runs")
    allowed_paths: list[str] = Field(default_factory=lambda: ["solution/"])


class TaskConfig(BaseModel):
    name: str
    description: str
    goal: str
    metric: MetricConfig = MetricConfig()
    secondary_metrics: list[str] = Field(default_factory=lambda: ["mape", "rmse", "bias_pct"])
    guardrails: list[str] = Field(default_factory=list)
    director: ModelConfig
    agents: AgentConfig
    budget: BudgetConfig = BudgetConfig()
    data: DataConfig = DataConfig()
    workspace: WorkspaceConfig = WorkspaceConfig()
    idea_hints: list[str] = Field(default_factory=list)
    config_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def normalize_models(self) -> TaskConfig:
        for cfg in (self.director, self.agents):
            if "/" not in cfg.model:
                raise ValueError(f"model must use provider/model format: {cfg.model}")
        return self

    @property
    def root(self) -> Path:
        assert self.config_path is not None
        return self.config_path.parent

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.root / path).resolve()


def load_config(path: Path) -> TaskConfig:
    path = path.expanduser().resolve()
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    config = TaskConfig.model_validate(raw)
    config.config_path = path
    return config
