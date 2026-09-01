from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .models import Idea

DEFAULT_MODEL = "openrouter/moonshotai/kimi-k3"


class MetricConfig(BaseModel):
    name: str = "wmape"
    direction: Literal["min", "max"] = "min"
    definition: Path | None = None


class ModelConfig(BaseModel):
    model: str = DEFAULT_MODEL
    temperature: float = 0.2


class AgentConfig(ModelConfig):
    count: int = Field(default=3, ge=1)
    timeout_s: int = Field(default=1200, ge=1)
    """Wall clock for one opencode session (one message of the worker loop)."""
    budget_s: int = Field(default=3600, ge=1)
    """Total wall clock for one experiment across all of its sessions."""


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


class FoundryDatasets(BaseModel):
    """Dataset RIDs the Foundry runtime reads and writes."""

    sales_raw: str = ""
    sales_train: str = ""
    forecast_request: str = ""
    forecasts: str = ""
    validation_actuals: str = ""
    holdout_actuals: str = ""
    evaluation_metrics: str = ""

    def pipeline_targets(self) -> list[str]:
        """Every buildable output of the pipeline (for a full rebuild).

        ``sales_train`` is a build target only when a raw feed is wired -
        without ``sales_raw`` there is no preprocessing transform producing it
        (it is an existing dataset), and asking Foundry to build it would hang
        on a missing job spec."""
        targets = [self.sales_train] if self.sales_raw and self.sales_train else []
        return targets + [rid for rid in (self.forecasts, self.evaluation_metrics) if rid]


class FoundryRuntimeConfig(BaseModel):
    """Everything needed to train on Foundry instead of a local subprocess.

    The transforms repo at ``repo_dir`` is pushed to Foundry; a build of the
    ``forecasts`` dataset runs the model on Foundry compute; the CLI reads the
    result and the sealed actuals back through readTable.
    """

    repo_dir: Path
    branch: str = "master"
    repo_rid: str = ""
    """Stemma repo RID, used to resolve the project folder during setup."""
    project_folder_rid: str = ""
    """Explicit project folder RID (overrides resolution from repo_rid)."""
    datasets: FoundryDatasets = FoundryDatasets()
    build_timeout_s: int = Field(default=900, ge=1)
    """Builds that outlive this are cancelled on Foundry, not just abandoned."""
    poll_s: int = Field(default=15, ge=1)


class OpeningRoundConfig(BaseModel):
    """Pin round 1 to a known-fast model family instead of letting the director route.

    Skill routing is an LLM call, so an opening round can land on a family that
    takes an agent a long time to write and a trainer a long time to run. Naming
    skills here replaces that call for round 1 only: later rounds route normally
    and can climb to heavier families once there is a result to beat.
    """

    skills: list[str] = Field(default_factory=list)
    instruction: str = (
        "This is the opening round, optimised for fast feedback rather than peak accuracy. "
        "Every idea must be implementable with the pinned skills above using only pandas, "
        "numpy and statsmodels - no gradient boosting, no neural or foundation models, and "
        "no hyperparameter search. Prefer changes that are quick to write and train in "
        "seconds, so the first scored result arrives early."
    )
    ideas: list[Idea] = Field(default_factory=list)
    """Pre-seeded round-1 experiments. When set, the director skips idea
    generation for the opening round and launches these verbatim - deterministic
    and instant, useful for demos or reproducing a known starting point."""
    think_seconds: int = Field(default=0, ge=0)
    """Cosmetic: when pre-seeded ``ideas`` are used, show the director 'thinking'
    spinner for this many seconds before revealing them (0 disables)."""


class TaskConfig(BaseModel):
    name: str = "autoresearch-task"
    description: str = ""
    goal: str = ""
    metric: MetricConfig = MetricConfig()
    secondary_metrics: list[str] = Field(default_factory=lambda: ["mape", "rmse", "bias_pct"])
    guardrails: list[str] = Field(default_factory=list)
    director: ModelConfig = ModelConfig()
    agents: AgentConfig = AgentConfig()
    budget: BudgetConfig = BudgetConfig()
    data: DataConfig = DataConfig()
    workspace: WorkspaceConfig = WorkspaceConfig()
    runtime: Literal["local", "foundry"] = "local"
    foundry: FoundryRuntimeConfig | None = None
    idea_hints: list[str] = Field(default_factory=list)
    context: str = ""
    skills: list[Path] = Field(default_factory=list)
    opening_round: OpeningRoundConfig = OpeningRoundConfig()
    config_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def normalize_models(self) -> TaskConfig:
        for cfg in (self.director, self.agents):
            if "/" not in cfg.model:
                raise ValueError(f"model must use provider/model format: {cfg.model}")
        if self.runtime == "foundry" and self.foundry is None:
            raise ValueError("runtime: foundry requires a 'foundry:' block")
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
    config = TaskConfig.model_validate(raw or {})
    config.config_path = path
    if config.opening_round.skills:
        from .skills import load_skills

        available = {skill.name for skill in load_skills(config)}
        unknown = [name for name in config.opening_round.skills if name not in available]
        if unknown:
            raise ValueError(
                f"opening_round.skills names no such skill: {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(available))}"
            )
    if config.metric.definition:
        from .metrics import load_spec

        spec = load_spec(config.resolve(config.metric.definition))
        config.metric.name = spec.name
        config.metric.direction = spec.direction
    return config
