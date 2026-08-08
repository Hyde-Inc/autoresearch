# Parallel Autoresearch

This CLI runs ML experiments for you.

You give it a goal like reducing WMAPE. An LLM proposes ideas, then parallel
[OpenCode](https://github.com/anomalyco/opencode) agents implement and test them. The best result
becomes the starting point for the next round.

It is inspired by [autoresearch](https://github.com/karpathy/autoresearch) and
[CORAL](https://github.com/langkhachhoha/CORAL).

## How it works

1. A research director uses demand forecasting skills to propose experiments.
2. Each OpenCode agent gets its own git worktree.
3. Agents edit the model code and create forecasts.
4. A protected evaluator checks WMAPE, MAPE, RMSE, bias, runtime, and hidden holdout results.
5. The best valid experiment is promoted.
6. The director reviews the results and proposes the next ideas.

## Setup

You need Python 3.12, `uv`, `git`, OpenCode, and an OpenRouter API key.

```bash
brew install anomalyco/tap/opencode
uv sync
cp .env.example .env
```

Add your key to `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key
```

Prepare the small demand forecasting demo:

```bash
uv run python examples/demand_forecasting/prepare.py
```

The demo has 60 synthetic retail SKUs with daily demand, promotions, prices, trends, and
seasonality.

## Run it

The easiest option is the interactive setup:

```bash
uv run autoresearch start -c examples/demand_forecasting/task.yaml
```

This opens a streaming chat with the Research Director, similar to a coding agent. You see its
thinking tokens live, type replies in a box, and there are no yes/no menus. It settles the goal,
metric, and guardrails with you, uses the seeded baseline unless you give it your own script,
runs the protected baseline evaluation, and starts the agents once you approve the round-1 plan.
Bad input (like a wrong file path) comes back as a normal chat reply, not a crash.

Check the baseline:

```bash
uv run autoresearch validate -c examples/demand_forecasting/task.yaml
```

Start parallel research:

```bash
uv run autoresearch run -c examples/demand_forecasting/task.yaml \
  --goal "Reduce WMAPE on the 28-day validation window" \
  --guardrail "rmse<=baseline*1.10" \
  --guardrail "runtime_s<=600" \
  --parallel 3 \
  --max-experiments 12
```

The demo can try ideas based on ARIMA, XGBoost, Chronos, calibration, and ensembles.

## Use your own sales data

The CSV needs `date`, `sku_name`, `sales`, and `selling_price` columns.

```bash
uv run autoresearch start --csv sales.csv --name my-forecast
```

You can also run each setup step separately:

```bash
uv run autoresearch ingest sales.csv --name my-forecast
uv run autoresearch metric -c tasks/my-forecast/task.yaml \
  "Penalize under-forecasting twice as much as over-forecasting"
```

## Research Director skills

The outer loop includes our demand forecasting playbooks. It chooses the relevant skills each
round based on the data, past results, holdout gaps, bias, and guardrail failures. Skills cover
model selection, leakage-safe tree features, intermittent demand, price and promotions, bias
correction, ensembling, and Chronos.

```bash
uv run autoresearch skills
```

Add client-specific markdown skills with a `skills:` path list in `task.yaml`.

## View results

```bash
uv run autoresearch status
uv run autoresearch leaderboard
uv run autoresearch show ATTEMPT_ID --run-dir runs/demand-forecasting-demo/TIMESTAMP
uv run autoresearch report --run-dir runs/demand-forecasting-demo/TIMESTAMP -o report.md
```

Stop or resume a run:

```bash
uv run autoresearch stop
uv run autoresearch resume -c examples/demand_forecasting/task.yaml
```

Results, patches, agent logs, metrics, and research notes are saved under `runs/`.

## Guardrails

Guardrails prevent an agent from improving one number while making the model worse somewhere else.

```yaml
guardrails:
  - "rmse<=baseline*1.10"
  - "bias_pct within -8..8"
  - "runtime_s<=600"
```

Validation and holdout actuals are not copied into agent worktrees. Agents can only change files
under `solution/` by default. This is suitable for a trusted local demo. Use a container or VM for
untrusted models.
