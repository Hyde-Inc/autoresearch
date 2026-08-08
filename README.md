# Parallel Autoresearch

This CLI runs ML experiments for you.

Point it at a data science repo. The Research Director explores the project itself - README, data
files, existing model code - agrees a goal with you in chat, builds a protected task, and then
parallel [OpenCode](https://github.com/anomalyco/opencode) agents implement and test experiment
ideas. The best result becomes the starting point for the next round.

It is inspired by [autoresearch](https://github.com/karpathy/autoresearch) and
[CORAL](https://github.com/langkhachhoha/CORAL).

## How it works

1. The Research Director explores your repo with tools (read files, profile data, run EDA) instead
   of asking you where things live.
2. It splits your history into train / validation / hidden holdout and seals the actuals away from
   the agents.
3. It writes the baseline itself: if your repo already has a model, it ports it faithfully so the
   research has to beat your current approach; if there is no model, it picks a first model from
   the EDA and its demand forecasting skills.
4. Each OpenCode agent gets its own git worktree and edits only `solution/`.
5. A protected evaluator checks WMAPE, MAPE, RMSE, bias, runtime, and hidden holdout results.
6. The best valid experiment is promoted. Between rounds the director studies the errors of past
   attempts (worst SKUs, weekday bias, horizon decay) and the data itself before proposing the
   next ideas.

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

## Run it on the demo project

`demo/retail-demand-forecasting` looks like a real data science repo: two years of daily sales for
60 SKUs (including intermittent slow movers), a README, and a production seasonal-naive model in
`models/seasonal_baseline.py`. Generate its data, then start:

```bash
uv run python demo/retail-demand-forecasting/scripts/make_dataset.py
uv run autoresearch start demo/retail-demand-forecasting
```

This opens a streaming chat with the Research Director, similar to a coding agent. You see its
thinking tokens live and type replies in a box; there are no yes/no menus. It reads the project,
profiles the data, settles the goal and guardrails with you, builds the protected workspace,
writes and evaluates the baseline, and starts the agents once you approve the round-1 plan. Bad
input (like a wrong file path) comes back as a normal chat reply, not a crash.

To see the no-baseline path, delete `models/` from the demo repo and start again: the director
runs EDA (seasonality, intermittency, promo/price drivers) and bootstraps a first model from its
skills instead.

## Configuration is optional

You do not write a task config. Everything is discovered in the setup chat and saved to
`<repo>/.autoresearch/task/task.yaml`. To pin run hyperparameters, add an optional
`autoresearch.yaml` to your repo root; every field has a default:

```yaml
director: { model: openrouter/moonshotai/kimi-k3, temperature: 0.35 }
agents: { model: openrouter/moonshotai/kimi-k3, count: 3, timeout_s: 900 }
budget: { rounds: 4, max_experiments: 12, train_timeout_s: 600 }
guardrails: ["rmse<=baseline*1.10", "runtime_s<=600"]
skills: [docs/forecasting-notes.md]
```

Non-interactive commands work against the generated config:

```bash
uv run autoresearch validate -c demo/retail-demand-forecasting/.autoresearch/task/task.yaml
uv run autoresearch run -c demo/retail-demand-forecasting/.autoresearch/task/task.yaml --parallel 3
uv run autoresearch resume -c demo/retail-demand-forecasting/.autoresearch/task/task.yaml
```

## Use a raw sales CSV

Without a repo, a CSV with `date`, `sku_name`, `sales`, and `selling_price` columns also works:

```bash
uv run autoresearch start --csv sales.csv --name my-forecast
uv run autoresearch metric -c tasks/my-forecast/task.yaml \
  "Penalize under-forecasting twice as much as over-forecasting"
```

## Research Director skills and tools

The outer loop includes our demand forecasting playbooks. It chooses the relevant skills each
round based on the data, past results, holdout gaps, bias, and guardrail failures. Skills cover
model selection, leakage-safe tree features, intermittent demand, price and promotions, bias
correction, ensembling, and Chronos.

```bash
uv run autoresearch skills
```

Between rounds the director also has analysis tools: EDA over the training data (profile,
seasonality, intermittency, promo/price drivers) and error analysis over every scored attempt's
validation forecasts (`analyze_errors`, `worst_items`). What it consulted is printed with each
round.

## View results

```bash
uv run autoresearch status
uv run autoresearch leaderboard
uv run autoresearch show ATTEMPT_ID --run-dir <repo>/.autoresearch/runs/TASK/TIMESTAMP
uv run autoresearch report --run-dir <repo>/.autoresearch/runs/TASK/TIMESTAMP -o report.md
uv run autoresearch stop
```

Results, patches, agent logs, metrics, per-attempt validation forecasts, and research notes are
saved under `<repo>/.autoresearch/runs/`.

## Guardrails

Guardrails prevent an agent from improving one number while making the model worse somewhere else.

```yaml
guardrails:
  - "rmse<=baseline*1.10"
  - "bias_pct within -8..8"
  - "runtime_s<=600"
```

Validation and holdout actuals are never copied into the seed workspace or agent worktrees; the
raw project data stays outside too, so agents only ever see the train split. Agents can only
change files under `solution/` by default. This is suitable for a trusted local demo. Use a
container or VM for untrusted models.
