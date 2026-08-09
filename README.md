# Parallel Autoresearch

This CLI runs ML experiments for you.

Point it at a data science repo. The Research Director explores the project itself - README, data
files, existing model code - agrees a goal with you in chat, builds a protected task, and then
parallel [OpenCode](https://github.com/anomalyco/opencode) agents implement and test experiment
ideas. The best result becomes the starting point for the next round.

It is inspired by [autoresearch](https://github.com/karpathy/autoresearch) and
[CORAL](https://github.com/langkhachhoha/CORAL).

## How it works

The flow has three phases: settle the metric, understand the repo, then research through
editable plan files.

1. In a streamed chat you settle what to optimize: the goal, the baseline, and the metric
   (standard or custom, verified with a hand-worked example).
2. An OpenCode agent surveys a throwaway copy of your repo in the background - models, data
   files and columns, evaluation conventions - while you chat; the chat opens instantly and
   the findings fold into the director's context the moment they are ready (cached at
   `.autoresearch/survey.md`). The director verifies anything load-bearing with its own
   read/EDA tools instead of asking you where things live.
3. It splits your history into train / validation / hidden holdout and seals the actuals away
   from the agents, then writes the baseline itself: if your repo already has a model, it ports
   it faithfully so the research has to beat your current approach; if there is no model, it
   picks a first model from the EDA and its demand forecasting skills.
4. Each research session gets its own numbered, goal-named folder under the visible `research/`
   directory, e.g. `research/001-reduce-wmape/`. The director writes its round-1 proposal there
   as `round-1-plan.md` and pops it open in your editor - like a coding agent's plan mode. You
   edit the file freely (reword hypotheses, delete or add experiments, change frontmatter), then
   type `execute`. The edited file is exactly what runs.
5. Each OpenCode agent gets its own git worktree and edits only `solution/`. A protected
   evaluator checks WMAPE, MAPE, RMSE, bias, runtime, and hidden holdout results, and the best
   valid experiment is promoted.
6. After the round, the findings open on your screen as `round-1-findings.md` (per-experiment
   scores plus the director's reflection), and the director writes `round-2-plan.md` (2-5
   parallel researches) after studying the errors of past attempts (worst SKUs, weekday bias,
   horizon decay) and the data itself. Each approval starts the next round. The session's
   `README.md` indexes every round with its status, so the folder reads like a lab notebook of
   all the research tried and how it scored.

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

`demo/retail-demand-forecasting` looks like a real data science repo: 14 months of daily sales
for 24 SKUs (including intermittent slow movers) - kept small so demo experiments train in
seconds - a README, a production seasonal-naive model in `models/seasonal_baseline.py`, and a
candidate ARIMA model in `models/arima.py`. Generate its data, then start:

```bash
uv run python demo/retail-demand-forecasting/scripts/make_dataset.py
uv run autoresearch start demo/retail-demand-forecasting
```

This opens a streaming chat with the Research Director, similar to a coding agent. You see its
thinking tokens live and type replies in a box. Two things must be clear before research starts -
the goal and the baseline. Set them with slash commands or just say them:

```
/goal reduce wmape
/baseline models/arima.py
go
```

The moment both are clear the director locks in: it builds the protected workspace, ports your
pinned model as the incumbent baseline, evaluates it, and writes the round-1 research plan to a
new session folder, e.g. `research/001-reduce-wmape/round-1-plan.md` - the file opens in your
editor automatically. Edit anything, and reply `execute` to launch the round - or give feedback
to have the plan rewritten, or `stop`.

Different teams score forecasts differently, so the metric is a first-class step too. Pass a
standard name or describe your own objective in plain English:

```
/metric penalize under-forecasting twice as much as over-forecasting
```

The director restates the metric, shows a hand-worked example with the exact grader code it will
use, and self-checks that the code reproduces the worked number. You confirm with `yes` (or ask
for changes), and from then on every experiment is scored and promoted on that metric. A custom
metric must be confirmed before the run can launch.

After each round a findings file (`round-N-findings.md`) lands next to the plan - and opens on
your screen - with per-experiment scores and the director's reflection, and the director writes
the next round's plan (2-5 parallel researches) after analyzing what failed and why. The gate is
the same every time: edit the plan file if you want, then `execute`, feedback, or `stop`. Files
auto-open via the `cursor`/`code` CLI or the OS default; set `AUTORESEARCH_NO_OPEN=1` to turn
that off.

To see the no-baseline path, delete `models/` from the demo repo and start again: the director
runs EDA (seasonality, intermittency, promo/price drivers) and bootstraps a first model from its
skills instead.

## No YAML, just slash commands

There is nothing to configure up front: defaults come from code, and you edit them inline in the
chat, Cursor/Claude style:

| command | what it does |
| --- | --- |
| `/goal <text>` | what the research must improve |
| `/baseline <path>` | pin an existing model script as the baseline |
| `/n_agents <1-5>` | parallel researches per round |
| `/metric <name\|description>` | wmape, mape, rmse, bias_pct - or describe a custom metric in plain English |
| `/guardrail <expr>` | add a guardrail, e.g. `bias_pct within -8..8` |
| `/rounds <n>` | maximum research rounds |
| `/timeout <seconds>` | per-experiment coding agent timeout |
| `/status`, `/help` | show settings / commands |

The agreed setup is saved internally to `<repo>/.autoresearch/task/task.yaml` so runs can be
resumed and inspected; you never write or edit it. Non-interactive commands work against it:

```bash
uv run autoresearch validate -c demo/retail-demand-forecasting/.autoresearch/task/task.yaml
uv run autoresearch run -c demo/retail-demand-forecasting/.autoresearch/task/task.yaml --parallel 3
uv run autoresearch resume -c demo/retail-demand-forecasting/.autoresearch/task/task.yaml
```

## Use a raw sales CSV

Without a repo, a CSV with `date`, `sku_name`, `sales`, and `selling_price` columns also works:

```bash
uv run autoresearch ingest sales.csv --name my-forecast
uv run autoresearch run -c tasks/my-forecast/task.yaml
uv run autoresearch metric -c tasks/my-forecast/task.yaml \
  "Penalize under-forecasting twice as much as over-forecasting"
```

## Research Director skills and tools

The outer loop includes our demand forecasting playbooks. It chooses the relevant skills each
round based on the data, past results, holdout gaps, bias, and guardrail failures, and cites
them per experiment in the plan file (`skills:` line - your edits there are honored too).
Skills cover model selection, leakage-safe tree features, an XGBoost demand playbook (Tweedie
objectives for zero-heavy demand, direct vs recursive multi-step, stockout bias traps), a
Chronos playbook (the zero-shot-first ladder, context/horizon limits, covariate regressors,
fine-tuning recipes), intermittent demand, price and promotions, bias correction, and
ensembling.

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
Add them in chat:

```
/guardrail rmse<=baseline*1.10
/guardrail bias_pct within -8..8
/guardrail runtime_s<=600
```

Validation and holdout actuals are never copied into the seed workspace or agent worktrees; the
raw project data stays outside too, so agents only ever see the train split. Agents can only
change files under `solution/` by default. This is suitable for a trusted local demo. Use a
container or VM for untrusted models.
