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
5. Each OpenCode agent gets its own git worktree and edits only `solution/`. The orchestrator
  drives it the way you drive a coding assistant, as an explicit build -> train -> evaluate
   loop with each step verified and tracked on the live dashboard: the agent *builds*
   `solution/train.py` (checked cheaply - exists and parses - before anything runs), then the
   orchestrator itself *trains* it on the validation request while streaming `train.py` output
   as live status - no coding agent is running or billed while training executes - then the
   harness *evaluates* the forecasts and sends the result (traceback, guardrail failure, or
   metric vs. the incumbent) back into the same conversation to fix or improve, looping until
   the experiment beats the incumbent or its time budget runs out. Every session is
   snapshot-committed so a timeout or late regression never loses a working state, and each
   session's step outcomes are recorded on the attempt. Holdout results are computed only at
   the final gate - agents never see them - and the best valid experiment is promoted.
6. After the round, the findings open on your screen as `round-1-findings.md` (per-experiment
  scores plus the director's reflection), and the director writes `round-2-plan.md` (2-5
   parallel researches) after studying the errors of past attempts (worst SKUs, weekday bias,
   horizon decay) and the data itself. Each approval starts the next round. The session's
   `README.md` indexes every round with its status, so the folder reads like a lab notebook of
   all the research tried and how it scored.



## Setup

You need Python 3.12+, `uv`, `git`, OpenCode, and an OpenRouter API key.

```bash
brew install anomalyco/tap/opencode
uv sync
cp .env.example .env
```

Add your key to `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key
# optional - only needed to ingest from a Palantir Foundry dataset:
FOUNDRY_HOSTNAME=yourstack.palantirfoundry.com
FOUNDRY_TOKEN=your-foundry-token
```

`.env` is discovered by searching upward from the directory you run in, so it works both from a
repo checkout and when the CLI is installed globally.

### Install as a global command (optional)

To run `autoresearch` from any repo (not just this checkout), install it as a `uv` tool:

```bash
uv tool install .            # from this repo
# or straight from git, without cloning:
uv tool install "git+<repo-url>"
```

This puts `autoresearch` on your PATH (via `~/.local/bin`). The installed tool is a snapshot: after
you change the source, rerun `uv tool install . --reinstall` to update it - or use
`uv run autoresearch ...` from the repo, which always runs the working-tree code.

### Check your setup

```bash
autoresearch doctor          # or: uv run autoresearch doctor
```

`doctor` verifies the external tools it shells out to (`git`, `uv`, `opencode`), reports which
`.env` was loaded, and pings OpenRouter to confirm your key is valid and show remaining credit. It
also reports optional Foundry credentials. Pass `--no-api` to skip the live network check. It exits
non-zero if anything required is missing, so it doubles as an onboarding gate for a new engineer:

```text
 ✓  python      3.12.12
 ✓  git         git version 2.39.5
 ✓  uv          uv 0.11.7
 ✓  opencode    1.18.15
 ✓  .env        /path/to/your/repo/.env
 ✓  openrouter  key valid; used $12.40 of $50
 !  foundry     not configured (only needed for Foundry ingestion)
```



## Run it on the demo project

`demo/retail-demand-forecasting` looks like a real data science repo: 32 weeks of daily sales
for 8 SKUs (including intermittent slow movers) - intentionally tiny so repeated training and
evaluation stays quick on camera - a README, a production seasonal-naive model in
`models/seasonal_baseline.py`, and a candidate ARIMA model in `models/arima.py`. Generate its
data, then start:

```bash
uv run python demo/retail-demand-forecasting/scripts/make_dataset.py
uv run autoresearch start demo/retail-demand-forecasting
```

This opens a streaming chat with the Research Director, similar to a coding agent. You see its
thinking tokens live and type replies in a box. Typing `/` pops a completion menu of every
command (Tab or arrows to pick). Two things must be clear before research starts - the goal and
the baseline. Set them with slash commands or just say them:

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

## Watching a round: the live agent dashboard

While a round runs, the terminal shows one live dashboard with a row per agent:

```
Round 1 · baseline wmape 0.1039 · 3 agents · spend $0.42

  #  Experiment                 Phase            Current action               Elapsed
 >1  Global XGBoost             Editing          write solution/train.py        01:42
  2  Intermittent-demand route  Testing          running local backtest         01:37
  3  ARIMA + promo              Protected eval   validation + holdout splits    01:31

Selected: read solution/train.py → edit features.py → running uv test · spend $0.18
Log: runs/t/.../logs/aaa11111.jsonl
[1-9/↑↓] select   [c] chat   [x] cancel selected   [q] stop round   [?] help
```

Phases are factual - `Setting up`, `Exploring`, `Editing`, `Testing`, `Committing`,
`Protected eval`, then `Passed` / `Failed` / `Rejected` / `Cancelled` - and the current
action comes straight from the agent's streamed events (which file it read or edited,
which command it ran). The header shows the round's running model spend; the selected row
shows a short trail of recent actions, that agent's spend, and the path to its full JSONL log.

Interruption is first-class:

- `c` opens a chat with the Research Director about the running agents without pausing them.
  It builds a fresh snapshot of each agent's phase, elapsed time, and recent actions so you can
  ask "what is agent 2 doing?" or "why did agent 3 fail?" mid-round; the table resumes when you
  leave the chat.
- `x` cancels the selected agent only. Its subprocesses are killed, the attempt is
  recorded as `cancelled` with its partial log kept, its worktree is removed, and the
  other agents keep running.
- `q` cancels every agent and ends the research run cleanly after the round is
  persisted.
- `autoresearch stop` (from another terminal) now takes effect mid-round within
  seconds instead of waiting for all agents to finish.

Outside a real terminal (CI, piped output) the dashboard degrades to periodic
plain-text summary lines and the key controls are disabled.

## Where the Markdown documentation is stored

The human-facing research notebook is written inside the project you passed to `start`:

```text
<project>/research/
└── 001-reduce-wmape/
    ├── README.md
    ├── round-1-plan.md
    ├── round-1-findings.md
    ├── round-2-plan.md
    └── round-2-findings.md
```

- `research/<session>/README.md` is the session index. It records the goal, metric, baseline,
  every round's status, and links to its plan and findings.
- `round-N-plan.md` is the editable plan awaiting human review. Its frontmatter contains the run
  settings; its experiment sections are exactly what the coding agents execute after you type
  `execute`.
- `round-N-findings.md` is written after execution. It records each experiment's validation and
  hidden-holdout score, status, promotion decision, and the Research Director's reflection.
- Each new `start` creates the next numbered, goal-named session directory. All rounds from that
  session stay together.

Supporting Markdown used internally is stored separately:

- `<project>/.autoresearch/survey.md` caches the coding agent's repository survey.
- `<project>/runs/<task>/<timestamp>/notes.md` stores the director's round-by-round reflections
  used when resuming or proposing later rounds.
- `autoresearch report ... -o report.md` creates a standalone report wherever you specify.

The visible `research/` folder is the documentation intended for humans to browse, edit, and
commit. `.autoresearch/` and `runs/` contain execution state and lower-level artifacts.

To see the no-baseline path, delete `models/` from the demo repo and start again: the director
runs EDA (seasonality, intermittency, promo/price drivers) and bootstraps a first model from its
skills instead.

## No YAML, just slash commands

There is nothing to configure up front: defaults come from code, and you edit them inline in the
chat, Cursor/Claude style:


| command                      | what it does                                                               |
| ---------------------------- | -------------------------------------------------------------------------- |
| `/goal <text>`               | what the research must improve                                             |
| `/baseline <path>`           | pin an existing model script as the baseline                               |
| `/n_agents <1-5>`            | parallel researches per round                                              |
| `/metric <name|description>` | wmape, mape, rmse, bias_pct - or describe a custom metric in plain English |
| `/guardrail <expr>`          | add a guardrail, e.g. `bias_pct within -8..8`                              |
| `/rounds <n>`                | maximum research rounds                                                    |
| `/timeout <seconds>`         | coding agent timeout for one session of the build-evaluate-fix loop        |
| `/budget <seconds>`          | total per-experiment wall clock across all sessions (default 3600)         |
| `/status`, `/help`           | show settings / commands                                                   |


The agreed setup is saved internally to `<repo>/.autoresearch/task/task.yaml` so runs can be
resumed and inspected; you never write or edit it. Non-interactive commands work against it:

```bash
uv run autoresearch validate -c demo/retail-demand-forecasting/.autoresearch/task/task.yaml
uv run autoresearch run -c demo/retail-demand-forecasting/.autoresearch/task/task.yaml --parallel 3
uv run autoresearch resume -c demo/retail-demand-forecasting/.autoresearch/task/task.yaml
```

`run`, `resume`, and `start` accept `--max-cost <usd>` to cap model spend (see
[Cost tracking](#cost-tracking)).



## Ingest data: a raw sales CSV or a Foundry dataset

Without a repo, `ingest` turns a sales history into a protected forecasting task. The standard
schema is `date`, `sku_name`, `sales`, and `selling_price`:

```bash
uv run autoresearch ingest sales.csv --name my-forecast
uv run autoresearch run -c tasks/my-forecast/task.yaml
uv run autoresearch metric -c tasks/my-forecast/task.yaml \
  "Penalize under-forecasting twice as much as over-forecasting"
```

If your columns are named differently, map them onto the standard schema:

```bash
uv run autoresearch ingest sales.csv --name my-forecast \
  --date-column day --id-column item_id --target-column units --price-column unit_price
```

You can also read directly from a **Palantir Foundry** dataset (needs `FOUNDRY_HOSTNAME` and
`FOUNDRY_TOKEN` in `.env`; the token needs the `api:datasets-read` scope). Pass a
`foundry://` reference or a bare dataset RID, plus an optional `--branch`:

```bash
uv run autoresearch ingest foundry://ri.foundry.main.dataset.<uuid> \
  --name pan-india-demo --branch sk/my_branch \
  --date-column date --id-column fsn --target-column units --price-column fsp
```

### Data-quality gate

Before it writes anything, `ingest` shows a per-column quality report - how many rows are
null, blank, or the wrong type - and how many rows would be dropped during cleaning, then asks
you to confirm:

```text
                Data quality
  Column         Unusable rows   % of source
  date                       0          0.0%
  sku_name                   0          0.0%
  sales                     12          0.4%
  selling_price            847         31.8%
847 of 2664 rows (31.8%) dropped during cleaning
Proceed with ingestion using the cleaned data? [y/N]
```

- `--max-drop-pct <pct>` (default `30`) hard-fails the ingest if more than that fraction of rows
  would be thrown away, so a mis-mapped column can't silently produce a tiny, unrepresentative
  task.
- `--yes` / `-y` skips the confirmation prompt for scripts and CI.



## Research Director skills and tools

The outer loop includes eight implementation-grade demand forecasting playbooks. It chooses the
relevant skills each round based on the data, past results, holdout gaps, bias, and guardrail
failures, and cites them per experiment in the plan file (`skills:` line - your edits there are
honored too). The library covers model-family routing; global boosting with XGBoost, LightGBM,
and CatBoost; ARIMA/SARIMAX/ETS statistical models; current Chronos-Bolt and Chronos-2 APIs;
Croston/SBA/TSB and hurdle models for intermittent demand; retail data semantics such as
stockout-censored sales, planned promotions, and cold starts; temporal validation and explicit
leakage tests; and constrained ensembling/calibration. Each model skill includes concrete APIs,
failure modes, dependency/runtime checks, fallbacks, and required verification before reporting.
The director uses the playbooks to design experiments, and each worker receives the full bodies
of the skills cited by its experiment—not just their names.

```bash
uv run autoresearch skills
```

Between rounds the director also has analysis tools: EDA over the training data (profile,
seasonality, intermittency, promo/price drivers) and error analysis over every scored attempt's
validation forecasts (`analyze_errors`, `worst_items`). What it consulted is printed with each
round.

## Cost tracking

Every model turn's paid cost and token counts are recorded in each agent's JSONL log, so the CLI
knows exactly what each experiment spent (training and evaluation run locally and cost nothing).
Spend surfaces everywhere you look at a run:

- the live dashboard header (round total) and the selected agent's row (per-agent spend),
- the `leaderboard` (a `Cost` column) and the `report` (total spend plus per-experiment cost),
- the saved run `state.json` (`cost_usd`), and a `Total model spend` line printed at the end of a run.

Cap spend with `--max-cost` on `run`, `resume`, or `start`:

```bash
uv run autoresearch run -c tasks/my-forecast/task.yaml --max-cost 5
```

The cap is checked before each new round begins: once total spend reaches the budget the run stops
cleanly (it does not kill agents mid-round, so a round already in flight can slightly overshoot).

## View results

```bash
uv run autoresearch status
uv run autoresearch leaderboard
uv run autoresearch show ATTEMPT_ID --run-dir <repo>/runs/TASK/TIMESTAMP
uv run autoresearch report --run-dir <repo>/runs/TASK/TIMESTAMP -o report.md
uv run autoresearch stop
```

Results, patches, agent logs, metrics, per-attempt validation forecasts, and research notes are
saved under `<repo>/runs/`.

## CLI command reference

Every command. Prefix with `uv run` when working from the repo checkout, or call `autoresearch`
directly if you installed it globally (`uv tool install .`). Run `autoresearch --help` or
`autoresearch <command> --help` for the authoritative flag list.

### `autoresearch start [REPO]`
Interactive setup + research on a project. `REPO` defaults to the current directory. Surveys the
repo, settles goal/baseline/metric in chat, then runs rounds behind the plan-file review gate.

```bash
autoresearch start .
autoresearch start path/to/project --max-cost 5
```

- `--max-cost <usd>` — stop before a new round once model spend hits this.

### `autoresearch ingest SOURCE --name NAME`
Turn a sales history (local CSV or Foundry dataset) into a protected task under `tasks/`.

```bash
autoresearch ingest sales.csv --name my-forecast
autoresearch ingest foundry://ri.foundry.main.dataset.<uuid> --name pan-india \
  --branch sk/my_branch --date-column date --id-column fsn \
  --target-column units --price-column fsp
```

- `--name, -n` — task name (required).
- `--tasks-root <dir>` — output directory (default `tasks`).
- `--branch <name>` — Foundry branch to read.
- `--id-column / --date-column / --target-column / --price-column` — map source columns onto the
  standard `sku_name / date / sales / selling_price` schema.
- `--validation-days <n>` / `--holdout-days <n>` — override the split horizons.
- `--max-drop-pct <pct>` — refuse ingest if more than this % of rows drop (default `30`).
- `--overwrite` — replace an existing task of the same name.
- `--yes, -y` — skip the data-quality confirmation prompt.

### `autoresearch run -c TASK_YAML`
Start a new non-interactive research run from a task config (no review gate).

```bash
autoresearch run -c tasks/my-forecast/task.yaml --parallel 3 --max-cost 5
```

- `--goal <text>` — override the configured goal.
- `--guardrail <expr>` — add a guardrail (repeatable).
- `--parallel <n>` — parallel experiments per round.
- `--max-experiments <n>` — cap total experiments.
- `--max-cost <usd>` — spend cap (checked before each new round).

### `autoresearch resume -c TASK_YAML`
Resume the latest (or a chosen) run: review the next proposed round, then continue. Keeps the
promoted incumbent, baseline, and notes.

```bash
autoresearch resume -c tasks/my-forecast/task.yaml
autoresearch resume -c tasks/my-forecast/task.yaml --run-dir runs/my-forecast/20260810-093424
```

- `--run-dir <dir>` — resume a specific run instead of the latest.
- `--parallel <n>`, `--max-experiments <n>`, `--max-cost <usd>` — as for `run`.

### `autoresearch metric -c TASK_YAML "DESCRIPTION"`
Create and verify a custom metric from plain English (shows a hand-worked example and self-checks
the grader code).

```bash
autoresearch metric -c tasks/my-forecast/task.yaml \
  "Penalize under-forecasting twice as much as over-forecasting"
```

- `--yes, -y` — adopt without confirmation.

### `autoresearch validate -c TASK_YAML`
Run the seed/baseline solution through the protected evaluator and print validation + holdout
metrics.

```bash
autoresearch validate -c tasks/my-forecast/task.yaml
```

### `autoresearch doctor`
Check that required tools (`git`, `uv`, `opencode`) and credentials (OpenRouter, optional Foundry)
are in place. Exits non-zero if anything required is missing.

```bash
autoresearch doctor
autoresearch doctor --no-api    # skip the live OpenRouter key check
```

### `autoresearch skills [-c TASK_YAML]`
List the forecasting playbooks available to the Research Director.

```bash
autoresearch skills
```

### `autoresearch status [--run-dir DIR]`
Print the run state (baseline, incumbent, round, counts, spend) as JSON.

```bash
autoresearch status --run-dir runs/my-forecast/20260810-093424
```

### `autoresearch leaderboard [--run-dir DIR] [-c TASK_YAML]`
Show ranked, guardrail-passing attempts with metrics and per-experiment cost.

```bash
autoresearch leaderboard -c tasks/my-forecast/task.yaml
```

### `autoresearch show ATTEMPT_ID [--run-dir DIR]`
Print one attempt's full record and its diff.

```bash
autoresearch show a1b2c3d4 --run-dir runs/my-forecast/20260810-093424
```

### `autoresearch report [--run-dir DIR] [-o OUTPUT]`
Generate a markdown research report (outcome, total spend, promoted experiments, full ledger).

```bash
autoresearch report --run-dir runs/my-forecast/20260810-093424 -o report.md
```

### `autoresearch stop`
Request a stop from another terminal: mid-round agents are cancelled within seconds and recorded
as cancelled.

```bash
autoresearch stop
```

## Train on Palantir Foundry (`runtime: foundry`)

The full research loop can train on Foundry compute instead of locally: agents rewrite a
Foundry Python transform, each experiment is pushed to its own Foundry branch and built there,
and the CLI reads the branch's `forecasts` output plus the sealed actuals back through
readTable to score it. Because Foundry datasets are branch-aware, parallel agents never
collide — each gets its own published transform and its own forecasts branch. The winning
experiment is merged and pushed to `master`, so promotion literally ships the model.

Requires `FOUNDRY_HOSTNAME` and `FOUNDRY_TOKEN` in `.env` (scopes: `api:datasets-read`,
`api:datasets-write`, `api:orchestration-write`).

The installer scaffolds a **three-stage pipeline** into the transforms repo, so the whole
lifecycle is visible in Foundry:

```
sales_raw ─▶ data_preprocessing/ ─▶ sales_train ─▶ model_running/ ─▶ forecasts
                                                                        │
                        evaluation_metrics ◀─ model_evaluation/ ◀───────┘
```

- `data_preprocessing/preprocess.py` — fixed infrastructure. Cleans the raw feed:
  deduplicates, clips negative demand, fills null demand, restores missing calendar dates.
- `model_running/forecast.py` — the only file agents may edit.
- `model_evaluation/evaluate.py` — fixed infrastructure. Writes per-SKU + overall metrics
  to the `evaluation_metrics` dataset so scores are visible inside Foundry too.

### `autoresearch foundry-setup --repo <transforms-repo>`
One-shot installer for any cloned Foundry Python transforms repository. Reads the repo's
`gradle.properties` for its identity, provisions the pipeline datasets (`sales_raw`,
`sales_train`, `forecast_request`, `forecasts`, `validation_actuals`, `holdout_actuals`,
`evaluation_metrics`) under the project's `autoresearch/` folder, scaffolds the three
pipeline stages + `AUTORESEARCH.md` contract + curated conda dependencies, pushes so the
baseline publishes, and writes `foundry-task.yaml`. Only the inputs are uploaded; the
`sales_train`, `forecasts`, and `evaluation_metrics` datasets are filled by Foundry builds.
An existing `datasets/forecast.py` (e.g. an agent-improved winner) is migrated into
`model_running/`, never clobbered.

```bash
autoresearch foundry-setup --repo ~/code/my-foundry-repo
autoresearch foundry-build -c foundry-task.yaml --all   # first full pipeline build
autoresearch run -c foundry-task.yaml        # the autonomous loop, training on Foundry
```

- `--n-skus`, `--n-days`, `--validation-days`, `--holdout-days` — synthetic data shape.
- `--no-push` — scaffold and provision only.

**Bring your own data.** If the datasets already exist on Foundry, skip the synthetic
provisioning entirely with `--dataset key=RID` (repeatable). Required: `sales_train`,
`forecast_request`, `validation_actuals`, `holdout_actuals`. The `forecasts` and
`evaluation_metrics` outputs are created automatically if not given; pass `sales_raw`
only if you also want the preprocessing stage scaffolded. Column names are configurable
and flow into the scaffolded transforms, the contract, and the scorer:

```bash
autoresearch foundry-setup --repo ~/code/my-foundry-repo \
  --dataset sales_train=ri.foundry.main.dataset.aaa \
  --dataset forecast_request=ri.foundry.main.dataset.bbb \
  --dataset validation_actuals=ri.foundry.main.dataset.ccc \
  --dataset holdout_actuals=ri.foundry.main.dataset.ddd \
  --id-col item_id --date-col ds --target-col demand --metric rmse
```

Custom metrics work on Foundry too: `autoresearch metric` turns a plain-English
definition into verified code and points the task config's `metric.definition` at it;
the Foundry scorer computes it alongside wmape/mape/rmse/bias_pct.

### `autoresearch foundry-build -c foundry-task.yaml [-m MSG] [--all] [--force] [--no-push]`
Manual one-shot: push the repo, wait for the transform to publish, build `forecasts` on the
configured branch, and poll to completion. With `--all` it builds the entire pipeline
(preprocessing → model → evaluation) in one build, jobs ordered by dependency. Useful for
demos and debugging outside the loop.

### `autoresearch foundry-score -c foundry-task.yaml [--holdout]`
Read the current `forecasts` output and the sealed actuals from Foundry and print the metrics
table without triggering a build.

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

## How we ship

This repo follows the Hyde SDLC standard (OPS-230). Two permanent branches:

| Branch | Role |
| --- | --- |
| `staging` | Integration branch. Feature branches merge here first. |
| `main` | Production source of truth. Only receives release PRs from `staging`. |

```
feature branch → PR → staging → verify → PR → main → tag vX.Y.Z
```

- Branch from the latest `staging` and open a PR back into `staging`. CI (lint, test,
  build) must be green and the PR needs one approving review. Put the Linear ticket ID
  in the PR title.
- "Deploying to staging" for this CLI means installing from the `staging` branch and
  smoke-testing it:

```bash
uv tool install --force git+https://github.com/Hyde-Inc/autoresearch@staging
autoresearch doctor
```

- Releases: open a PR from `staging` into `main` with the release sign-off checklist,
  merge, then create an immutable tag `vMAJOR.MINOR.PATCH` and a GitHub Release.
  "Production" installs come from the tag:

```bash
uv tool install --force git+https://github.com/Hyde-Inc/autoresearch@v0.1.0
```

- Rollback: reinstall the previous tag with the same command.
- Never push directly to `staging` or `main`; both are protected. Hotfixes go through a
  PR into `main`, get a PATCH tag, and are merged back to `staging` the same day.
- Secrets live in `.env` (never committed); required variable names are listed in
  `.env.example`.