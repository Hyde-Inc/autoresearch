# Demo guide: parallel autoresearch for demand forecasting

A step-by-step script for demoing the product end to end - what to run, what to
say, and which component to put on screen at each moment. Full run takes
20-30 minutes; the "fast settings" below keep every wait short.

## 0. Reset and prep (before recording)

```bash
cd ~/autoresearch
rm -rf demo/retail-demand-forecasting/.autoresearch \
       demo/retail-demand-forecasting/research \
       demo/retail-demand-forecasting/forecasts runs
uv run python demo/retail-demand-forecasting/scripts/make_dataset.py
cat .env   # OPENROUTER_API_KEY must be set
```

This gives a pristine "client repo". Keep `demo/retail-demand-forecasting/.autoresearch`
instead if you want the repo survey pre-cached (setup then skips the ~2-minute
background survey entirely).

Open in the editor beforehand, on a second screen or tab:

- `demo/retail-demand-forecasting/README.md` - the "client's" project
- `src/autoresearch/skills_library/chronos-playbook.md` and `xgboost-demand.md`

## 1. The pitch (30 seconds, before typing anything)

> "This is an autonomous ML research lab for demand forecasting. You point it
> at your data science repo, tell it what metric to improve, and it runs
> parallel coding agents that design, build, and evaluate experiments - with a
> human approving every research plan before it runs. It embeds our team's
> forecasting expertise, so it's not a generic agent: it knows when to
> fine-tune Chronos and when zero-shot is enough, which XGBoost objective fits
> intermittent demand, how to correct forecast bias."

Show the client repo README: a realistic project - 14 months of daily sales
for 24 SKUs, a production seasonal-naive model, a candidate ARIMA model.

## 2. Show the built-in expertise first

```bash
uv run autoresearch skills
```

> "These nine playbooks are our researchers' experience, written down: model
> selection, intermittent demand, promo signals, bias correction, ensembling,
> and deep playbooks for Chronos and XGBoost. The director cites them by name
> in every plan it writes - this is the moat."

Open `chronos-playbook.md` briefly: point at the situation-conditioned rules
(when zero-shot beats fine-tuning, when Chronos is the wrong tool).

## 3. Start the session

```bash
uv run autoresearch start demo/retail-demand-forecasting
```

The chat opens **instantly**; a dim line says a coding agent is surveying the
repository in the background.

> "An OpenCode subagent is exploring the repo behind the scenes - reading the
> models, profiling the data, learning the evaluation conventions. We don't
> wait for it: the director already works from a repo inventory and its own
> read/EDA tools, and the survey folds into its context the moment it's done."

## 4. Configure with slash commands

Type `/` and pause - the completion menu pops with every command and its
description. Pick or type:

```
/n_agents 2
/timeout 300
/goal reduce wmape
/baseline models/arima.py
go
```

> "No YAML, no config files. Slash commands set the run parameters inline -
> and everything has sensible defaults. We pin the team's candidate ARIMA
> model as the baseline: the research has to beat their best current
> approach, not a strawman."

(`/n_agents 2` and `/timeout 300` are the fast settings; drop them for a
fuller run with 3 agents and 20-minute experiment budgets.)

### Optional: the custom metric moment

Before `go`, show metric definition in plain English:

```
/metric penalize under-forecasting twice as much as over-forecasting
```

The director restates the metric, works a hand-calculated example, shows the
exact grader code, and self-checks that the code reproduces the number. Confirm
with `yes`.

> "Every team scores forecasts differently. You describe your objective in
> plain English; the system derives verified grader code and scores every
> experiment with it. Reward hacking is off the table - agents never see the
> holdout actuals."

## 5. Lock-in: protected workspace and baseline

After `go`, narrate as the director works (~2-3 minutes):

> "It's now building a protected task: splitting history into train,
> validation, and a hidden holdout whose actuals are sealed away from the
> agents. Then it ports the ARIMA model faithfully as the incumbent."

When the baseline evaluation panel appears, point at the two numbers:

> "That's the bar: the baseline's WMAPE on validation, and on a hidden holdout
> the agents can never touch."

## 6. The research plan (the plan-mode moment)

The round-1 plan **pops open in the editor automatically** at
`demo/retail-demand-forecasting/research/001-reduce-wmape/round-1-plan.md`.

> "Like a coding agent's plan mode. YAML frontmatter holds the run parameters;
> each experiment has a hypothesis and instructions, and cites which of our
> playbooks informed it - see `skills: xgboost-demand`."

**Edit the file on camera**: reword a hypothesis, or delete an experiment, or
change `n_agents` in the frontmatter. Then type in the chat:

```
execute
```

> "The edited file is exactly what runs. Feedback instead of execute would have
> the director rewrite the plan; stop ends the session."

## 7. Round 1 runs (~5 minutes with fast settings)

> "Each experiment gets its own coding agent in its own git worktree - they
> can only edit the solution, and a protected evaluator scores WMAPE, MAPE,
> RMSE, bias, runtime, and the hidden holdout. Guardrails catch degenerate
> wins, and the best valid experiment is promoted as the new incumbent."

While waiting, open the session `README.md` in the same folder - the lab
notebook index showing every round and its status.

## 8. Findings

When the round completes, `round-1-findings.md` **opens on screen
automatically**: per-experiment scores, holdout numbers, promotion verdict,
and the director's reflection.

> "Full paper trail: what was tried, what it scored, what the director learned."

## 9. Round 2: the director learns

The director studies the errors before proposing again - worst SKUs, weekday
bias, horizon decay - then writes `round-2-plan.md` (2-5 parallel researches)
and it opens in the editor.

> "It's not guessing twice. It analyzed where round 1 failed and is doubling
> down on what worked - that error analysis plus the playbooks is why round 2
> is smarter than round 1."

Same gate: edit, then `execute` for another round, or `stop` to end the demo.

## 10. Wrap-up

```bash
uv run autoresearch leaderboard   # ranked, guardrail-passing attempts
uv run autoresearch report        # markdown research report
```

Finish on the `research/001-reduce-wmape/` folder in the editor:

> "This folder is the permanent lab notebook - every plan, every result, every
> reflection, indexed in a README. New team members read this to know exactly
> what's been tried and why."

## Timing cheat sheet

| Step | Duration |
|---|---|
| Chat opens | instant |
| Background survey | ~2 min (skipped when cached) |
| Custom metric verification | ~1 min |
| Lock-in + baseline eval | ~2-3 min |
| Round of 2 agents at /timeout 300 | ~5-6 min |
| Director reflection + round-2 plan | ~2 min |

## Troubleshooting

- **No files pop open**: the opener uses the `cursor`/`code` CLI or the Cursor
  app bundle; `AUTORESEARCH_NO_OPEN=1` disables it (make sure it isn't set).
- **Experiments time out**: raise `/timeout` (default 1200 s) - Chronos or
  heavy XGBoost sweeps need more than 300 s.
- **Session ends unexpectedly**: `uv run autoresearch resume` picks up the
  latest run at the next review gate.
