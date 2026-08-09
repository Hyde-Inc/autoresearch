"""Web dashboard backend: serves run data from the runs/ directory."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import load_config
from .ingest import ingest_csv, preview_csv
from .metrics import (
    MetricInterpreter,
    MetricSpec,
    adopt_spec,
    eval_columns,
    load_task_spec,
    validate_spec,
)
from .report import build_report

STATIC_DIR = Path(__file__).parent / "web" / "static"
DEFAULT_CONFIG = "examples/demand_forecasting/task.yaml"


def _run_dir(runs_root: Path, task: str, run: str) -> Path:
    path = (runs_root / task / run).resolve()
    if not str(path).startswith(str(runs_root.resolve())) or not path.is_dir():
        raise HTTPException(status_code=404, detail="run not found")
    return path


def _load_state(run_dir: Path) -> dict:
    state_file = run_dir / "state.json"
    if not state_file.exists():
        return {}
    return json.loads(state_file.read_text())


def _load_attempts(run_dir: Path) -> list[dict]:
    attempts_file = run_dir / "attempts.jsonl"
    if not attempts_file.exists():
        return []
    result = []
    for line in attempts_file.read_text().splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def _parse_notes(run_dir: Path) -> list[dict]:
    notes_file = run_dir / "notes.md"
    if not notes_file.exists():
        return []
    sections: list[dict] = []
    current: dict | None = None
    for line in notes_file.read_text().splitlines():
        if line.startswith("## "):
            if current:
                current["body"] = current["body"].strip()
                sections.append(current)
            current = {"heading": line[3:].strip(), "body": ""}
        elif current is not None:
            current["body"] += line + "\n"
    if current:
        current["body"] = current["body"].strip()
        sections.append(current)
    return sections


def _parse_agent_log(log_path: Path) -> dict:
    """Convert an OpenCode JSONL session log into renderable turns."""
    turns: list[dict] = []
    total_tokens = 0
    total_cost = 0.0
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        part = event.get("part") or {}
        timestamp = event.get("timestamp")
        if etype == "text":
            text = part.get("text", "").strip()
            if text:
                turns.append({"type": "text", "text": text, "timestamp": timestamp})
        elif etype == "tool_use":
            state = part.get("state") or {}
            input_data = state.get("input") or {}
            summary = (
                input_data.get("filePath")
                or input_data.get("command")
                or input_data.get("pattern")
                or input_data.get("description")
                or ""
            )
            turns.append(
                {
                    "type": "tool",
                    "tool": part.get("tool", "?"),
                    "status": state.get("status", "?"),
                    "summary": str(summary)[:200],
                    "error": (state.get("error") or "")[:200] or None,
                    "timestamp": timestamp,
                }
            )
        elif etype == "step_finish":
            tokens = part.get("tokens") or {}
            total_tokens += tokens.get("output", 0) + tokens.get("input", 0)
            cost = part.get("cost")
            if isinstance(cost, (int, float)):
                total_cost += cost
    return {"turns": turns, "total_tokens": total_tokens, "total_cost_usd": round(total_cost, 4)}


class StartRunRequest(BaseModel):
    config: str = DEFAULT_CONFIG
    goal: str | None = None
    parallel: int | None = None
    max_experiments: int | None = None
    rounds: int | None = None
    round_timeout_s: int | None = None
    cost_limit_usd: float | None = None
    tracks: list[str] = []
    guardrails: list[str] = []


class InterpretMetricRequest(BaseModel):
    description: str
    config: str = DEFAULT_CONFIG


class ConfirmMetricRequest(BaseModel):
    spec: dict
    config: str = DEFAULT_CONFIG


class SaveBaselineRequest(BaseModel):
    config: str
    code: str


def create_app(runs_root: Path, project_root: Path | None = None) -> FastAPI:
    runs_root = runs_root.resolve()
    project_root = (project_root or Path.cwd()).resolve()
    app = FastAPI(title="Autoresearch Dashboard")

    @app.get("/api/runs")
    def list_runs() -> dict:
        tasks = []
        if runs_root.is_dir():
            for task_dir in sorted(runs_root.iterdir()):
                if not task_dir.is_dir():
                    continue
                runs = []
                for run_dir in sorted(task_dir.iterdir(), reverse=True):
                    if not run_dir.is_dir():
                        continue
                    state = _load_state(run_dir)
                    attempts = _load_attempts(run_dir)
                    metric = state.get("primary_metric", "wmape")
                    baseline = (state.get("baseline") or {}).get(metric)
                    best = (state.get("incumbent") or {}).get(metric)
                    improvement = (
                        (baseline - best) / baseline
                        if baseline and best is not None and baseline > 0
                        else None
                    )
                    promoted = sum(1 for a in attempts if a.get("promoted"))
                    runs.append(
                        {
                            "run": run_dir.name,
                            "status": state.get("status", "unknown"),
                            "round": state.get("round", 0),
                            "completed": state.get("completed", len(attempts)),
                            "attempts": len(attempts),
                            "goal": state.get("goal", ""),
                            "primary_metric": metric,
                            "created_at": state.get("created_at"),
                            "run_config": state.get("run_config"),
                            "total_cost_usd": state.get("total_cost_usd"),
                            "baseline_metric": baseline,
                            "best_metric": best,
                            "improvement": improvement,
                            "promoted": promoted,
                        }
                    )
                if runs:
                    tasks.append({"task": task_dir.name, "runs": runs})
        return {"tasks": tasks}

    @app.get("/api/runs/{task}/{run}/state")
    def get_state(task: str, run: str) -> dict:
        return _load_state(_run_dir(runs_root, task, run))

    @app.get("/api/runs/{task}/{run}/attempts")
    def get_attempts(task: str, run: str) -> list[dict]:
        return _load_attempts(_run_dir(runs_root, task, run))

    @app.get("/api/runs/{task}/{run}/notes")
    def get_notes(task: str, run: str) -> list[dict]:
        return _parse_notes(_run_dir(runs_root, task, run))

    @app.get("/api/runs/{task}/{run}/attempts/{attempt_id}/diff", response_class=PlainTextResponse)
    def get_diff(task: str, run: str, attempt_id: str) -> str:
        run_dir = _run_dir(runs_root, task, run)
        diff_path = run_dir / "attempts" / f"{attempt_id}.patch"
        if not diff_path.exists():
            raise HTTPException(status_code=404, detail="diff not found")
        return diff_path.read_text()

    @app.get("/api/runs/{task}/{run}/attempts/{attempt_id}/log")
    def get_log(task: str, run: str, attempt_id: str) -> dict:
        run_dir = _run_dir(runs_root, task, run)
        log_path = run_dir / "logs" / f"{attempt_id}.jsonl"
        if not log_path.exists():
            raise HTTPException(status_code=404, detail="log not found")
        return _parse_agent_log(log_path)

    tasks_root = project_root / "tasks"

    def _safe_config_path(relative: str) -> Path:
        config_path = (project_root / relative).resolve()
        if not str(config_path).startswith(str(project_root)):
            raise HTTPException(status_code=400, detail="config path escapes project root")
        if not config_path.exists():
            raise HTTPException(status_code=400, detail=f"config not found: {relative}")
        return config_path

    def _config_for(relative: str):
        return load_config(_safe_config_path(relative))

    @app.get("/api/tasks")
    def list_tasks() -> dict:
        """All runnable task configs (bundled example + ingested tasks)."""
        found: list[dict] = []
        candidates = [project_root / DEFAULT_CONFIG]
        if tasks_root.is_dir():
            candidates += sorted(tasks_root.glob("*/task.yaml"))
        for path in candidates:
            if not path.exists():
                continue
            try:
                raw = yaml.safe_load(path.read_text())
            except Exception:  # noqa: BLE001
                continue
            rel = os.path.relpath(path, project_root)
            spec = None
            if raw.get("metric", {}).get("definition"):
                spec = raw["metric"]
            found.append(
                {
                    "config": rel,
                    "name": raw.get("name", path.parent.name),
                    "description": raw.get("description", ""),
                    "goal": raw.get("goal", ""),
                    "metric": raw.get("metric", {}).get("name", "wmape"),
                    "custom_metric": bool(spec and spec.get("definition")),
                    "bundled": rel == DEFAULT_CONFIG,
                }
            )
        return {"tasks": found}

    @app.post("/api/ingest/preview")
    async def ingest_preview(file: UploadFile = File(...)) -> dict:
        content = await file.read()
        return dataclasses.asdict(preview_csv(content))

    @app.post("/api/ingest")
    async def ingest(
        file: UploadFile = File(...),
        name: str = Form(...),
        validation_days: int | None = Form(None),
        holdout_days: int | None = Form(None),
        overwrite: bool = Form(False),
    ) -> dict:
        content = await file.read()
        try:
            result = ingest_csv(
                content,
                name=name,
                tasks_root=tasks_root,
                validation_days=validation_days,
                holdout_days=holdout_days,
                overwrite=overwrite,
            )
        except (ValueError, FileExistsError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        payload = dataclasses.asdict(result)
        payload["config"] = os.path.relpath(result.config_path, project_root)
        payload["config_path"] = str(result.config_path)
        payload["task_dir"] = str(result.task_dir)
        return payload

    @app.get("/api/baseline")
    def get_baseline(config: str = DEFAULT_CONFIG) -> dict:
        cfg = _config_for(config)
        path = cfg.resolve(cfg.workspace.seed) / "solution" / "train.py"
        return {
            "code": path.read_text() if path.exists() else "",
            "path": os.path.relpath(path, project_root) if path.exists() else None,
        }

    @app.post("/api/baseline")
    def save_baseline(request: SaveBaselineRequest) -> dict:
        cfg = _config_for(request.config)
        path = cfg.resolve(cfg.workspace.seed) / "solution" / "train.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(request.code)
        return {"ok": True, "path": os.path.relpath(path, project_root)}

    @app.get("/api/runs/{task}/{run}/report", response_class=PlainTextResponse)
    def get_report(task: str, run: str) -> str:
        return build_report(_run_dir(runs_root, task, run))

    @app.get("/api/metric")
    def get_metric(config: str = DEFAULT_CONFIG) -> dict:
        cfg = _config_for(config)
        spec = load_task_spec(cfg)
        return {
            "name": cfg.metric.name,
            "direction": cfg.metric.direction,
            "custom": spec.model_dump() if spec else None,
            "columns": eval_columns(cfg),
        }

    @app.post("/api/metric/interpret")
    async def interpret_metric(request: InterpretMetricRequest) -> dict:
        cfg = _config_for(request.config)
        columns = eval_columns(cfg)
        interpreter = MetricInterpreter(cfg.director.model, cfg.director.temperature)
        try:
            spec, validation = await interpreter.interpret(request.description, columns)
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "spec": spec.model_dump(),
            "validation": validation.model_dump(),
            "columns": columns,
        }

    @app.post("/api/metric/confirm")
    def confirm_metric(request: ConfirmMetricRequest) -> dict:
        cfg = _config_for(request.config)
        spec = MetricSpec.model_validate(request.spec)
        validation = validate_spec(spec, eval_columns(cfg))
        if not validation.passed:
            raise HTTPException(status_code=422, detail=validation.failure_summary())
        saved = adopt_spec(cfg, spec)
        return {
            "ok": True,
            "definition": str(saved),
            "name": spec.name,
            "direction": spec.direction,
        }

    @app.post("/api/start")
    def start_run(request: StartRunRequest) -> dict:
        config_path = (project_root / request.config).resolve()
        if not config_path.exists():
            raise HTTPException(status_code=400, detail=f"config not found: {request.config}")
        command = [sys.executable, "-m", "autoresearch.cli", "run", "-c", str(config_path)]
        if request.goal:
            command += ["--goal", request.goal]
        if request.parallel:
            command += ["--parallel", str(request.parallel)]
        if request.rounds:
            command += ["--rounds", str(request.rounds)]
        if request.round_timeout_s:
            command += ["--round-timeout", str(request.round_timeout_s)]
        if request.cost_limit_usd is not None:
            command += ["--cost-limit", str(request.cost_limit_usd)]
        if request.max_experiments:
            command += ["--max-experiments", str(request.max_experiments)]
        for focus in request.tracks:
            command += ["--track", focus]
        for guardrail in request.guardrails:
            command += ["--guardrail", guardrail]
        log_file = project_root / "dashboard-run.log"
        with log_file.open("a") as handle:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )
        return {"ok": True, "pid": process.pid}

    @app.post("/api/stop")
    def stop_run() -> dict:
        (project_root / ".autoresearch-stop").write_text("stop\n")
        return {"ok": True}

    if STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = STATIC_DIR / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")

    return app
