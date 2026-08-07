"""Web dashboard backend: serves run data from the runs/ directory."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import load_config
from .metrics import (
    MetricInterpreter,
    MetricSpec,
    adopt_spec,
    eval_columns,
    load_task_spec,
    validate_spec,
)

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
    guardrails: list[str] = []


class InterpretMetricRequest(BaseModel):
    description: str
    config: str = DEFAULT_CONFIG


class ConfirmMetricRequest(BaseModel):
    spec: dict
    config: str = DEFAULT_CONFIG


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
                    runs.append(
                        {
                            "run": run_dir.name,
                            "status": state.get("status", "unknown"),
                            "round": state.get("round", 0),
                            "completed": state.get("completed", len(attempts)),
                            "attempts": len(attempts),
                            "goal": state.get("goal", ""),
                            "primary_metric": state.get("primary_metric", "wmape"),
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

    def _config_for(relative: str):
        config_path = (project_root / relative).resolve()
        if not config_path.exists():
            raise HTTPException(status_code=400, detail=f"config not found: {relative}")
        return load_config(config_path)

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
        if request.max_experiments:
            command += ["--max-experiments", str(request.max_experiments)]
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
