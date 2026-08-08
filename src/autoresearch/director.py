from __future__ import annotations

import asyncio
import json
import os
import re

from openai import OpenAI

from . import eda, evals
from .config import TaskConfig
from .metrics import load_task_spec
from .models import Attempt, Idea
from .skills import (
    SkillSelection,
    load_skills,
    select_skills,
    selected_skill_context,
    selection_json,
)
from .store import RunStore


def openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/Hyde-Inc/autoresearch",
            "X-Title": "Parallel Autoresearch",
        },
    )


def parse_json_object(content: str) -> dict:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"director did not return a JSON object: {text[:200]!r}")


def analysis_tool_schemas() -> list[dict]:
    def schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    return [
        schema(
            "explore_training_data",
            "Run exploratory analysis on the training data before choosing model families.",
            {
                "analysis": {
                    "type": "string",
                    "enum": ["profile", "seasonality", "intermittency", "drivers"],
                }
            },
            ["analysis"],
        ),
        schema(
            "analyze_errors",
            "Study a scored attempt's validation errors: overall metrics, weekday and "
            "horizon breakdowns, and over/under-forecast counts. Use attempt_id 'baseline' "
            "for the incumbent baseline.",
            {"attempt_id": {"type": "string"}},
            ["attempt_id"],
        ),
        schema(
            "worst_items",
            "List the items contributing the most validation error for a scored attempt, "
            "with their bias direction.",
            {"attempt_id": {"type": "string"}, "limit": {"type": "integer"}},
            ["attempt_id"],
        ),
    ]


class ResearchDirector:
    def __init__(self, config: TaskConfig):
        self.config = config
        self.client = openrouter_client()
        model = config.director.model
        self.model = model.removeprefix("openrouter/")
        seed = config.resolve(config.workspace.seed)
        task_path = seed / "TASK.md"
        project_path = seed / "pyproject.toml"
        self.task_contract = task_path.read_text() if task_path.exists() else config.description
        self.project_environment = (
            project_path.read_text() if project_path.exists() else "No dependency file provided."
        )
        self.skills = load_skills(config)
        self.last_skill_selection = SkillSelection()
        self.last_analysis: list[str] = []

    async def _json_completion(self, system: str, user: str) -> dict:
        def call() -> dict:
            last_error: ValueError | None = None
            prompt = user
            for _ in range(3):
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.config.director.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response.choices[0].message.content
                if content:
                    try:
                        return parse_json_object(content)
                    except ValueError as exc:
                        last_error = exc
                else:
                    last_error = ValueError("director returned an empty response")
                prompt = (
                    user
                    + "\n\nReturn only one valid JSON object with no markdown or explanation."
                )
            assert last_error is not None
            raise last_error

        return await asyncio.to_thread(call)

    def _run_analysis_tool(self, store: RunStore, name: str, arguments: dict) -> dict:
        data = self.config.data
        try:
            if name == "explore_training_data":
                frame = eda.load_table(self.config.resolve(data.train))
                analysis = arguments.get("analysis", "profile")
                if analysis == "profile":
                    return eda.profile(frame)
                if analysis == "seasonality":
                    return eda.seasonality(frame, data.date_column, data.target_column)
                if analysis == "intermittency":
                    return eda.intermittency(
                        frame, data.id_column, data.date_column, data.target_column
                    )
                if analysis == "drivers":
                    return eda.drivers(
                        frame, data.target_column, exclude=[data.id_column, data.date_column]
                    )
                return {"error": f"unknown analysis: {analysis}"}
            frame = store.load_validation_frame(str(arguments.get("attempt_id", "")))
            if frame is None:
                return {
                    "error": "no stored validation forecasts for that attempt",
                    "available": store.list_validation_frames(),
                }
            if name == "analyze_errors":
                return evals.error_summary(
                    frame, data.id_column, data.date_column, data.target_column
                )
            if name == "worst_items":
                return evals.worst_items(
                    frame,
                    data.id_column,
                    data.date_column,
                    data.target_column,
                    limit=int(arguments.get("limit", 10)),
                )
            return {"error": f"unknown tool: {name}"}
        except Exception as exc:  # noqa: BLE001 - surfaced to the director
            return {"error": str(exc)}

    async def _json_with_analysis(
        self, system: str, user: str, store: RunStore, max_tool_rounds: int = 4
    ) -> dict:
        """Let the director inspect data and errors with tools, then demand JSON."""
        tools = analysis_tool_schemas()

        def call() -> dict:
            messages: list[dict] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            for _ in range(max_tool_rounds):
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.config.director.temperature,
                    messages=messages,
                    tools=tools,
                )
                message = response.choices[0].message
                if not message.tool_calls:
                    if message.content:
                        try:
                            return parse_json_object(message.content)
                        except ValueError:
                            pass
                    break
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in message.tool_calls
                        ],
                    }
                )
                for tool_call in message.tool_calls:
                    try:
                        arguments = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    result = self._run_analysis_tool(store, tool_call.function.name, arguments)
                    rendered = ", ".join(f"{k}={v}" for k, v in arguments.items())
                    self.last_analysis.append(f"{tool_call.function.name}({rendered})")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return the final answer now as one valid JSON object with no "
                        "markdown or explanation."
                    ),
                }
            )
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.config.director.temperature,
                response_format={"type": "json_object"},
                messages=messages,
            )
            return parse_json_object(response.choices[0].message.content or "")

        return await asyncio.to_thread(call)

    async def propose(
        self,
        *,
        count: int,
        round_number: int,
        attempts: list[Attempt],
        notes: str,
        store: RunStore | None = None,
    ) -> list[Idea]:
        self.last_analysis = []
        history = [
            {
                "title": item.idea.title,
                "hypothesis": item.idea.hypothesis,
                "status": item.status,
                "metrics": item.metrics,
                "holdout_metrics": item.holdout_metrics,
                "failures": item.guardrail_failures,
                "error": item.error,
                "promoted": item.promoted,
            }
            for item in attempts[-20:]
        ]
        situation = (
            f"Task: {self.config.description}\nGoal: {self.config.goal}\n"
            f"Business context: {self.config.context or 'not provided'}\n"
            f"Primary metric: {self.config.metric.name} ({self.config.metric.direction})\n"
            f"Guardrails: {self.config.guardrails}\nRound: {round_number}\n"
            f"Prior attempts: {json.dumps(history)}\nLab notes:\n{notes[-5000:]}"
        )
        self.last_skill_selection = await select_skills(
            situation, self.skills, self._json_completion
        )
        skill_context = selected_skill_context(self.last_skill_selection, self.skills)
        selected_names = {item.name for item in self.last_skill_selection.selected}
        system = (
            "You are the Research Director for an autonomous ML lab. Propose diverse, testable, "
            "single-change experiments. Learn from results, avoid repeating failed ideas, and "
            "respect runtime and metric guardrails. Every idea must be standalone and implementable "
            "using only installed dependencies. Agents can edit only solution/. Use the supplied "
            "forecasting skills as decision guidance. Return strict JSON with key 'ideas', an "
            "array of objects with exactly: title, hypothesis, instructions, category, skills_used. "
            "skills_used must contain only selected skill names."
        )
        if store is not None:
            system += (
                " Before answering you may call the analysis tools to study the training data "
                "and where scored attempts made their errors; ground your ideas in what you find."
            )
        spec = load_task_spec(self.config)
        metric_context = f" Definition: {spec.understanding}" if spec else ""
        user = (
            f"Task: {self.config.description}\nGoal: {self.config.goal}\n"
            f"Business context: {self.config.context or 'not provided'}\n"
            f"Task contract:\n{self.task_contract}\n"
            f"Installed project environment:\n{self.project_environment}\n"
            f"Primary metric: {self.config.metric.name} "
            f"({self.config.metric.direction}).{metric_context}\n"
            f"Guardrails: {self.config.guardrails}\nRound: {round_number}\n"
            f"Need {count} ideas.\nSuggested families (not mandatory): {self.config.idea_hints}\n"
            f"Prior attempts: {json.dumps(history)}\nLab notes:\n{notes[-8000:]}\n\n"
            f"Skill selection: {selection_json(self.last_skill_selection)}\n\n"
            f"Selected skill guidance:\n{skill_context or 'No skill selected.'}"
        )
        if store is not None:
            frames = store.list_validation_frames()
            user += (
                f"\n\nAttempts with stored validation forecasts for analyze_errors/"
                f"worst_items: {frames or 'none yet'}"
            )
            payload = await self._json_with_analysis(system, user, store)
        else:
            payload = await self._json_completion(system, user)
        ideas = [Idea.model_validate(item) for item in payload.get("ideas", [])]
        for idea in ideas:
            idea.skills_used = [name for name in idea.skills_used if name in selected_names]
        if len(ideas) < count:
            raise RuntimeError(f"director returned {len(ideas)} ideas; expected {count}")
        return ideas[:count]

    async def reflect(self, attempts: list[Attempt]) -> str:
        system = (
            "You are an ML Research Director. Summarize this round into a concise lab note: "
            "what worked, what failed, likely causal explanations, and the best next directions. "
            "Return JSON with one string key: reflection."
        )
        user = json.dumps([item.model_dump(mode="json") for item in attempts])
        payload = await self._json_completion(system, user)
        return str(payload.get("reflection", "No reflection returned."))
