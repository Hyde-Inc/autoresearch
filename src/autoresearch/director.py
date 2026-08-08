from __future__ import annotations

import asyncio
import json
import os
import re

from openai import OpenAI

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


class ResearchDirector:
    def __init__(self, config: TaskConfig):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/Hyde-Inc/autoresearch",
                "X-Title": "Parallel Autoresearch",
            },
        )
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

    async def propose(
        self,
        *,
        count: int,
        round_number: int,
        attempts: list[Attempt],
        notes: str,
    ) -> list[Idea]:
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
