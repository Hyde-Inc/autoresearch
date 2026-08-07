from __future__ import annotations

import asyncio
import json
import os
import re

from openai import OpenAI

from .config import TaskConfig
from .models import Attempt, Idea

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)


def _extract_json(content: str) -> dict:
    """Parse JSON, tolerating models that wrap it in a markdown code fence."""
    match = _JSON_FENCE.match(content)
    if match:
        content = match["body"]
    return json.loads(content)


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

    async def _json_completion(self, system: str, user: str) -> dict:
        def call() -> dict:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.config.director.temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("director returned an empty response")
            return _extract_json(content)

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
        system = (
            "You are the Research Director for an autonomous ML lab. Propose diverse, testable, "
            "single-change experiments. Learn from results, avoid repeating failed ideas, and "
            "respect runtime and metric guardrails. Return strict JSON with key 'ideas', an array "
            "of objects with exactly: title, hypothesis, instructions, category."
        )
        user = (
            f"Task: {self.config.description}\nGoal: {self.config.goal}\n"
            f"Primary metric: {self.config.metric.name} ({self.config.metric.direction})\n"
            f"Guardrails: {self.config.guardrails}\nRound: {round_number}\n"
            f"Need {count} ideas.\nSuggested families (not mandatory): {self.config.idea_hints}\n"
            f"Prior attempts: {json.dumps(history)}\nLab notes:\n{notes[-8000:]}"
        )
        payload = await self._json_completion(system, user)
        ideas = [Idea.model_validate(item) for item in payload.get("ideas", [])]
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
