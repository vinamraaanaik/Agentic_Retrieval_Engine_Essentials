from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .orchestrator import StructuredAgent


class OpenAIStructuredAgent(StructuredAgent):
    """Optional OpenAI adapter for planning, reflection and grounded generation."""

    def __init__(self, api_key: str, model: str = "gpt-5", timeout: float = 30.0) -> None:
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key, timeout=timeout)

    @staticmethod
    def _parse_json(text: str) -> Mapping[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Model did not return a JSON object.")
        return json.loads(match.group(0))

    def plan(self, prompt: str) -> Mapping[str, Any]:
        response = self.client.responses.create(model=self.model, input=prompt)
        return self._parse_json(response.output_text or "")

    def reflect(self, prompt: str) -> Mapping[str, Any]:
        response = self.client.responses.create(model=self.model, input=prompt)
        return self._parse_json(response.output_text or "")

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(model=self.model, input=prompt)
        return (response.output_text or "").strip()
