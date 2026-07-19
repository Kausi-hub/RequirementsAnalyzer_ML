"""Conflict analysis module for requirement contradictions and violations."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from .graph_rag import GraphRAGEngine


class ConflictAnalyzer:
    """Runs contextual conflict checks using GraphRAG + LLM reasoning."""

    def __init__(self, graph_engine: GraphRAGEngine, api_key: str, model: str) -> None:
        self.graph_engine = graph_engine
        self.client = OpenAI(api_key=api_key, timeout=30.0)
        self.model = model

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _heuristic_conflicts(new_requirement: str, context_records: List[Dict[str, str]]) -> List[Dict[str, str]]:
        conflicts: List[Dict[str, str]] = []
        negation_tokens = {"not", "never", "no", "cannot", "must not", "shall not"}

        new_lower = new_requirement.lower()
        new_has_negation = any(token in new_lower for token in negation_tokens)

        for record in context_records:
            existing_text = record.get("text", "")
            existing_lower = existing_text.lower()
            existing_has_negation = any(token in existing_lower for token in negation_tokens)

            shared_terms = set(re.findall(r"[a-zA-Z]{4,}", new_lower)) & set(
                re.findall(r"[a-zA-Z]{4,}", existing_lower)
            )

            if len(shared_terms) < 3:
                continue

            if new_has_negation != existing_has_negation:
                conflicts.append(
                    {
                        "conflicting_requirement_id": record.get("req_id", "unknown"),
                        "severity": "medium",
                        "reason": "Potential opposite intent detected over shared domain terms.",
                        "citation": existing_text[:180],
                    }
                )

        return conflicts

    def analyze(self, new_requirement: str, conflict_system_prompt: str) -> Dict[str, Any]:
        retrieved = self.graph_engine.retrieve(new_requirement, top_k=8)
        context_records = [
            {
                "req_id": rec.req_id,
                "source": rec.source,
                "text": rec.text,
            }
            for rec in retrieved["records"]
        ]
        formatted_system_prompt = (
            conflict_system_prompt.replace(
                "{{existing_context}}", json.dumps(context_records, indent=2)
            ).replace("{{new_requirement}}", new_requirement)
        )

        user_prompt = (
            "Analyze the new requirement for conflicts with the existing pool.\n"
            "Use the provided existing context only.\n\n"
            f"Existing context:\n{json.dumps(context_records, indent=2)}\n\n"
            f"New requirement:\n{new_requirement}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": formatted_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            ai_payload = self._extract_json_object(response.choices[0].message.content or "{}")
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            ai_payload = {
                "conflicts": [],
                "summary": f"Conflict model unavailable: {exc}",
            }

        ai_conflicts = ai_payload.get("conflicts", []) if isinstance(ai_payload, dict) else []
        heuristic = self._heuristic_conflicts(new_requirement, context_records)

        merged = ai_conflicts + [
            item for item in heuristic if item.get("conflicting_requirement_id") not in {
                c.get("conflicting_requirement_id") for c in ai_conflicts if isinstance(c, dict)
            }
        ]

        return {
            "conflicts": merged,
            "summary": ai_payload.get("summary", "Conflict analysis completed.")
            if isinstance(ai_payload, dict)
            else "Conflict analysis completed.",
            "retrieved_context": context_records,
        }
