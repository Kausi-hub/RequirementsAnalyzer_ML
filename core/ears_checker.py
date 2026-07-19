"""EARS template validation module."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI


class EARSChecker:
    """Validates requirements against EARS patterns and suggests rewrites."""

    PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
        (
            "Ubiquitous",
            re.compile(r"^The\s+.+\s+shall\s+.+$", re.IGNORECASE),
        ),
        (
            "Event-driven",
            re.compile(r"^WHEN\s+.+\s+The\s+.+\s+shall\s+.+$", re.IGNORECASE),
        ),
        (
            "State-driven",
            re.compile(r"^WHILE\s+.+\s+The\s+.+\s+shall\s+.+$", re.IGNORECASE),
        ),
        (
            "Unwanted behavior",
            re.compile(r"^IF\s+.+\s+THEN\s+The\s+.+\s+shall\s+.+$", re.IGNORECASE),
        ),
        (
            "Optional feature",
            re.compile(r"^WHERE\s+.+\s+The\s+.+\s+shall\s+.+$", re.IGNORECASE),
        ),
    ]

    def __init__(self, api_key: str, model: str) -> None:
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

    def _regex_validate(self, requirement_text: str) -> Dict[str, Any]:
        trimmed = requirement_text.strip()
        for pattern_name, pattern in self.PATTERNS:
            if pattern.match(trimmed):
                return {
                    "is_valid": True,
                    "matched_pattern": pattern_name,
                    "issues": [],
                    "suggestion": trimmed,
                }

        issues = [
            "Requirement does not match any canonical EARS sentence shape.",
            "Expected one of: Ubiquitous, Event-driven, State-driven, Unwanted behavior, Optional feature.",
        ]
        return {
            "is_valid": False,
            "matched_pattern": "None",
            "issues": issues,
            "suggestion": "",
        }

    def validate(self, requirement_text: str, ears_system_prompt: str) -> Dict[str, Any]:
        regex_result = self._regex_validate(requirement_text)
        formatted_system_prompt = ears_system_prompt.replace(
            "{{new_requirement}}", requirement_text
        )

        user_prompt = (
            "Validate this requirement against EARS and return strict JSON.\n\n"
            f"Requirement:\n{requirement_text}"
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
            ai_result = self._extract_json_object(response.choices[0].message.content or "{}")
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            ai_result = {
                "is_valid": regex_result["is_valid"],
                "matched_pattern": regex_result["matched_pattern"],
                "issues": regex_result["issues"] + [f"AI validation unavailable: {exc}"],
                "suggestion": regex_result["suggestion"],
            }

        if not isinstance(ai_result, dict):
            return regex_result

        is_valid = bool(ai_result.get("is_valid", regex_result["is_valid"]))
        matched_pattern = ai_result.get("matched_pattern", regex_result["matched_pattern"])
        issues = ai_result.get("issues", regex_result["issues"])
        suggestion = ai_result.get("suggestion", regex_result["suggestion"])

        if is_valid and not suggestion:
            suggestion = requirement_text.strip()

        return {
            "is_valid": is_valid,
            "matched_pattern": matched_pattern,
            "issues": issues,
            "suggestion": suggestion,
            "regex_check": regex_result,
        }
