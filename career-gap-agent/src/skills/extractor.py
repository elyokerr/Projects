from __future__ import annotations

import json
import re
from collections.abc import Callable

from src.generation.prompts import SKILL_EXTRACTION_PROMPT

LlmFn = Callable[[str], str]


def _strip_fence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    return re.sub(r"```$", "", s).strip()


def extract_skill_phrases(text: str, llm: LlmFn) -> list[str]:
    raw = llm(SKILL_EXTRACTION_PROMPT.format(text=text[:6000]))
    try:
        items = json.loads(_strip_fence(raw))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(items, list):
        return []
    return [str(x).strip().lower() for x in items if str(x).strip()]
