"""Final-answer safety checks shared by agents and validators."""

from __future__ import annotations

import re
from typing import Iterable


_PSEUDO_TOOL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "deepseek_tool_calls_begin",
        re.compile(r"<\s*[|｜]\s*tool[_▁\s-]*calls[_▁\s-]*begin\s*[|｜]\s*>", re.I),
    ),
    (
        "deepseek_tool_calls_end",
        re.compile(r"<\s*[|｜]\s*tool[_▁\s-]*calls[_▁\s-]*end\s*[|｜]\s*>", re.I),
    ),
    (
        "dsml_tool_call",
        re.compile(
            r"<\s*(?:[|｜]\s*){1,2}dsml(?:\s*[|｜]){1,2}\s*"
            r"(?:tool[_▁\s-]*calls|invoke|parameter)\b",
            re.I,
        ),
    ),
    (
        "xml_tool_call",
        re.compile(r"<\s*/?\s*(?:tool_call|tool_calls|invoke)\b", re.I),
    ),
    ("tool_calls_key", re.compile(r"\btool[_▁\s-]*calls\b", re.I)),
    ("function_call_key", re.compile(r"\bfunction[_▁\s-]*call\b", re.I)),
    ("call_sub_agent_text", re.compile(r"\bcall[_\s-]*sub[_\s-]*agent\b", re.I)),
    ("action_observation", re.compile(r"^\s*(?:action|observation)\s*:", re.I | re.M)),
    (
        "json_name_arguments",
        re.compile(
            r"\{\s*[\"']name[\"']\s*:\s*.+?[\"']arguments[\"']\s*:",
            re.I | re.S,
        ),
    ),
)


def _normalization_variants(text: str) -> Iterable[str]:
    """Yield raw and lightly unescaped variants for robust pattern matching."""
    yield text
    unescaped = (
        text.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
        .replace("\\uFF5C", "｜")
        .replace("\\uff5c", "｜")
        .replace("\\u2581", "▁")
        .replace("\\u003c", "<")
        .replace("\\u003e", ">")
    )
    if unescaped != text:
        yield unescaped


def pseudo_tool_call_reasons(text: object) -> list[str]:
    """Return matching pseudo-tool-call reason names for final-answer text."""
    if not isinstance(text, str) or not text:
        return []
    reasons: list[str] = []
    for variant in _normalization_variants(text):
        for name, pattern in _PSEUDO_TOOL_PATTERNS:
            if name not in reasons and pattern.search(variant):
                reasons.append(name)
    return reasons


def contains_pseudo_tool_call(text: object) -> bool:
    return bool(pseudo_tool_call_reasons(text))


def extract_answer_text(text: object) -> str:
    """Extract an <answer> block when present, otherwise return stripped text."""
    if not isinstance(text, str):
        return ""
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return text.strip()


def safe_fallback_prediction(reason: str = "") -> str:
    suffix = f" ({reason})" if reason else ""
    return f"[Failed: invalid tool-call-like final answer suppressed{suffix}]"


def is_suppressed_prediction(text: object) -> bool:
    if not isinstance(text, str):
        return False
    return text.strip().startswith(
        "[Failed: invalid tool-call-like final answer suppressed"
    )


def is_failed_placeholder_prediction(text: object) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    return (
        is_suppressed_prediction(stripped)
        or stripped in {"[Failed]", "[Failed.]"}
        or stripped.startswith("[Failed:")
    )


def is_usable_prediction(text: object) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    stripped = text.strip()
    return (
        not contains_pseudo_tool_call(stripped)
        and not stripped.lower().startswith("no answer found")
        and not is_failed_placeholder_prediction(stripped)
    )
