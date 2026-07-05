import json
import re
from typing import Any


def parse_tool_arguments(raw: Any) -> dict:
    """Normalize provider tool arguments to a dict (Groq may return null/empty)."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped or stripped in ("null", "{}"):
            return {}
        return json.loads(stripped)
    raise TypeError(f"Unsupported tool arguments type: {type(raw).__name__}")


def normalize_tool_call(obj: dict, index: int = 0) -> dict | None:
    name = obj.get("name") or obj.get("function")
    if not name:
        return None

    args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    if not isinstance(args, dict):
        return None

    return {
        "id": obj.get("id") or f"call_{index}",
        "name": name,
        "arguments": args,
    }


def extract_tool_calls_from_text(text: str) -> list[dict] | None:
    """Recover tool calls when the model prints JSON in plain text instead of tool_calls."""
    if not text or not text.strip():
        return None

    calls: list[dict] = []
    seen: set[str] = set()

    def add_call(obj: dict) -> None:
        normalized = normalize_tool_call(obj, len(calls))
        if not normalized:
            return
        key = json.dumps({"name": normalized["name"], "arguments": normalized["arguments"]}, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        calls.append(normalized)

    stripped = text.strip()
    try:
        whole = json.loads(stripped)
        if isinstance(whole, dict):
            add_call(whole)
        elif isinstance(whole, list):
            for item in whole:
                if isinstance(item, dict):
                    add_call(item)
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"[^{}]*\}", text, re.DOTALL):
        try:
            add_call(json.loads(match.group()))
        except json.JSONDecodeError:
            continue

    # Brace-scan for nested JSON objects (e.g. parameters with nested dicts)
    for idx, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        for end in range(idx, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[idx : end + 1]
                    try:
                        obj = json.loads(snippet)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and obj.get("name"):
                        add_call(obj)
                    break

    return calls or None


def _find_tool_json_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for idx, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        for end in range(idx, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[idx : end + 1]
                    try:
                        obj = json.loads(snippet)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and obj.get("name"):
                        spans.append((idx, end + 1))
                    break
    return spans


def strip_tool_json_from_text(text: str) -> str | None:
    """Remove embedded tool-call JSON from assistant text shown to the user."""
    if not text:
        return None

    cleaned = text
    for start, end in reversed(_find_tool_json_spans(text)):
        cleaned = cleaned[:start] + cleaned[end:]

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or None
