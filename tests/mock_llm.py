"""Deterministic mock LLM for evaluation harness (no API keys required)."""
import json
import re
from typing import Any


class MockLLMClient:
    """Rule-based LLM stub that routes common eval scenarios to the correct tools."""

    provider = "mock"

    def generate_response(
        self,
        system_prompt: str,
        messages: list[dict],
        tools_def: list[dict] | None = None,
    ) -> dict:
        user_text = self._current_user_text(messages)
        lower = user_text.lower()

        # After a tool runs, either chain another tool or finish with text
        if messages and messages[-1]["role"] == "tool":
            nxt = self._after_tool(messages, user_text, lower, system_prompt)
            if nxt:
                return nxt
            return {"text": "Done.", "tool_calls": None}

        # Confirmation follow-ups
        if lower.strip() in ("yes", "yes.", "confirm", "go ahead", "ok", "sure"):
            token, note_id, action = self._last_pending_confirmation(messages)
            if action == "delete" and note_id:
                return self._tool("delete_note", {"note_id": note_id, "confirmation_token": token})
            if action == "update" and note_id:
                return self._tool("update_note", {"note_id": note_id, "confirmation_token": token})

        if "summarise" in lower or "summarize" in lower:
            return self._tool("search_notes", {"tags": ["urgent"]})

        if "contradict" in lower:
            if self._tool_was_called(messages, "search_notes"):
                note_ids = self._note_ids_from_last_search(messages)
                return self._tool("answer_question", {"question": user_text, "relevant_note_ids": note_ids})
            return self._tool("search_notes", {"query": "API"})

        if lower.strip() == "delete it." or lower.strip() == "delete it":
            note_id = self._last_note_id(system_prompt)
            return self._tool("delete_note", {"note_id": note_id})

        if lower.startswith("delete ") or lower.startswith("delete the"):
            if "aliens" in lower:
                return self._tool("search_notes", {"query": "aliens"})
            if "api note" in lower and self._created_count_matching(messages, "api") > 1:
                return self._tool("search_notes", {"query": "API"})
            note_id = self._resolve_note_id(messages, lower)
            if note_id:
                return self._tool("delete_note", {"note_id": note_id})
            return self._tool("search_notes", {"query": self._extract_query(lower, "delete")})

        if "find notes about aliens" in lower:
            return self._tool("search_notes", {"query": "aliens"})

        if "search for" in lower:
            query = re.sub(r".*search for\s+", "", lower).strip(". ")
            return self._tool("search_notes", {"query": query})

        if "what did i write last week" in lower:
            return self._tool("search_notes", {"date_from": "2025-06-28", "date_to": "2025-07-05"})

        if "show me urgent" in lower:
            return self._tool("search_notes", {"tags": ["urgent"]})

        if "actually" in lower and "that note" in lower:
            note_id = self._last_note_id(system_prompt)
            body = self._last_created_body(messages) or "Project planning notes."
            return self._tool("update_note", {"note_id": note_id, "body": body + " Deadline: end of month."})

        if "add the deadline tag" in lower:
            note_id = self._last_note_id(system_prompt)
            return self._tool("update_note", {"note_id": note_id, "tags": ["deadline"]})

        if "update" in lower and "standup" in lower:
            note_id = self._note_id_by_title(messages, "standup") or self._last_note_id(system_prompt)
            return self._tool("update_note", {"note_id": note_id, "body": "Wednesday"})

        if "create note 1 tagged urgent" in lower:
            return self._tool("create_note", {"title": "Note 1", "body": "Urgent item 1", "tags": ["urgent"]})
        if "create note 2 tagged urgent" in lower:
            return self._tool("create_note", {"title": "Note 2", "body": "Urgent item 2", "tags": ["urgent"]})

        if "create note saying api is json" in lower:
            return self._tool("create_note", {"title": "API JSON", "body": "The API is JSON.", "tags": []})
        if "create note saying api is xml" in lower:
            return self._tool("create_note", {"title": "API XML", "body": "The API is XML.", "tags": []})

        if "titled 'api v1'" in lower or "titled \"api v1\"" in lower:
            return self._tool("create_note", {"title": "API v1", "body": "API version 1 notes", "tags": []})
        if "titled 'api v2'" in lower or "titled \"api v2\"" in lower:
            return self._tool("create_note", {"title": "API v2", "body": "API version 2 notes", "tags": []})

        if "old address" in lower:
            return self._tool("create_note", {"title": "Old address", "body": "123 Old Street", "tags": []})
        if "note to delete" in lower:
            return self._tool("create_note", {"title": "To delete", "body": "temporary", "tags": []})

        if "about apples" in lower:
            return self._tool("create_note", {"title": "Apples", "body": "Notes about apples.", "tags": []})
        if "about the project" in lower:
            return self._tool("create_note", {"title": "Project", "body": "Project planning notes.", "tags": []})
        if "about the design" in lower:
            return self._tool("create_note", {"title": "Design", "body": "Design notes.", "tags": []})
        if "groceries" in lower:
            return self._tool("create_note", {"title": "Groceries", "body": user_text, "tags": ["urgent"]})
        if "tagged urgent" in lower and "create" in lower:
            return self._tool("create_note", {"title": "Urgent note", "body": user_text, "tags": ["urgent"]})

        if "team standup" in lower or ("standup" in lower and "tuesday" in lower):
            title = "Team Standup" if "team standup" in lower else "Standup"
            body = "Move to Tuesdays" if "team standup" in lower else "Tuesday"
            m = re.search(r"body\s+'([^']+)'", user_text, re.I)
            if m:
                body = m.group(1)
            m = re.search(r"titled\s+'([^']+)'", user_text, re.I)
            if m:
                title = m.group(1)
            return self._tool("create_note", {"title": title, "body": body, "tags": []})

        return {"text": "I could not determine the intent.", "tool_calls": None}

    def _after_tool(self, messages, user_text, lower, system_prompt) -> dict | None:
        last_tool = messages[-1]
        name = last_tool.get("name")

        if name == "search_notes" and ("summar" in lower):
            note_ids = self._note_ids_from_last_search(messages)
            return self._tool("answer_question", {"question": user_text, "relevant_note_ids": note_ids})

        if name == "search_notes" and "contradict" in lower:
            note_ids = self._note_ids_from_last_search(messages)
            return self._tool("answer_question", {"question": user_text, "relevant_note_ids": note_ids})

        if name in ("delete_note", "update_note"):
            try:
                data = json.loads(last_tool["content"])
                if data.get("status") == "requires_confirmation":
                    return None  # finish turn with confirmation prompt text
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _tool(self, name: str, arguments: dict) -> dict:
        return {"text": None, "tool_calls": [{"id": f"mock_{name}", "name": name, "arguments": arguments}]}

    def _current_user_text(self, messages: list[dict]) -> str:
        return next(m["content"] for m in reversed(messages) if m["role"] == "user")

    def _last_note_id(self, system_prompt: str) -> str:
        m = re.search(r"last_referenced_note_id=([^\n]+)", system_prompt)
        if m and m.group(1).strip() not in ("None", "none"):
            return m.group(1).strip()
        return "unknown"

    def _last_created_body(self, messages: list[dict]) -> str | None:
        for msg in reversed(messages):
            if msg["role"] == "tool" and msg.get("name") == "create_note":
                try:
                    return json.loads(msg["content"]).get("body")
                except (json.JSONDecodeError, TypeError):
                    pass
        return None

    def _note_ids_from_last_search(self, messages: list[dict]) -> list[str]:
        for msg in reversed(messages):
            if msg["role"] == "tool" and msg.get("name") == "search_notes":
                try:
                    data = json.loads(msg["content"])
                    if isinstance(data, list):
                        return [n["id"] for n in data]
                except (json.JSONDecodeError, TypeError):
                    pass
        return []

    def _tool_was_called(self, messages: list[dict], tool_name: str) -> bool:
        return any(m["role"] == "tool" and m.get("name") == tool_name for m in messages)

    def _created_count_matching(self, messages: list[dict], keyword: str) -> int:
        count = 0
        for msg in messages:
            if msg["role"] == "tool" and msg.get("name") == "create_note":
                try:
                    if keyword in json.loads(msg["content"]).get("title", "").lower():
                        count += 1
                except (json.JSONDecodeError, TypeError):
                    pass
        return count

    def _note_id_by_title(self, messages: list[dict], title_sub: str) -> str | None:
        for msg in reversed(messages):
            if msg["role"] == "tool" and msg.get("name") == "create_note":
                try:
                    data = json.loads(msg["content"])
                    if title_sub in data.get("title", "").lower():
                        return data["id"]
                except (json.JSONDecodeError, TypeError):
                    pass
        return None

    def _resolve_note_id(self, messages: list[dict], lower: str) -> str | None:
        if "old address" in lower:
            return self._note_id_by_title(messages, "old address")
        if "standup" in lower:
            return self._note_id_by_title(messages, "standup")
        return None

    def _last_pending_confirmation(self, messages: list[dict]) -> tuple[str | None, str | None, str | None]:
        for msg in reversed(messages):
            if msg["role"] == "tool" and msg.get("name") in ("delete_note", "update_note"):
                try:
                    data = json.loads(msg["content"])
                    if data.get("status") == "requires_confirmation":
                        token = data.get("confirmation_token")
                        for prev in reversed(messages):
                            if prev["role"] == "assistant" and prev.get("tool_calls"):
                                for tc in prev["tool_calls"]:
                                    if tc["name"] in ("delete_note", "update_note"):
                                        action = "delete" if tc["name"] == "delete_note" else "update"
                                        return token, tc["arguments"].get("note_id"), action
                except (json.JSONDecodeError, TypeError):
                    pass
        return None, None, None

    def _extract_query(self, lower: str, verb: str) -> str:
        return re.sub(rf".*{verb}\s+(?:the\s+)?(?:note\s+about\s+)?", "", lower).strip(". ")
