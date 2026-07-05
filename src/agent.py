import json
from datetime import datetime
from pydantic import ValidationError

from src.llm_client import LLMClient
from src.tools import NoteTools
from src.context import ConversationContext
from src.tool_parse import extract_tool_calls_from_text, strip_tool_json_from_text
from src.tool_schemas import (
    CreateNoteInput, SearchNotesInput, GetNoteInput, 
    UpdateNoteInput, DeleteNoteInput, ListNotesInput,
    AnswerQuestionInput
)

SYSTEM_PROMPT_TEMPLATE = """You are a personal note-taking assistant. Rules:
1. If a search/list returns >1 note and the user's intent targets a specific one, LIST the matches and ASK which one they mean. Do NOT guess and do NOT execute updates/deletes until the user clarifies.
2. Destructive actions use a two-phase confirmation flow via tools:
   - delete_note / update_note (major body changes) return {{"status": "requires_confirmation", "confirmation_token": "..."}} on the first call.
   - When the user confirms ("yes", "confirm", "go ahead"), call the SAME tool again with the SAME note_id AND the confirmation_token from the tool result.
   - Never delete or replace a note without this second confirmed call.
3. For create_note, search_notes, list_notes, get_note, and answer_question, act immediately without asking for confirmation. Always invoke tools via the tool-calling API — never print raw JSON tool payloads in your reply.
4. For list_notes, omit limit to use the default (20) or pass an integer between 1 and 100. Never pass strings like "all".
5. If the user refers to a note by keyword/title (e.g. "the Standup note"), call search_notes first to check for multiple matches. Use last_referenced_note_id only when the user says "that note", "the last one", "it", or similar pronouns.
6. If a search returns 0 results, retry with semantic=true or suggest broadening the query. Tag and keyword matching is case-insensitive (e.g. 'Urgent' matches 'urgent').
7. For summarise/compare/contradiction questions, first search_notes or list_notes to find relevant notes, then call answer_question with their IDs.
8. Today's date is {today}. Use this for relative date parsing ("last week").

Current context:
last_referenced_note_id={last_note_id}
"""

class Agent:
    def __init__(self, llm_client: LLMClient, note_tools: NoteTools):
        self.llm = llm_client
        self.tools = note_tools
        self.history = []
        
        self.tools_def = [
            {"name": "create_note", "description": "Create a new note.", "schema_class": CreateNoteInput},
            {"name": "search_notes", "description": "Search for notes.", "schema_class": SearchNotesInput},
            {"name": "get_note", "description": "Get a specific note by ID.", "schema_class": GetNoteInput},
            {"name": "update_note", "description": "Update an existing note.", "schema_class": UpdateNoteInput},
            {"name": "delete_note", "description": "Delete a note.", "schema_class": DeleteNoteInput},
            {"name": "list_notes", "description": "List notes.", "schema_class": ListNotesInput},
            {"name": "answer_question", "description": "Answer questions based on notes.", "schema_class": AnswerQuestionInput},
        ]

    def _get_system_prompt(self):
        ctx = self.tools.context
        return SYSTEM_PROMPT_TEMPLATE.format(
            today=datetime.now().strftime("%Y-%m-%d"),
            last_note_id=ctx.last_note_id or "None"
        )

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        
        max_iterations = 5
        for _ in range(max_iterations):
            response = self.llm.generate_response(
                system_prompt=self._get_system_prompt(),
                messages=self.history,
                tools_def=self.tools_def
            )

            if not response.get("tool_calls") and response.get("text"):
                recovered = extract_tool_calls_from_text(response["text"])
                if recovered:
                    response["tool_calls"] = recovered
                    response["text"] = strip_tool_json_from_text(response["text"])

            if not response.get("tool_calls"):
                # No more tools to call, return the final text
                if response.get("text"):
                    self.history.append({"role": "assistant", "content": response["text"]})
                return response.get("text", "I'm sorry, I couldn't process that.")

            # Record the tool calls in history (ensuring strict alternating user/assistant roles)
            self.history.append({
                "role": "assistant",
                "content": response.get("text", "") or "",
                "tool_calls": response["tool_calls"]
            })

            # Execute tool calls
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["name"]
                args = tool_call.get("arguments")
                if args is None:
                    args = {}
                
                tool_result_content = ""
                
                try:
                    # Find schema class
                    schema_class = next(t["schema_class"] for t in self.tools_def if t["name"] == tool_name)
                    parsed_args = schema_class.model_validate(args)
                    
                    # Dispatch to tool function
                    func = getattr(self.tools, tool_name)
                    result = func(parsed_args)
                    
                    if result.success:
                        if result.requires_confirmation:
                            tool_result_content = json.dumps({
                                "status": "requires_confirmation",
                                "message": result.data.get("message", "Confirmation needed."),
                                "confirmation_token": result.confirmation_token
                            })
                        else:
                            tool_result_content = json.dumps(result.data, default=str)
                    else:
                        tool_result_content = json.dumps({"error": result.error})
                except ValidationError as e:
                    tool_result_content = json.dumps({"error": f"Validation Error: {str(e)}"})
                except Exception as e:
                    tool_result_content = json.dumps({"error": f"System Error: {str(e)}"})
                
                self.history.append({
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": tool_call["id"],
                    "content": tool_result_content
                })

        return "I've reached my thinking limit for this turn. Please try again."
