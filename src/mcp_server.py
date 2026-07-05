import sys
import os

# Ensure the root directory is in sys.path so 'src' imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from src.database import Database
from src.context import ConversationContext
from src.tools import NoteTools
from src.tool_schemas import (
    CreateNoteInput, SearchNotesInput, GetNoteInput, 
    UpdateNoteInput, DeleteNoteInput, ListNotesInput
)
from src.embeddings import SemanticSearch

# Initialize components
load_dotenv()
db_path = os.getenv("DB_PATH", "./notes.db")
db = Database(db_path)

# Auth strategy stub: In a real app, FastMCP would extract the user from a Bearer token in the request header.
# For this MCP stub, we use a single active user.
user_id = os.getenv("DEFAULT_USER_ID", "default")
context = ConversationContext(active_user_id=user_id)
tools = NoteTools(db, context)

# Add embeddings support
try:
    semantic_search = SemanticSearch(db)
    tools.embedding_func = semantic_search.embed_text
    # Add a semantic_search wrapper to tools
    tools.semantic_search = semantic_search.search
except Exception:
    # Optional dependency might fail if sentence-transformers not installed
    pass

mcp = FastMCP("notes-agent")

@mcp.tool()
def create_note(title: str, body: str, tags: list[str] = None, category: str = None) -> dict:
    """Create a new note."""
    if tags is None: tags = []
    res = tools.create_note(CreateNoteInput(title=title, body=body, tags=tags, category=category))
    return {"success": res.success, "data": res.data, "error": res.error}

@mcp.tool()
def search_notes(query: str = None, tags: list[str] = None, semantic: bool = False, limit: int = 10) -> dict:
    """Search for notes by keyword, tags, or semantic similarity."""
    if tags is None: tags = []
    
    if semantic and hasattr(tools, 'semantic_search') and query:
        # Override with semantic search
        try:
            results = tools.semantic_search(query, context.active_user_id, limit)
            summaries = [
                {"id": r.id, "title": r.title, "tags": r.tags, "created_at": r.created_at.isoformat()}
                for r in results
            ]
            return {"success": True, "data": summaries}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    res = tools.search_notes(SearchNotesInput(query=query, tags=tags, semantic=semantic, limit=limit))
    return {"success": res.success, "data": res.data, "error": res.error}

@mcp.tool()
def get_note(note_id: str) -> dict:
    """Retrieve a specific note by ID."""
    res = tools.get_note(GetNoteInput(note_id=note_id))
    return {"success": res.success, "data": res.data, "error": res.error}

@mcp.tool()
def update_note(note_id: str, title: str = None, body: str = None, tags: list[str] = None, category: str = None, confirmation_token: str = None) -> dict:
    """Update an existing note (major updates of >80% body change will require two calls: first gets token, second confirms)."""
    res = tools.update_note(UpdateNoteInput(note_id=note_id, title=title, body=body, tags=tags, category=category, confirmation_token=confirmation_token))
    return {"success": res.success, "data": res.data, "error": res.error, "requires_confirmation": res.requires_confirmation, "confirmation_token": res.confirmation_token}

@mcp.tool()
def delete_note(note_id: str, confirmation_token: str = None) -> dict:
    """Delete a note (requires two calls: first gets token, second confirms)."""
    res = tools.delete_note(DeleteNoteInput(note_id=note_id, confirmation_token=confirmation_token))
    return {"success": res.success, "data": res.data, "error": res.error, "requires_confirmation": res.requires_confirmation, "confirmation_token": res.confirmation_token}

@mcp.tool()
def list_notes(tags: list[str] = None, category: str = None, limit: int = 20) -> dict:
    """List notes with optional tag/category filters."""
    if tags is None:
        tags = []
    res = tools.list_notes(ListNotesInput(tags=tags, category=category, limit=limit))
    return {"success": res.success, "data": res.data, "error": res.error}

@mcp.tool()
def answer_question(question: str, relevant_note_ids: list[str] = None) -> dict:
    """Fetch note context to answer a natural-language question."""
    if relevant_note_ids is None:
        relevant_note_ids = []
    from src.tool_schemas import AnswerQuestionInput
    res = tools.answer_question(AnswerQuestionInput(question=question, relevant_note_ids=relevant_note_ids))
    return {"success": res.success, "data": res.data, "error": res.error}

if __name__ == "__main__":
    mcp.run()
