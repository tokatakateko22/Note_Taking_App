import uuid
import time
import hashlib
from datetime import datetime
from typing import Any

from src.models import Note, ToolResult
from src.database import Database
from src.context import ConversationContext, PendingAction
from src.text_match import tags_match, category_matches
from src.tool_schemas import (
    CreateNoteInput, SearchNotesInput, GetNoteInput, 
    UpdateNoteInput, DeleteNoteInput, ListNotesInput,
    AnswerQuestionInput
)

def _generate_token(action: str, note_id: str) -> str:
    # Simple token generator for confirmation
    raw = f"{action}-{note_id}-{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

class NoteTools:
    def __init__(self, db: Database, context: ConversationContext):
        self.db = db
        self.context = context
        # In a real app we might pass in the embedding function here if we don't want it tight-coupled
        self.embedding_func = None

    def create_note(self, params: CreateNoteInput) -> ToolResult:
        try:
            note_id = str(uuid.uuid4())
            now = datetime.now()
            note = Note(
                id=note_id,
                title=params.title,
                body=params.body,
                tags=params.tags,
                category=params.category,
                created_at=now,
                updated_at=now,
                user_id=self.context.active_user_id
            )
            
            if self.embedding_func:
                note.embedding = self.embedding_func(f"{note.title}\n{note.body}")
                
            created_note = self.db.insert(note)
            self.context.last_note_id = created_note.id
            
            # Serialize for LLM return
            return ToolResult(success=True, data=self._serialize_note(created_note))
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def _semantic_search(self, query: str, limit: int) -> list[Note]:
        if not hasattr(self, "_semantic_search_obj"):
            from src.embeddings import SemanticSearch
            self._semantic_search_obj = SemanticSearch(self.db)
        return self._semantic_search_obj.search(query, self.context.active_user_id, limit)

    def _search_by_query(self, query: str, limit: int, semantic: bool) -> list[Note]:
        if semantic and self.embedding_func:
            return self._semantic_search(query, limit)

        results = self.db.fts_search(query, self.context.active_user_id, limit)

        # Semantic fallback when keyword search finds nothing (handles casing & related concepts)
        if not results and self.embedding_func and query.strip():
            results = self._semantic_search(query, limit)

        return results

    def search_notes(self, params: SearchNotesInput) -> ToolResult:
        try:
            results = []
            if params.query:
                results = self._search_by_query(params.query, params.limit, params.semantic)
            elif params.tags:
                results = self.db.filter_by_tags(params.tags, self.context.active_user_id)
            elif params.date_from or params.date_to:
                results = self.db.filter_by_date_range(params.date_from, params.date_to, self.context.active_user_id)
            else:
                results = self.db.get_all(self.context.active_user_id, params.limit)
            
            # Filter results if multiple parameters were provided (e.g. query AND tags)
            if params.query and params.tags:
                results = [r for r in results if tags_match(r.tags, params.tags)]

            if params.category:
                results = [r for r in results if category_matches(r.category, params.category)]
                
            self.context.last_search_results = [r.id for r in results]
            
            summaries = [
                {"id": r.id, "title": r.title, "tags": r.tags, "created_at": r.created_at.isoformat()}
                for r in results
            ]
            
            # If exactly one result, maybe update context?
            if len(results) == 1:
                self.context.last_note_id = results[0].id
                
            return ToolResult(success=True, data=summaries)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def get_note(self, params: GetNoteInput) -> ToolResult:
        try:
            note = self.db.get_by_id(params.note_id, self.context.active_user_id)
            if not note:
                return ToolResult(success=False, data=None, error="Note not found")
            self.context.last_note_id = note.id
            return ToolResult(success=True, data=self._serialize_note(note))
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def update_note(self, params: UpdateNoteInput) -> ToolResult:
        try:
            existing_note = self.db.get_by_id(params.note_id, self.context.active_user_id)
            if not existing_note:
                return ToolResult(success=False, data=None, error="Note not found")

            # Check if this is a major body change that requires confirmation
            if params.body is not None:
                import difflib
                import time
                from src.context import PendingAction
                ratio = difflib.SequenceMatcher(None, existing_note.body, params.body).ratio()
                is_major_change = (ratio < 0.2) # >80% change
                
                if is_major_change:
                    if params.confirmation_token:
                        pending = self.context.pending_confirmation
                        if pending and pending.action == 'update' and pending.note_id == params.note_id:
                            if pending.token == params.confirmation_token:
                                if time.time() > pending.expires_at:
                                    self.context.pending_confirmation = None
                                    return ToolResult(success=False, data=None, error="Confirmation token expired")
                                # Valid token — restore stored payload and proceed
                                stored = pending.payload or {}
                                self.context.pending_confirmation = None
                                if params.title is None and stored.get('title') is not None:
                                    params = params.model_copy(update={'title': stored['title']})
                                if params.body is None and stored.get('body') is not None:
                                    params = params.model_copy(update={'body': stored['body']})
                                if params.tags is None and stored.get('tags') is not None:
                                    params = params.model_copy(update={'tags': stored['tags']})
                                if params.category is None and stored.get('category') is not None:
                                    params = params.model_copy(update={'category': stored['category']})
                            else:
                                return ToolResult(success=False, data=None, error="Invalid confirmation token")
                        else:
                            return ToolResult(success=False, data=None, error="No pending confirmation matches this token")
                    else:
                        # First call, generate token and store pending changes
                        token = _generate_token('update', params.note_id)
                        self.context.pending_confirmation = PendingAction(
                            action='update',
                            token=token,
                            note_id=params.note_id,
                            payload={
                                'title': params.title,
                                'body': params.body,
                                'tags': params.tags,
                                'category': params.category,
                            },
                            expires_at=time.time() + 60.0
                        )
                        return ToolResult(
                            success=True,
                            data={"message": "Updating this note changes more than 80% of its body. Are you sure you want to replace it?"},
                            requires_confirmation=True,
                            confirmation_token=token
                        )

            patches = {}
            if params.title is not None: patches['title'] = params.title
            if params.body is not None: patches['body'] = params.body
            if params.tags is not None: patches['tags'] = params.tags
            if params.category is not None: patches['category'] = params.category

            if self.embedding_func and params.body is not None:
                new_title = params.title if params.title is not None else existing_note.title
                patches['embedding'] = self.embedding_func(f"{new_title}\n{params.body}")

            updated_note = self.db.update(params.note_id, patches, self.context.active_user_id)
            self.context.last_note_id = updated_note.id
            return ToolResult(success=True, data=self._serialize_note(updated_note))
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def delete_note(self, params: DeleteNoteInput) -> ToolResult:
        try:
            # Check if note exists
            existing_note = self.db.get_by_id(params.note_id, self.context.active_user_id)
            if not existing_note:
                return ToolResult(success=False, data=None, error="Note not found")

            # Phase 2: verify token
            if params.confirmation_token:
                pending = self.context.pending_confirmation
                if pending and pending.action == 'delete' and pending.note_id == params.note_id:
                    if pending.token == params.confirmation_token:
                        if time.time() > pending.expires_at:
                            self.context.pending_confirmation = None
                            return ToolResult(success=False, data=None, error="Confirmation token expired")
                        
                        # Valid token, proceed
                        self.db.delete(params.note_id, self.context.active_user_id)
                        self.context.pending_confirmation = None
                        if self.context.last_note_id == params.note_id:
                            self.context.last_note_id = None
                        return ToolResult(success=True, data={"deleted": True, "note_id": params.note_id})
                    else:
                        return ToolResult(success=False, data=None, error="Invalid confirmation token")
                else:
                    return ToolResult(success=False, data=None, error="No pending confirmation matches this token")

            # Phase 1: generate token
            token = _generate_token('delete', params.note_id)
            self.context.pending_confirmation = PendingAction(
                action='delete',
                token=token,
                note_id=params.note_id,
                expires_at=time.time() + 60.0
            )
            return ToolResult(
                success=True, 
                data={"message": f"Are you sure you want to delete '{existing_note.title}'?"},
                requires_confirmation=True,
                confirmation_token=token
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def list_notes(self, params: ListNotesInput) -> ToolResult:
        try:
            results = self.db.get_all(self.context.active_user_id, params.limit)
            if params.tags:
                results = [r for r in results if tags_match(r.tags, params.tags)]
            if params.category:
                results = [r for r in results if category_matches(r.category, params.category)]
                
            # Basic sort (SQLite already sorted by updated_at desc)
            if params.sort_by == "created_at":
                results.sort(key=lambda x: x.created_at, reverse=(params.sort_order=="desc"))
            elif params.sort_by == "title":
                results.sort(key=lambda x: x.title, reverse=(params.sort_order=="desc"))
            elif params.sort_by == "updated_at" and params.sort_order == "asc":
                results.sort(key=lambda x: x.updated_at, reverse=False)

            summaries = [
                {"id": r.id, "title": r.title, "tags": r.tags, "created_at": r.created_at.isoformat()}
                for r in results
            ]
            return ToolResult(success=True, data=summaries)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def answer_question(self, params: AnswerQuestionInput) -> ToolResult:
        try:
            # We fetch notes and return them. The outer LLM agent synthesizes the response,
            # unless we instantiate a separate LLM call here. 
            # The prompt says: "Returns: synthesised answer string (LLM call with note context injected)"
            # Since LLM client is built in Phase 2, we return a placeholder here, which will be updated.
            notes = []
            for nid in params.relevant_note_ids:
                n = self.db.get_by_id(nid, self.context.active_user_id)
                if n: notes.append(n)
            
            context_text = "\n\n".join([f"Note: {n.title}\n{n.body}" for n in notes])
            
            return ToolResult(
                success=True, 
                data={"context": context_text, "message": "Synthesized answer pending LLM integration."}
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def _serialize_note(self, note: Note) -> dict:
        return {
            "id": note.id,
            "title": note.title,
            "body": note.body,
            "tags": note.tags,
            "category": note.category,
            "created_at": note.created_at.isoformat(),
            "updated_at": note.updated_at.isoformat()
        }
