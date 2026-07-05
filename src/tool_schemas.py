from pydantic import BaseModel, Field, model_validator
from typing import Literal

class CreateNoteInput(BaseModel):
    title: str = Field(..., description="Short descriptive title for the note")
    body: str = Field(..., description="Full text content of the note")
    tags: list[str] = Field(default=[], description="Categorical tags, e.g. ['meetings', 'urgent']")
    category: str | None = Field(None, description="Top-level category, e.g. 'work', 'personal'")

class SearchNotesInput(BaseModel):
    query: str | None = Field(None, description="Keyword or phrase to search title and body")
    tags: list[str] = Field(default=[], description="Filter by one or more tags (OR logic)")
    category: str | None = None
    date_from: str | None = Field(None, description="ISO 8601 date lower bound, e.g. '2025-01-01'")
    date_to: str | None = Field(None, description="ISO 8601 date upper bound")
    semantic: bool = Field(
        False,
        description="Use semantic similarity search. When false, keyword search still falls back to semantic if no matches are found.",
    )
    limit: int = Field(10, ge=1, le=50)

class GetNoteInput(BaseModel):
    note_id: str = Field(..., description="UUID of the note to retrieve")

class UpdateNoteInput(BaseModel):
    note_id: str
    title: str | None = None
    body: str | None = None
    tags: list[str] | None = None
    category: str | None = None
    confirmation_token: str | None = Field(None, description="Token to confirm a major note update, obtained from the first call")

    @model_validator(mode="after")
    def check_at_least_one_field(self) -> 'UpdateNoteInput':
        # If confirmation_token is provided, we don't strictly require other fields in the validator,
        # but we check if at least one field or token is provided.
        if (self.title is None and self.body is None and self.tags is None and 
            self.category is None and self.confirmation_token is None):
            raise ValueError("At least one field to update or confirmation_token must be provided")
        return self

class DeleteNoteInput(BaseModel):
    note_id: str
    confirmation_token: str | None = Field(
        None,
        description="Must be provided on second call after user confirms deletion"
    )

class ListNotesInput(BaseModel):
    tags: list[str] = []
    category: str | None = None
    limit: int = Field(20, ge=1, le=100)
    sort_by: Literal["created_at", "updated_at", "title"] = "updated_at"
    sort_order: Literal["asc", "desc"] = "desc"

class AnswerQuestionInput(BaseModel):
    question: str = Field(..., description="Natural language question to answer using note content")
    relevant_note_ids: list[str] = Field(
        default=[],
        description="Specific note IDs to reason over; if empty, agent searches first"
    )
