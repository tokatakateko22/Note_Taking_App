# Tool Schemas Reference

This document describes the 7 tool schemas exposed by the agent to perform CRUD and querying operations on notes. All schemas are backed by Pydantic models in `src/tool_schemas.py`.

## 1. `create_note`
Creates a new note with a title, body, and optional tags/category.

**Parameters**:
- `title` (str, required): Short descriptive title.
- `body` (str, required): Full text content.
- `tags` (list[str], default `[]`): Categorical tags.
- `category` (str | null, default `null`): Top-level category.

**Returns**: The created `Note` object.

## 2. `search_notes`
Retrieves notes by keyword, tags, category, date range, or semantic similarity.

**Parameters**:
- `query` (str | null, default `null`): Keyword or phrase to search.
- `tags` (list[str], default `[]`): Filter by tags (OR logic).
- `category` (str | null, default `null`): Filter by category.
- `date_from` (str | null, default `null`): ISO 8601 lower bound.
- `date_to` (str | null, default `null`): ISO 8601 upper bound.
- `semantic` (bool, default `false`): If true, uses semantic similarity search. When false, keyword search still falls back to semantic automatically if no matches are found.
- `limit` (int, default `10`): Max results to return.

Tag, category, and keyword matching are **case-insensitive** (`Urgent` matches `urgent`).

**Returns**: List of `Note` summaries.

## 3. `get_note`
Fetches a single note by its UUID.

**Parameters**:
- `note_id` (str, required): UUID of the note.

**Returns**: The full `Note` object.

## 4. `update_note`
Modifies an existing note. At least one field must be provided. For major updates (>80% body text change), this is a two-phase operation requiring user confirmation.

**Parameters**:
- `note_id` (str, required): UUID of the note to update.
- `title` (str | null): New title.
- `body` (str | null): New body.
- `tags` (list[str] | null): New tags list.
- `category` (str | null): New category.
- `confirmation_token` (str | null, default `null`): Token to confirm a major note update, obtained from the first call.

**Returns**: 
- Standard Update: The updated `Note` object.
- Major Update Turn 1: `{"requires_confirmation": true, "confirmation_token": "..."}`
- Major Update Turn 2: The updated `Note` object.

## 5. `delete_note`
Deletes a note. This is a two-phase operation requiring user confirmation.

**Parameters**:
- `note_id` (str, required): UUID of the note to delete.
- `confirmation_token` (str | null, default `null`): Token obtained from the first call.

**Returns**: 
- Turn 1: `{"requires_confirmation": true, "confirmation_token": "..."}`
- Turn 2: `{"deleted": true}`

## 6. `list_notes`
Lists all notes, optionally filtered and sorted.

**Parameters**:
- `tags` (list[str], default `[]`)
- `category` (str | null)
- `limit` (int, default `20`)
- `sort_by` (Literal["created_at", "updated_at", "title"], default `"updated_at"`)
- `sort_order` (Literal["asc", "desc"], default `"desc"`)

**Returns**: List of `Note` summaries.

## 7. `answer_question`
Answers questions by fetching specific notes.

**Parameters**:
- `question` (str, required): The question to answer.
- `relevant_note_ids` (list[str], default `[]`): Specific notes to pull as context.

**Returns**: Raw text of the requested notes for the agent to synthesize into an answer.
