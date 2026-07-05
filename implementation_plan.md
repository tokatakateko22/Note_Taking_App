# Conversational Note-Taking Agent — Implementation Plan

> **Author perspective**: Senior AI / Data Engineer  
> **Working directory**: `d:\Note_Taking_App\` (greenfield)  
> **Target**: Production-quality submission demonstrating conversational AI design, tool engineering, and state management.

---

## Problem Framing

The core challenge is **not CRUD** — it is building a stateful conversational loop that maps ambiguous natural language into deterministic, typed tool calls, handles multi-turn references, guards against destructive actions, and degrades gracefully. Every design decision flows from this.

---

## Key Design Decisions & Justifications

### Storage: SQLite (not JSON)
| Concern | SQLite | JSON File |
|---|---|---|
| Concurrent writes | ✅ WAL mode, ACID | ❌ Race conditions |
| Full-text search | ✅ FTS5 extension built-in | ❌ Manual iteration |
| Indexed queries (tag, date) | ✅ Native B-tree indexes | ❌ O(n) scan |
| Schema evolution | ✅ ALTER TABLE / migrations | ❌ Manual versioning |
| Portability | ✅ Single `.db` file | ✅ Single `.json` file |

**Decision**: SQLite with FTS5 for keyword search. This gives near-zero-dependency full-text search without spinning up a vector DB for the core path.

### LLM Provider: Google Gemini 2.0 Flash (free tier primary) + Groq / Llama 3.3
| Feature | Implementation | Notes |
|---|---|---|
| **LLM (primary)** | Gemini 2.0 Flash (free) | Native function calling, 1M TPM free tier, JSON output mode; via `google-genai` SDK |
| **LLM (fallback)** | Groq / Llama 3.3 70B (free) | OpenAI-compatible tool calling, sub-second inference, generous free tier |

- `llm_client.py` wraps both SDKs behind a common interface — switching provider is a single env var change (`LLM_PROVIDER=gemini|groq`).
- **Gemini chosen as default**: 1M TPM free ceiling + native function-calling JSON response format align best with multi-turn tool loops.
- **Groq as fallback**: OpenAI-compatible API format means near-zero code change to swap.

### Tool Schema Definition: Pydantic v2 + JSON Schema export
- Type safety at definition time.
- `.model_json_schema()` auto-generates the `tools` / `function_declarations` array for both Gemini and Groq.
- Validation on tool *return values* prevents silent data corruption.

### Conversation State: In-memory (per session) + SQLite (notes)
- The **message history list** is held in RAM for the active session — this is the standard agentic pattern.
- Notes persist to SQLite across sessions.
- A `ConversationContext` dataclass tracks the **last referenced note ID** to enable follow-up resolution ("that last note").

### Semantic Search: `sentence-transformers` (all-MiniLM-L6-v2) + NumPy cosine similarity
- Runs **fully locally** — no embedding API costs or latency.
- `all-MiniLM-L6-v2` is 80 MB, 384-dim, excellent speed/quality tradeoff for personal note corpora.
- Embeddings stored as BLOB in SQLite (serialised NumPy array). No extra vector DB dependency.
- Upgraded to ChromaDB if scale demands it (drop-in swap).

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                  CLI / Web UI               │
│            (cli.py / app.py)                │
└───────────────────┬─────────────────────────┘
                    │ user message
                    ▼
┌─────────────────────────────────────────────┐
│              Agent Core (agent.py)          │
│  • Maintains message history                │
│  • Runs LLM → tool call loop                │
│  • Manages ConversationContext              │
│  • Handles confirmation state machine       │
└────────┬──────────────────────┬─────────────┘
         │ tool calls           │ LLM API calls
         ▼                      ▼
┌─────────────────┐    ┌──────────────────────────────┐
│  Tool Layer     │    │  LLM Client (llm_client.py)  │
│  (tools.py)     │    │  • Gemini 2.0 Flash (default)│
│                 │    │  • Groq / Llama 3.3 (alt.)   │
│  • create_note  │    └──────────────────────┘
│  • search_notes │
│  • get_note     │
│  • update_note  │
│  • delete_note  │
│  • list_notes   │
│  • answer_question│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│           Database Layer (database.py)      │
│  • SQLite + FTS5                            │
│  • ORM-lite via dataclasses + raw SQL       │
│  • Embedding BLOB column                    │
└─────────────────────────────────────────────┘
```

---

## Proposed Project Structure

```
Note_Taking_App/
├── src/
│   ├── __init__.py
│   ├── agent.py            # Agent loop, multi-turn state, confirmation FSM
│   ├── tools.py            # Tool implementations (callable by agent)
│   ├── tool_schemas.py     # Pydantic models → JSON schema for LLM
│   ├── database.py         # SQLite layer: schema, CRUD, FTS5 queries
│   ├── embeddings.py       # Embedding generation + cosine search (bonus)
│   ├── llm_client.py       # Thin wrapper around OpenAI client
│   ├── models.py           # Shared dataclasses: Note, SearchResult, ToolResult
│   ├── context.py          # ConversationContext: last note ref, pending actions
│   ├── mcp_server.py       # MCP server via FastMCP (bonus)
│   └── cli.py              # REPL entry point
├── tests/
│   ├── conftest.py         # Fixtures: in-memory DB, mock LLM
│   ├── test_tools.py       # Unit tests for each tool function
│   ├── test_agent.py       # Integration tests for agent loop
│   ├── scenarios.py        # 10–15 conversational scenario definitions
│   └── evaluator.py        # Scenario runner + pass/fail reporter
├── docs/
│   └── tool_schemas.md     # Full tool schema reference
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Component Breakdown

---

### 1. Data Model (`src/models.py`)

```python
@dataclass
class Note:
    id: str                    # UUID4
    title: str
    body: str
    tags: list[str]
    category: str | None
    created_at: datetime
    updated_at: datetime
    user_id: str = "default"  # Multi-user isolation (bonus)
    embedding: np.ndarray | None = None  # Semantic search (bonus)

@dataclass
class ToolResult:
    success: bool
    data: Any
    error: str | None = None
    requires_confirmation: bool = False
    confirmation_token: str | None = None
```

---

### 2. Database Layer (`src/database.py`)

**Schema**:
```sql
CREATE TABLE notes (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT 'default',
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    tags        TEXT NOT NULL,          -- JSON array
    category    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    embedding   BLOB                   -- serialised np.float32 array
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE notes_fts USING fts5(
    title, body, tags,
    content=notes, content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER notes_ai AFTER INSERT ON notes ...
CREATE TRIGGER notes_au AFTER UPDATE ON notes ...
CREATE TRIGGER notes_ad AFTER DELETE ON notes ...

CREATE INDEX idx_notes_user    ON notes(user_id);
CREATE INDEX idx_notes_tags    ON notes(tags);       -- partial, JSON
CREATE INDEX idx_notes_created ON notes(created_at);
```

**Key methods**:
- `insert(note: Note) → Note`
- `get_by_id(note_id: str, user_id: str) → Note | None`
- `fts_search(query: str, user_id: str, limit: int) → list[Note]`
- `filter_by_tags(tags: list[str], user_id: str) → list[Note]`
- `filter_by_date_range(from_dt, to_dt, user_id) → list[Note]`
- `update(note_id: str, patches: dict, user_id: str) → Note`
- `delete(note_id: str, user_id: str) → bool`
- `get_all(user_id: str, limit: int) → list[Note]`

---

### 3. Tool Schemas (`src/tool_schemas.py`)

All schemas are Pydantic BaseModel → `.model_json_schema()` produces the `parameters` block for OpenAI.

#### `create_note`
```python
class CreateNoteInput(BaseModel):
    title: str = Field(..., description="Short descriptive title for the note")
    body: str = Field(..., description="Full text content of the note")
    tags: list[str] = Field(default=[], description="Categorical tags, e.g. ['meetings', 'urgent']")
    category: str | None = Field(None, description="Top-level category, e.g. 'work', 'personal'")
```
**Returns**: `Note` (serialised as dict)

#### `search_notes`
```python
class SearchNotesInput(BaseModel):
    query: str | None = Field(None, description="Keyword or phrase to search title and body")
    tags: list[str] = Field(default=[], description="Filter by one or more tags (OR logic)")
    category: str | None = None
    date_from: str | None = Field(None, description="ISO 8601 date lower bound, e.g. '2025-01-01'")
    date_to: str | None = Field(None, description="ISO 8601 date upper bound")
    semantic: bool = Field(False, description="Use semantic similarity search instead of keyword match")
    limit: int = Field(10, ge=1, le=50)
```
**Returns**: `list[Note]` (summaries, not full bodies unless limit=1)

#### `get_note`
```python
class GetNoteInput(BaseModel):
    note_id: str = Field(..., description="UUID of the note to retrieve")
```
**Returns**: `Note` (full)

#### `update_note`
```python
class UpdateNoteInput(BaseModel):
    note_id: str
    title: str | None = None
    body: str | None = None
    tags: list[str] | None = None
    category: str | None = None
    # At least one field must be non-None (validated in model_validator)
```
**Returns**: `Note` (updated) — **triggers confirmation flow**

#### `delete_note`
```python
class DeleteNoteInput(BaseModel):
    note_id: str
    confirmation_token: str | None = Field(
        None,
        description="Must be provided on second call after user confirms deletion"
    )
```
**Returns**: `{"deleted": true, "note_id": "..."}` — **two-phase: first call returns token + prompt, second call with token executes**

#### `list_notes`
```python
class ListNotesInput(BaseModel):
    tags: list[str] = []
    category: str | None = None
    limit: int = Field(20, ge=1, le=100)
    sort_by: Literal["created_at", "updated_at", "title"] = "updated_at"
    sort_order: Literal["asc", "desc"] = "desc"
```
**Returns**: `list[Note]` (title + id + tags + dates only)

#### `answer_question`
```python
class AnswerQuestionInput(BaseModel):
    question: str = Field(..., description="Natural language question to answer using note content")
    relevant_note_ids: list[str] = Field(
        default=[],
        description="Specific note IDs to reason over; if empty, agent searches first"
    )
```
**Returns**: synthesised answer string (LLM call with note context injected)

---

### 4. Tool Implementations (`src/tools.py`)

Each function:
1. Validates input via Pydantic (already done by agent dispatch).
2. Calls the DB layer.
3. Returns a `ToolResult` — never raises; errors are captured and returned as `success=False`.
4. Updates `ConversationContext.last_note_id` on create/get/update.

**Critical: `delete_note` two-phase pattern**:
```
Turn 1: Agent calls delete_note(note_id="abc")
        → DB is NOT touched
        → Returns {requires_confirmation: True, token: "del-abc-<hash>", message: "Are you sure?"}
Turn 2: User says "yes"
        → Agent calls delete_note(note_id="abc", confirmation_token="del-abc-<hash>")
        → Token validated against in-memory store (TTL: 60s)
        → DB delete executed
```

The same pattern applies to large `update_note` calls (body replacement >80% character change).

---

### 5. Agent Core (`src/agent.py`)

**Agent loop (per user message)**:
```
1. Append user message to history
2. Call LLM with: system_prompt + history + tool_definitions
3. If response has tool_calls:
   a. Validate each call against Pydantic schema
   b. Dispatch to tool function
   c. Append tool result to history
   d. Loop back to step 2 (max 5 iterations)
4. If response is text: stream to user, break loop
5. Update ConversationContext from tool results
```

**System Prompt design** (critical for intent disambiguation):
```
You are a personal note-taking assistant. Rules:
1. If a search returns >1 note and the user's intent targets a specific one, 
   LIST the matches and ASK which one they mean. Do NOT guess.
2. For delete or major updates, ALWAYS confirm with the user before calling the tool.
3. When the user says "that note", "the last one", "it" — resolve using 
   ConversationContext.last_note_id injected below.
4. If a search returns 0 results, say so clearly and suggest broadening the query.
5. Today's date is {today}. Use this for relative date parsing ("last week").

Current context: last_referenced_note_id={ctx.last_note_id}
```

**`ConversationContext` dataclass** (`src/context.py`):
```python
@dataclass
class ConversationContext:
    last_note_id: str | None = None
    last_search_results: list[str] = field(default_factory=list)  # list of IDs
    pending_confirmation: PendingAction | None = None
    active_user_id: str = "default"
```

---

### 6. Embeddings (Bonus) (`src/embeddings.py`)

- Model: `sentence-transformers/all-MiniLM-L6-v2`
  - 80 MB, 384 dimensions, ~14k sentences/sec on CPU
  - **Rationale**: Best speed/quality tradeoff for local personal note corpora; no API cost; open source
- Embeddings generated on `create_note` and `update_note` (body change)
- Stored as `BLOB` in SQLite (serialised `np.float32`)
- Search: load all embeddings into memory (fine for <10k notes), compute cosine similarity with NumPy
- Threshold: 0.35 cosine similarity (configurable in `.env`)

---

### 7. MCP Server (Bonus) (`src/mcp_server.py`)

Using `mcp` (FastMCP) library:
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("notes-agent")

@mcp.tool()
def create_note(title: str, body: str, tags: list[str]) -> dict: ...

@mcp.tool()
def search_notes(query: str, tags: list[str]) -> list[dict]: ...
```
Run with `mcp run src/mcp_server.py`. All 6 tools exposed. No auth in stub; token header hook documented.

---

### 8. Multi-User Isolation (Bonus)

- `user_id` column on `notes` table (already in schema).
- `ConversationContext.active_user_id` set at session start.
- CLI flag: `--user <name>` (hashed to UUID in prod).
- Auth strategy stub: Bearer token → `user_id` mapping in a `users` table. Described in README.

---

### 9. Evaluation Harness (`tests/evaluator.py` + `tests/scenarios.py`)

**Scenario format**:
```python
@dataclass
class Scenario:
    name: str
    turns: list[str]              # User messages in order
    expected_tool_calls: list[str]  # Tool names that MUST be called
    expected_outcome: Callable[[DB], bool]  # State assertion
    tags: list[str] = []          # "happy_path", "edge_case", "disambiguation"
```

**15 Scenarios**:

| # | Name | Tags | Key Assertion |
|---|------|------|---------------|
| 1 | Create basic note | happy_path | Note exists in DB with correct title/body |
| 2 | Create note with tags | happy_path | Tags stored correctly |
| 3 | Search by keyword | happy_path | Correct note returned |
| 4 | Search by tag | happy_path | Only tagged notes returned |
| 5 | Search by date range | happy_path | Notes filtered by date |
| 6 | Update note body | happy_path | Confirmation requested, then updated |
| 7 | Delete note | happy_path | Confirmation requested, then deleted |
| 8 | Ambiguous search (2 matches) | disambiguation | Agent asks clarification, does NOT act |
| 9 | Follow-up "that note" reference | multi_turn | Correct note ID resolved from context |
| 10 | Delete non-existent note | edge_case | Graceful error message |
| 11 | Search returns 0 results | edge_case | Clear message + suggestion |
| 12 | Refuse deletion without confirmation | edge_case | DB unchanged after single "delete" message |
| 13 | Summarise tagged notes | reasoning | Answer contains content from all tagged notes |
| 14 | Contradiction detection | reasoning | Agent identifies conflicting notes |
| 15 | Update via follow-up ("add deadline to that") | multi_turn | Correct note patched |

**Pass criteria**: tool call appears in history AND DB state assertion passes.
**Report**: JSON + human-readable table with pass/fail per scenario and overall pass rate.

---

## Proposed File-by-File Implementation Order

```
Phase 1 — Foundation (Days 1–2)
  [1] src/models.py           ← shared types
  [2] src/database.py         ← SQLite + FTS5 schema + CRUD
  [3] src/tool_schemas.py     ← Pydantic schemas, JSON schema export
  [4] src/tools.py            ← Tool implementations
  [5] tests/test_tools.py     ← Unit tests for each tool

Phase 2 — Agent Loop (Day 3)
  [6] src/llm_client.py       ← OpenAI thin wrapper
  [7] src/context.py          ← ConversationContext
  [8] src/agent.py            ← Agent loop, system prompt, dispatch
  [9] src/cli.py              ← REPL

Phase 3 — Evaluation (Day 4)
  [10] tests/scenarios.py     ← 15 scenario definitions
  [11] tests/evaluator.py     ← Runner + reporter
  [12] tests/conftest.py      ← Fixtures

Phase 4 — Bonus (Day 5)
  [13] src/embeddings.py      ← Semantic search
  [14] src/mcp_server.py      ← MCP exposure
  [15] Dockerfile + docker-compose.yml
  [16] docs/tool_schemas.md
  [17] README.md
```

---

## Dependencies (`requirements.txt`)

```
# LLM — free tier providers (no credit card required)
google-genai>=1.5.0            # Gemini 2.0 Flash via Google AI Studio (primary)
groq>=0.9.0                    # Groq / Llama 3.3 70B (fallback)

# Core
pydantic>=2.7.0
python-dotenv>=1.0.0

# Bonus: semantic search (runs fully locally, no API cost)
sentence-transformers>=3.0.0
numpy>=1.26.0

# Bonus: MCP server
mcp[cli]>=1.3.0

# CLI & evaluation
rich>=13.0.0                   # pretty terminal output
pytest>=8.0.0
pytest-asyncio>=0.23.0
tabulate>=0.9.0                # eval report tables
```

---

## Environment Variables (`.env.example`)

```
# LLM provider selection — set to 'gemini' or 'groq' (both free tier)
LLM_PROVIDER=gemini

# Google Gemini (primary) — free key from https://aistudio.google.com
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.0-flash

# Groq (fallback) — free key from https://console.groq.com
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.3-70b-versatile

# Storage
DB_PATH=./notes.db

# Semantic search (bonus — runs locally, no API key needed)
EMBEDDING_MODEL=all-MiniLM-L6-v2
SEMANTIC_THRESHOLD=0.35

# App
DEFAULT_USER_ID=default
LOG_LEVEL=INFO
```

---

## Verification Plan

### Automated
```bash
pytest tests/ -v --tb=short          # unit + integration
python tests/evaluator.py            # conversational scenario harness
```
Target: **≥13/15 scenarios passing** (87%+).

### Manual Smoke Test
```bash
python -m src.cli
> Save a note about the team standup — we agreed to move it to Tuesdays, tag it as meetings.
> What did I write about standups?
> Update my standup note to say the meeting is now on Wednesdays.
> Delete the standup note.
```

### MCP Validation (Bonus)
```bash
mcp run src/mcp_server.py
mcp dev src/mcp_server.py   # inspect tool list in MCP inspector
```

---

## Open Questions / Decisions for Review

> [!IMPORTANT]
> **LLM Provider**: Plan defaults to **Google Gemini 2.0 Flash** (free, from [AI Studio](https://aistudio.google.com)) with **Groq / Llama 3.3 70B** as a drop-in fallback (free, from [console.groq.com](https://console.groq.com)). Both require only a free API key — no billing setup. The `llm_client.py` abstraction keeps the swap to a single env var: `LLM_PROVIDER=gemini|groq`.

> [!IMPORTANT]
> **Semantic Search Scope**: Embedding generation on every `create`/`update` adds ~50–200ms per call locally. If this is a concern, embeddings can be computed lazily (on first semantic search). Confirm preferred behaviour.

> [!NOTE]
> **Web UI**: The plan targets a terminal REPL (`rich`-styled). If a minimal web UI (FastAPI + simple HTML) is preferred over CLI, this is a one-day addition and the agent core is unchanged.

> [!NOTE]
> **Confirmation Mechanism**: The two-phase delete/update uses an in-memory token store (TTL 60s). This is intentionally simple — tokens are not persisted across restarts. This is a pragmatic tradeoff for a personal tool.
