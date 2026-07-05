# Conversational Note-Taking Agent

A chat-based system that manages personal notes entirely through natural language. Built as an assessment for an AI/Data Engineer role.

## Features
- **Conversational CRUD**: Add, list, search, modify, and delete notes.
- **Intent Disambiguation**: The agent asks clarifying questions if a search returns multiple matches.
- **Stateful FSM Confirmation**: Destructive actions (e.g., delete) require a two-phase confirmation step, mediated via short-lived tokens in memory.
- **Multi-turn Context Resolution**: The agent keeps track of the "last referenced note" to handle follow-ups like *"Actually, add a deadline to that note."*
- **Semantic Search**: Uses `all-MiniLM-L6-v2` via `sentence-transformers` to find conceptually similar notes.
- **MCP Server**: Includes an MCP server wrapper (`mcp_server.py`) using `FastMCP`.
- **Free-Tier LLMs**: Defaults to Groq `llama-3.1-8b-instant` (fast, free), with Gemini 2.0 Flash and local Ollama as fallbacks. No paid API key needed.
- **Automatic provider fallback**: If the configured API runs out of quota, the agent switches to the other cloud API; if both are exhausted, it falls back to local Ollama. Exhausted providers are remembered in `.provider_state.json` so dead APIs are not retried on every message.
- **SQLite FTS5 Storage**: ACID-compliant local persistence with full-text search. SQLite was chosen for zero-config deployment, structured tag/date queries, and built-in FTS5 keyword search without a separate search engine.

See [docs/tool_schemas.md](docs/tool_schemas.md) for full tool API documentation.

## Setup Instructions

### 1. Requirements
- Python 3.12+

### 2. Environment
Copy the example environment file and add your API keys:
```bash
cp .env.example .env
```
Edit `.env` to set either `GEMINI_API_KEY` (primary) or `GROQ_API_KEY` (fallback).

### 3. Installation
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Unix:
source venv/bin/activate

pip install -r requirements.txt
```

## Running the Agent

### Terminal CLI REPL
Launch the interactive agent interface:
```bash
python -m src.cli
```

### Web UI
Launch the browser-based chat interface:
```bash
streamlit run src/web_ui.py
```
Then open http://localhost:8501

### Evaluation Harness
Run the automated test suite across 15 conversational scenarios (happy paths, disambiguation, edge cases, and multi-turn resolution):

```bash
python tests/evaluator.py
```

Results are printed as a table and saved to `eval_results.json`.

**How intent is measured:** each scenario specifies expected tool calls (minimum set) and a DB outcome assertion. A scenario passes only when both succeed — separating intent routing from execution correctness.

Optional: `EVAL_MOCK=1 python tests/evaluator.py` for fast deterministic testing without API keys (100% harness validation). Use the default command above to evaluate against a live free-tier LLM (Groq/Gemini).

### Model Context Protocol (MCP) Server
Launch the MCP server to expose note tools to MCP-compatible clients (e.g., Claude Desktop):
```bash
mcp run src/mcp_server.py
```

### Docker
Run the entire solution using Docker Compose:
```bash
docker-compose up --build
```
*Note: This will attach to the CLI REPL by default.*

## Multi-User Isolation Strategy (Bonus)
The database schema includes a `user_id` column. By default, the CLI and tools use `user_id = 'default'`. 
To support multiple users in a real application:
1. Pass `--user <username>` to the CLI, or extract the identity from an HTTP `Bearer` token in an API context.
2. The `ConversationContext` class maintains an `active_user_id` per session.
3. Every database query implicitly scopes to `WHERE user_id = ?`, ensuring complete isolation at the query level.

## Semantic Search (Bonus)
Notes are embedded on create/update using **`all-MiniLM-L6-v2`** (`sentence-transformers`). This model is small (~80 MB), runs locally on CPU, and produces 384-dimensional vectors suitable for cosine-similarity search over note title + body. Falls back to Gemini `text-embedding-004` if local embeddings fail to load.

Set `semantic=true` on `search_notes` for pure semantic search, or rely on automatic semantic fallback when keyword search returns no results.

## Assessment Requirements Checklist

| Requirement | Status | Implementation |
|---|---|---|
| Add notes (title, body, tags, category) | Done | `create_note` tool |
| List & search (keyword, tag, date, NL query) | Done | `search_notes`, `list_notes` + FTS5 + semantic |
| Modify notes | Done | `update_note` tool |
| Delete notes | Done | `delete_note` tool (two-phase confirmation) |
| Answer questions over notes | Done | `answer_question` + LLM synthesis |
| Intent disambiguation | Done | System prompt + search before destructive ops |
| Confirmation on destructive actions | Done | Token-based FSM for delete & major updates |
| Multi-turn awareness | Done | `ConversationContext.last_note_id` |
| Graceful error handling | Done | Tool errors + empty-search guidance in prompt |
| Evaluation harness (10–15 scenarios) | Done | `tests/evaluator.py` — 15 scenarios, see `eval_results.json` |
| Python + LLM integration | Done | Groq / Gemini / Ollama via `src/llm_client.py` |
| Tool/function schemas | Done | Pydantic models in `src/tool_schemas.py`, docs in `docs/tool_schemas.md` |
| Persistence | Done | SQLite + FTS5 (`src/database.py`) |
| CLI interface | Done | `python -m src.cli` |
| MCP server (bonus) | Done | `mcp run src/mcp_server.py` |
| Multi-user isolation (bonus) | Done | `user_id` column + `--user` flag |
| Docker (bonus) | Done | `Dockerfile` + `docker-compose.yml` |
| Semantic search (bonus) | Done | `src/embeddings.py` |
