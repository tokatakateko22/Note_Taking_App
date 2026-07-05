import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure the root directory is in sys.path so 'src' imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.markdown import Markdown

from src.database import Database
from src.context import ConversationContext
from src.tools import NoteTools
from src.llm_client import LLMClient
from src.agent import Agent

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Conversational note-taking agent")
    parser.add_argument("--user", default=os.getenv("DEFAULT_USER_ID", "default"),
                        help="User ID for note isolation (default: default)")
    args = parser.parse_args()

    console = Console()
    
    db_path = os.getenv("DB_PATH", "./notes.db")
    user_id = args.user
    
    db = Database(db_path)
    context = ConversationContext(active_user_id=user_id)
    tools = NoteTools(db, context)

    # Lazy-load embeddings on first create/search (avoids slow model load at startup)
    _semantic_search = {"obj": None}

    def _embed_text(text: str):
        if _semantic_search["obj"] is None:
            from src.embeddings import SemanticSearch
            _semantic_search["obj"] = SemanticSearch(db)
        return _semantic_search["obj"].embed_text(text)

    try:
        tools.embedding_func = _embed_text
    except Exception as e:
        console.print(f"[yellow]Warning: Could not initialize semantic search: {e}[/yellow]")
        
    try:
        llm = LLMClient()
    except Exception as e:
        console.print(f"[red]Error initializing LLM Client: {e}[/red]")
        return
        
    agent = Agent(llm, tools)

    provider_line = f"[bold green]Note Taking Agent[/bold green] (Provider: {llm.provider.upper()}, User: {user_id})"
    console.print(provider_line)
    if llm.using_fallback:
        console.print(
            f"[yellow]Warning: Using {llm.provider.upper()} fallback because "
            f"{llm.configured_provider.upper()} is unavailable. Responses may be slow.[/yellow]"
        )
        if llm.provider == "ollama":
            console.print(
                "[yellow]Cloud APIs are exhausted. For faster replies, add a fresh Groq/Gemini key "
                "and delete .provider_state.json[/yellow]"
            )
    console.print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
            if user_input.lower() in ('exit', 'quit'):
                break
            if not user_input.strip():
                continue
                
            response = agent.chat(user_input)
            
            console.print("\n[bold magenta]Agent:[/bold magenta]")
            console.print(Markdown(response))
            console.print("\n" + "-"*50 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
