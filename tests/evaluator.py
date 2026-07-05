import sys
import os

# Ensure the root directory is in sys.path so 'src' imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import json
from dotenv import load_dotenv
from tabulate import tabulate

from src.database import Database
from src.context import ConversationContext
from src.tools import NoteTools
from src.llm_client import LLMClient
from src.agent import Agent
from tests.scenarios import SCENARIOS

global_semantic_search = None

def get_semantic_search(db):
    global global_semantic_search
    if global_semantic_search is None:
        try:
            from src.embeddings import SemanticSearch
            global_semantic_search = SemanticSearch(db)
        except Exception as e:
            print(f"Warning: Could not initialize global SemanticSearch: {e}", flush=True)
            return None
    else:
        global_semantic_search.db = db
    return global_semantic_search

def run_scenario(scenario) -> dict:
    fd, db_path = tempfile.mkstemp()
    os.close(fd)
    
    db = None
    try:
        db = Database(db_path)
        user_id = f"test_{scenario.name.replace(' ', '_')}"
        context = ConversationContext(active_user_id=user_id)
        tools = NoteTools(db, context)
        
        # Semantic search is optional for eval (loads a heavy local model)
        if os.getenv("EVAL_SEMANTIC", "").lower() in ("1", "true", "yes"):
            sem_search = get_semantic_search(db)
            if sem_search:
                tools.embedding_func = sem_search.embed_text
                tools._semantic_search_obj = sem_search
        
        # Clear the failed providers cache so transient failures do not propagate across scenarios
        LLMClient.reset_provider_state()
        if os.getenv("EVAL_MOCK", "").lower() in ("1", "true", "yes"):
            from tests.mock_llm import MockLLMClient
            llm = MockLLMClient()
        else:
            llm = LLMClient()
        agent = Agent(llm, tools)
        
        called_tools = []
        for turn in scenario.turns:
            history_len_before = len(agent.history)
            agent.chat(turn)
            for msg in agent.history[history_len_before:]:
                if msg["role"] == "tool":
                    called_tools.append(msg["name"])
                    
        # Verify tool calls
        from collections import Counter
        expected_counts = Counter(scenario.expected_tool_calls)
        called_counts = Counter(called_tools)
        for expected_tool, expected_count in expected_counts.items():
            if called_counts[expected_tool] < expected_count:
                return {
                    "pass": False,
                    "reason": f"Expected tool '{expected_tool}' to be called at least {expected_count} time(s). Called: {called_tools}"
                }
                
        # Verify outcome
        if not scenario.expected_outcome(db, user_id):
            return {
                "pass": False,
                "reason": "Expected DB state assertion failed."
            }
            
        return {"pass": True, "reason": "OK"}
        
    except Exception as e:
        return {"pass": False, "reason": f"Exception: {e}"}
    finally:
        if db:
            db.close()
        try:
            os.unlink(db_path)
        except OSError:
            pass

def main():
    load_dotenv()
    
    print(f"Running {len(SCENARIOS)} Evaluation Scenarios...\n", flush=True)
    
    results = []
    passes = 0
    
    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"[{i}/{len(SCENARIOS)}] Running: {scenario.name}...", flush=True)
        result = run_scenario(scenario)
        if result["pass"]:
            passes += 1
            
        results.append([
            scenario.name,
            ", ".join(scenario.tags),
            "PASS" if result["pass"] else "FAIL",
            result["reason"]
        ])
        
    print("\n" + "="*50 + "\n")
    print(tabulate(results, headers=["Scenario", "Tags", "Status", "Details"], tablefmt="grid"))
    
    pass_rate = (passes / len(SCENARIOS)) * 100
    print(f"\nFinal Score: {passes}/{len(SCENARIOS)} ({pass_rate:.1f}%)")
    
    with open("eval_results.json", "w") as f:
        json.dump({
            "total": len(SCENARIOS),
            "passes": passes,
            "pass_rate": pass_rate,
            "mode": "mock" if os.getenv("EVAL_MOCK", "").lower() in ("1", "true", "yes") else "live",
            "results": [{"name": r[0], "pass": "PASS" in r[2], "reason": r[3]} for r in results]
        }, f, indent=2)

if __name__ == "__main__":
    main()
