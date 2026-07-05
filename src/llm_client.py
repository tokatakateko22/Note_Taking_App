import os
import json
import logging
import time
from pathlib import Path
from google import genai
from google.genai import types
from groq import Groq

from src.tool_parse import extract_tool_calls_from_text, strip_tool_json_from_text, parse_tool_arguments

# Configure a file-based logger for internal agent warnings to prevent terminal output corruption
logger = logging.getLogger("note_agent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    try:
        fh = logging.FileHandler("agent.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
    except Exception:
        logger.addHandler(logging.NullHandler())

_PROVIDER_STATE_FILE = Path(__file__).resolve().parent.parent / ".provider_state.json"
_CLOUD_PROVIDERS = ("gemini", "groq")
_DEFAULT_EXHAUSTED_RETRY_MINUTES = 15


class LLMClient:
    # Providers skipped for this process: misconfigured keys or quota/auth exhaustion
    _failed_providers: set[str] = set()
    _exhausted_at: dict[str, float] = {}
    _state_loaded = False

    def __init__(self):
        self._load_persisted_state()

        configured = os.getenv("LLM_PROVIDER", "gemini").lower()
        if configured not in ("gemini", "groq", "ollama"):
            configured = "gemini"
        self.configured_provider = configured

        # Initialize Gemini
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
            self.gemini_client = None
            LLMClient._failed_providers.add("gemini")
        else:
            self.gemini_client = genai.Client(api_key=gemini_api_key)

        # Initialize Groq
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not groq_api_key or groq_api_key == "YOUR_GROQ_API_KEY_HERE":
            self.groq_client = None
            if "groq" not in LLMClient._failed_providers:
                LLMClient._failed_providers.add("groq")
        else:
            self.groq_client = Groq(api_key=groq_api_key)

        # Initialize Ollama (always available as final fallback)
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        self.provider = self._resolve_active_provider()
        self.using_fallback = self.provider != self.configured_provider

    @classmethod
    def _retry_after_seconds(cls) -> float:
        minutes = float(os.getenv("EXHAUSTED_RETRY_MINUTES", str(_DEFAULT_EXHAUSTED_RETRY_MINUTES)))
        return max(1.0, minutes * 60.0)

    @classmethod
    def _is_still_exhausted(cls, provider: str, exhausted_at: float) -> bool:
        return (time.time() - exhausted_at) < cls._retry_after_seconds()

    @classmethod
    def _load_persisted_state(cls) -> None:
        if cls._state_loaded:
            return
        cls._state_loaded = True
        if not _PROVIDER_STATE_FILE.exists():
            return
        try:
            data = json.loads(_PROVIDER_STATE_FILE.read_text(encoding="utf-8"))
            exhausted = data.get("exhausted_providers", {})
            # Legacy list format blocked providers forever — ignore so cloud APIs are retried.
            if isinstance(exhausted, list):
                return
            if not isinstance(exhausted, dict):
                return
            for provider, timestamp in exhausted.items():
                if provider not in _CLOUD_PROVIDERS:
                    continue
                try:
                    exhausted_at = float(timestamp)
                except (TypeError, ValueError):
                    continue
                if cls._is_still_exhausted(provider, exhausted_at):
                    cls._failed_providers.add(provider)
                    cls._exhausted_at[provider] = exhausted_at
        except Exception:
            pass

    @classmethod
    def _persist_exhausted_providers(cls) -> None:
        payload = {
            "exhausted_providers": {
                provider: cls._exhausted_at[provider]
                for provider in sorted(cls._failed_providers)
                if provider in _CLOUD_PROVIDERS and provider in cls._exhausted_at
            }
        }
        try:
            _PROVIDER_STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def reset_provider_state(cls) -> None:
        """Clear in-memory and persisted quota exhaustion (for tests or after renewing keys)."""
        cls._failed_providers.clear()
        cls._exhausted_at.clear()
        cls._state_loaded = True
        if _PROVIDER_STATE_FILE.exists():
            try:
                _PROVIDER_STATE_FILE.unlink()
            except Exception:
                pass

    @classmethod
    def _clear_provider_exhaustion(cls, provider: str) -> None:
        cls._failed_providers.discard(provider)
        cls._exhausted_at.pop(provider, None)
        cls._persist_exhausted_providers()

    def _resolve_active_provider(self) -> str:
        for provider in self._fallback_chain():
            if self._is_provider_available(provider):
                return provider
        return "ollama"

    def _fallback_chain(self) -> list[str]:
        """
        Fallback order:
        1. Configured cloud provider (LLM_PROVIDER)
        2. The other cloud provider
        3. Local Ollama
        """
        if self.configured_provider == "gemini":
            return ["gemini", "groq", "ollama"]
        if self.configured_provider == "groq":
            return ["groq", "gemini", "ollama"]
        return ["ollama", "groq", "gemini"]

    def _is_provider_available(self, provider: str) -> bool:
        if provider in LLMClient._failed_providers:
            return False
        if provider == "gemini":
            return self.gemini_client is not None
        if provider == "groq":
            return self.groq_client is not None
        return True

    def _is_quota_or_auth_error(self, error_str: str) -> bool:
        error_str = error_str.lower()
        quota_signals = (
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "rate_limit",
            "limit exceeded",
            "too many requests",
            "insufficient_quota",
        )
        auth_signals = (
            "invalid api key",
            "invalid_api_key",
            "api key not valid",
            "incorrect api key",
            "unauthorized",
            "authentication",
            "401",
            "permission denied",
        )
        return any(signal in error_str for signal in quota_signals + auth_signals)

    def _mark_provider_exhausted(self, provider: str, reason: str) -> None:
        if provider not in _CLOUD_PROVIDERS:
            return
        LLMClient._failed_providers.add(provider)
        LLMClient._exhausted_at[provider] = time.time()
        self._persist_exhausted_providers()
        retry_min = int(self._retry_after_seconds() // 60)
        logger.warning(
            f"{provider.upper()} unavailable ({reason}). "
            f"Will retry in up to {retry_min} min. Trying next provider..."
        )

        cloud_exhausted = all(p in LLMClient._failed_providers for p in _CLOUD_PROVIDERS)
        if cloud_exhausted:
            logger.warning(
                "Both cloud APIs (Gemini and Groq) are out of quota or misconfigured. "
                "Falling back to local LLM (Ollama)."
            )

    def generate_response(self, system_prompt: str, messages: list[dict], tools_def: list[dict] = None) -> dict:
        """
        messages: [{"role": "user"|"assistant"|"tool", "content": "...", "tool_calls": [...], "tool_call_id": "..."}]
        tools_def: [{"name": "create_note", "description": "...", "schema_class": CreateNoteInput}]

        Returns: {
            "text": str | None,
            "tool_calls": [{"id": "...", "name": "...", "arguments": dict}] | None
        }
        """
        providers_to_try = [
            p for p in self._fallback_chain() if self._is_provider_available(p)
        ]

        if not providers_to_try:
            raise Exception(
                "No LLM provider is available. Configure GEMINI_API_KEY or GROQ_API_KEY, "
                "or start Ollama locally."
            )

        last_error = None
        provider_errors: dict[str, str] = {}
        for index, provider in enumerate(providers_to_try):
            try:
                self.provider = provider
                self.using_fallback = provider != self.configured_provider
                if provider == "gemini":
                    result = self._call_gemini(system_prompt, messages, tools_def)
                elif provider == "groq":
                    result = self._call_groq(system_prompt, messages, tools_def)
                else:
                    if not self._ollama_is_reachable():
                        raise Exception(
                            f"Ollama at {self.ollama_host} is not running or model "
                            f"'{self.ollama_model}' is not available. "
                            f"Start Ollama and run: ollama pull {self.ollama_model.split(':')[0]}"
                        )
                    result = self._call_ollama(system_prompt, messages, tools_def)
                if provider in _CLOUD_PROVIDERS:
                    self._clear_provider_exhaustion(provider)
                return result
            except Exception as e:
                last_error = e
                error_str = str(e)
                provider_errors[provider] = error_str

                if provider in _CLOUD_PROVIDERS and self._is_quota_or_auth_error(error_str):
                    self._mark_provider_exhausted(
                        provider,
                        "quota exhausted or invalid API key",
                    )
                elif provider in _CLOUD_PROVIDERS:
                    logger.warning(
                        f"{provider.upper()} transient error: {error_str}. Trying next provider..."
                    )
                else:
                    logger.warning(f"OLLAMA error: {error_str}")

                if index == len(providers_to_try) - 1:
                    break
                continue

        if last_error:
            raise Exception(self._format_all_providers_failed(provider_errors))
        raise Exception("No active LLM provider has valid API keys or connection.")

    def _format_all_providers_failed(self, errors: dict[str, str]) -> str:
        lines = ["All LLM providers failed:"]
        for provider, err in errors.items():
            lines.append(f"  - {provider.upper()}: {err[:350]}")

        combined = " ".join(errors.values()).lower()
        if "rate limit" in combined or "429" in combined or "resource_exhausted" in combined:
            lines.append(
                "Cloud API quota/rate limit reached. Wait for reset, create a new free key "
                "(Groq: console.groq.com, Gemini: aistudio.google.com), then delete "
                ".provider_state.json and restart."
            )
        if "ollama" in errors and "timed out" in errors["ollama"].lower():
            lines.append(
                f"Local fallback is too slow with '{self.ollama_model}'. "
                "Use a smaller model, e.g. OLLAMA_MODEL=llama3.2:1b"
            )
        return "\n".join(lines)

    def _ollama_is_reachable(self) -> bool:
        import httpx

        try:
            response = httpx.get(f"{self.ollama_host}/api/tags", timeout=3.0)
            response.raise_for_status()
        except Exception:
            return False

        model_root = self.ollama_model.split(":")[0]
        try:
            models = response.json().get("models", [])
            names = [m.get("name", "") for m in models]
            return any(name.split(":")[0] == model_root or name.startswith(model_root) for name in names)
        except Exception:
            return True

    def _ollama_read_timeout(self) -> float:
        return float(os.getenv("OLLAMA_READ_TIMEOUT", "45"))

    def _call_gemini(self, system_prompt: str, messages: list[dict], tools_def: list[dict]) -> dict:
        gemini_tools = []
        if tools_def:
            func_decls = []
            for t in tools_def:
                schema = t["schema_class"].model_json_schema()
                properties = {}
                for k, v in schema.get("properties", {}).items():
                    prop_type = v.get("type", "string").upper()
                    if prop_type == "ARRAY":
                        properties[k] = types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.STRING)
                        )
                    elif prop_type == "INTEGER":
                        properties[k] = types.Schema(type=types.Type.INTEGER)
                    elif prop_type == "BOOLEAN":
                        properties[k] = types.Schema(type=types.Type.BOOLEAN)
                    else:
                        properties[k] = types.Schema(type=types.Type.STRING, description=v.get("description", ""))

                func_decl = types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties=properties,
                        required=schema.get("required", [])
                    )
                )
                func_decls.append(func_decl)
            
            gemini_tools = [types.Tool(function_declarations=func_decls)]

        # Map messages
        contents = []
        for msg in messages:
            if msg["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])]))
            elif msg["role"] == "assistant":
                if msg.get("tool_calls"):
                    parts = [types.Part.from_function_call(
                        name=tc["name"], 
                        args=tc["arguments"]
                    ) for tc in msg["tool_calls"]]
                    contents.append(types.Content(role="model", parts=parts))
                else:
                    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg.get("content", ""))]))
            elif msg["role"] == "tool":
                part = types.Part.from_function_response(
                    name=msg.get("name", "tool"),
                    response={"result": json.loads(msg["content"])}
                )
                contents.append(types.Content(role="user", parts=[part]))

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=gemini_tools if gemini_tools else None,
            temperature=0.2,
        )

        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=contents,
            config=config
        )

        result = {"text": None, "tool_calls": None}
        if response.function_calls:
            result["tool_calls"] = []
            for i, fc in enumerate(response.function_calls):
                result["tool_calls"].append({
                    "id": f"call_{i}",
                    "name": fc.name,
                    "arguments": fc.args
                })
        elif response.text:
            result["text"] = response.text

        return result

    def _finalize_tool_response(self, text: str | None, tool_calls: list[dict] | None) -> dict:
        """Normalize provider output; recover tool calls printed as plain JSON text."""
        if tool_calls:
            return {"text": text, "tool_calls": tool_calls}
        if text:
            recovered = extract_tool_calls_from_text(text)
            if recovered:
                return {
                    "text": strip_tool_json_from_text(text),
                    "tool_calls": recovered,
                }
        return {"text": text, "tool_calls": None}

    def _call_groq(self, system_prompt: str, messages: list[dict], tools_def: list[dict]) -> dict:
        groq_tools = []
        if tools_def:
            for t in tools_def:
                schema = t["schema_class"].model_json_schema()
                groq_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": {
                            "type": "object",
                            "properties": schema.get("properties", {}),
                            "required": schema.get("required", [])
                        }
                    }
                })

        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            api_msg = {"role": msg["role"]}
            if msg.get("content"):
                api_msg["content"] = msg["content"]
            if msg.get("tool_calls"):
                api_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"])
                        }
                    } for tc in msg["tool_calls"]
                ]
            if msg.get("tool_call_id"):
                api_msg["tool_call_id"] = msg["tool_call_id"]
            if msg.get("name"):
                api_msg["name"] = msg["name"]
            
            api_messages.append(api_msg)

        params = {
            "model": self.groq_model,
            "messages": api_messages,
            "temperature": 0.2
        }
        if groq_tools:
            params["tools"] = groq_tools
            params["tool_choice"] = "auto"

        response = self.groq_client.chat.completions.create(**params)
        msg = response.choices[0].message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": parse_tool_arguments(tc.function.arguments),
                })

        return self._finalize_tool_response(msg.content, tool_calls)

    def _call_ollama(self, system_prompt: str, messages: list[dict], tools_def: list[dict]) -> dict:
        import httpx
        
        ollama_tools = []
        if tools_def:
            for t in tools_def:
                schema = t["schema_class"].model_json_schema()
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": {
                            "type": "object",
                            "properties": schema.get("properties", {}),
                            "required": schema.get("required", [])
                        }
                    }
                })

        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            api_msg = {"role": msg["role"]}
            if msg.get("content"):
                api_msg["content"] = msg["content"]
            if msg.get("tool_calls"):
                api_msg["tool_calls"] = [
                    {
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    } for tc in msg["tool_calls"]
                ]
            if msg.get("name"):
                api_msg["name"] = msg["name"]
            
            api_messages.append(api_msg)

        payload = {
            "model": self.ollama_model,
            "messages": api_messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
            }
        }
        if ollama_tools:
            payload["tools"] = ollama_tools

        url = f"{self.ollama_host}/api/chat"
        read_timeout = self._ollama_read_timeout()
        timeout = httpx.Timeout(read_timeout, connect=15.0)
        try:
            response = httpx.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                err_detail = response.json().get("error", response.text)
            except Exception:
                err_detail = response.text
            raise Exception(f"Ollama server error (HTTP {response.status_code}): {err_detail}")
        except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            raise Exception(f"Ollama request timed out. The local model is taking too long to load or evaluate: {str(e)}")
        
        resp_json = response.json()
        msg = resp_json["message"]

        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = []
            for i, tc in enumerate(msg["tool_calls"]):
                fn = tc["function"]
                tool_calls.append({
                    "id": tc.get("id") or f"call_{i}",
                    "name": fn["name"],
                    "arguments": parse_tool_arguments(fn.get("arguments")),
                })

        return self._finalize_tool_response(msg.get("content"), tool_calls)
