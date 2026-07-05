import sys
import os
import streamlit as st
from dotenv import load_dotenv

# Ensure the root directory is in sys.path so 'src' imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import Database
from src.context import ConversationContext
from src.tools import NoteTools
from src.llm_client import LLMClient
from src.agent import Agent

# Page configuration
st.set_page_config(
    page_title="Note-Taking Agent",
    page_icon="📝",
    layout="centered"
)

AGENT_VERSION = 3


def _init_agent() -> None:
    load_dotenv()
    db_path = os.getenv("DB_PATH", "./notes.db")
    user_id = os.getenv("DEFAULT_USER_ID", "default")

    db = Database(db_path)
    context = ConversationContext(active_user_id=user_id)
    tools = NoteTools(db, context)

    _semantic_search = {"obj": None}

    def _embed_text(text: str):
        if _semantic_search["obj"] is None:
            from src.embeddings import SemanticSearch
            _semantic_search["obj"] = SemanticSearch(db)
        return _semantic_search["obj"].embed_text(text)

    try:
        tools.embedding_func = _embed_text
    except Exception as e:
        st.warning(f"Could not initialize semantic search: {e}")

    llm = LLMClient()
    st.session_state.agent = Agent(llm, tools)
    st.session_state.provider = llm.provider.upper()
    st.session_state.agent_version = AGENT_VERSION


def _provider_label() -> str:
    llm = st.session_state.agent.llm
    label = getattr(llm, "provider", st.session_state.get("provider", "unknown")).upper()
    if getattr(llm, "using_fallback", False):
        configured = getattr(llm, "configured_provider", label).upper()
        label = f"{label} (fallback from {configured})"
    return label


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if (
    "agent" not in st.session_state
    or st.session_state.get("agent_version", 0) < AGENT_VERSION
    or not hasattr(st.session_state.agent.llm, "using_fallback")
):
    try:
        _init_agent()
    except Exception as e:
        st.error(f"Error initializing LLM Client: {e}")
        st.stop()

# Inject visual styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600&display=swap');

/* Main container and fonts */
.main .block-container {
    font-family: 'Inter', sans-serif;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Beautiful Title styling */
h1 {
    font-family: 'Outfit', sans-serif;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #6366F1 0%, #EC4899 50%, #F59E0B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px !important;
}

/* Card and message hover effects */
.stChatMessage {
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stChatMessage:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
}

/* Chat Input custom borders and focus states */
div[data-testid="stChatInput"] {
    border-radius: 20px !important;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

st.title("📝 Note-Taking Agent")
st.caption(f"Conversational AI for managing your notes (Provider: {_provider_label()})")
llm = st.session_state.agent.llm
if llm.provider == "ollama":
    st.info(
        "Using local Ollama — responses are slower than cloud APIs. "
        "For faster replies, ensure Groq/Gemini keys are valid in `.env`, "
        "or use a smaller model like `llama3.2:1b`."
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me to create, search, or update a note..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.chat(prompt)
                st.session_state.provider = st.session_state.agent.llm.provider.upper()
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error processing request: {e}")
