import os
import warnings
import numpy as np

# Suppress Hugging Face warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

from src.database import Database
from src.models import Note

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore


def _valid_gemini_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key and key not in ("", "YOUR_GEMINI_API_KEY_HERE"):
        return key
    return None


class SemanticSearch:
    def __init__(self, db: Database):
        self.db = db
        self.threshold = float(os.getenv("SEMANTIC_THRESHOLD", "0.35"))
        self.mode = None
        self.model = None
        self.client = None

        if HAS_SENTENCE_TRANSFORMERS and SentenceTransformer is not None:
            try:
                model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                self.model = SentenceTransformer(model_name)
                self.mode = "local"
                return
            except Exception as e:
                warnings.warn(f"Local embedding model failed to load ({e}); trying Gemini fallback.")

        gemini_key = _valid_gemini_key()
        if gemini_key:
            from google import genai
            self.client = genai.Client(api_key=gemini_key)
            self.mode = "gemini"
            return

        raise ImportError(
            "Semantic search unavailable: install torchvision (pip install torchvision) "
            "or set a valid GEMINI_API_KEY for cloud embeddings."
        )

    def embed_text(self, text: str) -> np.ndarray:
        if self.mode == "local":
            return self.model.encode(text, convert_to_numpy=True)
        elif self.mode == "gemini":
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            return np.array(response.embeddings[0].values, dtype=np.float32)
        else:
            raise ValueError("Unknown embedding mode")

    def search(self, query: str, user_id: str, limit: int = 10) -> list[Note]:
        query_emb = self.embed_text(query)
        
        # Load all notes for this user to compare
        all_notes = self.db.get_all(user_id, limit=10000)
        
        scored_notes = []
        for note in all_notes:
            if note.embedding is not None:
                # Cosine similarity
                norm_query = np.linalg.norm(query_emb)
                norm_doc = np.linalg.norm(note.embedding)
                if norm_query > 0 and norm_doc > 0:
                    # Only compare if dimensions match
                    if len(note.embedding) == len(query_emb):
                        sim = np.dot(query_emb, note.embedding) / (norm_query * norm_doc)
                        if sim >= self.threshold:
                            scored_notes.append((sim, note))
                        
        scored_notes.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored_notes[:limit]]

