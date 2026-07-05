from dataclasses import dataclass
from datetime import datetime
from typing import Any
import numpy as np

@dataclass
class Note:
    id: str                    # UUID4
    title: str
    body: str
    tags: list[str]
    category: str | None
    created_at: datetime
    updated_at: datetime
    user_id: str = "default"  # Multi-user isolation
    embedding: np.ndarray | None = None  # Semantic search

@dataclass
class ToolResult:
    success: bool
    data: Any
    error: str | None = None
    requires_confirmation: bool = False
    confirmation_token: str | None = None
