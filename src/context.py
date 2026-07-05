from dataclasses import dataclass, field

@dataclass
class PendingAction:
    action: str  # 'delete' or 'update'
    token: str
    note_id: str
    payload: dict | None = None
    expires_at: float = 0.0

@dataclass
class ConversationContext:
    last_note_id: str | None = None
    last_search_results: list[str] = field(default_factory=list)  # list of IDs
    pending_confirmation: PendingAction | None = None
    active_user_id: str = "default"
