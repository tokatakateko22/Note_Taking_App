from dataclasses import dataclass
from typing import Callable
from src.database import Database

@dataclass
class Scenario:
    name: str
    turns: list[str]
    expected_tool_calls: list[str]
    expected_outcome: Callable[[Database, str], bool]
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

def s1_check(db: Database, user_id: str) -> bool:
    notes = db.get_all(user_id)
    return len(notes) == 1 and notes[0].title == "Team Standup"

def s2_check(db: Database, user_id: str) -> bool:
    notes = db.get_all(user_id)
    return len(notes) == 1 and "urgent" in notes[0].tags

def s3_check(db: Database, user_id: str) -> bool:
    # Just verify it ran the search
    return True 

def s4_check(db: Database, user_id: str) -> bool:
    return True

def s5_check(db: Database, user_id: str) -> bool:
    return True

def s6_check(db: Database, user_id: str) -> bool:
    notes = db.get_all(user_id)
    return len(notes) == 1 and "Wednesday" in notes[0].body

def s7_check(db: Database, user_id: str) -> bool:
    notes = db.get_all(user_id)
    return len(notes) == 0

def s8_check(db: Database, user_id: str) -> bool:
    # Ambiguous search should NOT delete anything, even if mentioned
    notes = db.get_all(user_id)
    return len(notes) == 2

def s9_check(db: Database, user_id: str) -> bool:
    notes = db.get_all(user_id)
    return len(notes) == 1 and "deadline" in notes[0].body.lower()

def s10_check(db: Database, user_id: str) -> bool:
    return True

def s11_check(db: Database, user_id: str) -> bool:
    return True

def s12_check(db: Database, user_id: str) -> bool:
    notes = db.get_all(user_id)
    return len(notes) == 1  # Should not be deleted

def s13_check(db: Database, user_id: str) -> bool:
    return True

def s14_check(db: Database, user_id: str) -> bool:
    return True

def s15_check(db: Database, user_id: str) -> bool:
    notes = db.get_all(user_id)
    # Check if patched via follow-up
    return len(notes) == 1 and "deadline" in notes[0].tags

SCENARIOS = [
    Scenario(
        name="Create basic note",
        turns=["Save a note titled 'Team Standup' with body 'Move to Tuesdays'."],
        expected_tool_calls=["create_note"],
        expected_outcome=s1_check,
        tags=["happy_path"]
    ),
    Scenario(
        name="Create note with tags",
        turns=["Create a note about groceries, tag it as urgent."],
        expected_tool_calls=["create_note"],
        expected_outcome=s2_check,
        tags=["happy_path"]
    ),
    Scenario(
        name="Search by keyword",
        turns=["Create a note about apples.", "Search for apples."],
        expected_tool_calls=["create_note", "search_notes"],
        expected_outcome=s3_check,
        tags=["happy_path"]
    ),
    Scenario(
        name="Search by tag",
        turns=["Create a note tagged urgent.", "Show me urgent notes."],
        expected_tool_calls=["create_note", "search_notes"],
        expected_outcome=s4_check,
        tags=["happy_path"]
    ),
    Scenario(
        name="Search by date range",
        turns=["What did I write last week?"],
        expected_tool_calls=["search_notes"],
        expected_outcome=s5_check,
        tags=["happy_path"]
    ),
    Scenario(
        name="Update note body",
        turns=[
            "Create a note titled 'Standup' with body 'Tuesday'.",
            "Update the Standup note to say Wednesday instead.",
            "Yes"
        ],
        expected_tool_calls=["create_note", "update_note"],
        expected_outcome=s6_check,
        tags=["happy_path"]
    ),
    Scenario(
        name="Delete note",
        turns=[
            "Create a note about an old address.",
            "Delete the note about the old address.",
            "Yes"
        ],
        expected_tool_calls=["create_note", "delete_note", "delete_note"],
        expected_outcome=s7_check,
        tags=["happy_path"]
    ),
    Scenario(
        name="Ambiguous search",
        turns=[
            "Create a note titled 'API v1'.",
            "Create a note titled 'API v2'.",
            "Delete the API note."
        ],
        expected_tool_calls=["create_note", "create_note", "search_notes"],
        expected_outcome=s8_check,
        tags=["disambiguation"]
    ),
    Scenario(
        name="Follow-up that note",
        turns=[
            "Create a note about the project.",
            "Actually, add a deadline to that note."
        ],
        expected_tool_calls=["create_note", "update_note"],
        expected_outcome=s9_check,
        tags=["multi_turn"]
    ),
    Scenario(
        name="Delete non-existent note",
        turns=["Delete the note about aliens."],
        expected_tool_calls=["search_notes"],
        expected_outcome=s10_check,
        tags=["edge_case"]
    ),
    Scenario(
        name="Search returns 0 results",
        turns=["Find notes about aliens."],
        expected_tool_calls=["search_notes"],
        expected_outcome=s11_check,
        tags=["edge_case"]
    ),
    Scenario(
        name="Refuse deletion without confirmation",
        turns=[
            "Create a note to delete.",
            "Delete it."
        ],
        expected_tool_calls=["create_note", "delete_note"],
        expected_outcome=s12_check,
        tags=["edge_case"]
    ),
    Scenario(
        name="Summarise tagged notes",
        turns=[
            "Create note 1 tagged urgent.",
            "Create note 2 tagged urgent.",
            "Summarise everything tagged as urgent."
        ],
        expected_tool_calls=["create_note", "create_note", "search_notes", "answer_question"],
        expected_outcome=s13_check,
        tags=["reasoning"]
    ),
    Scenario(
        name="Contradiction detection",
        turns=[
            "Create note saying API is JSON.",
            "Create note saying API is XML.",
            "Do I have any contradictory notes about the API?"
        ],
        expected_tool_calls=["create_note", "create_note", "search_notes", "answer_question"],
        expected_outcome=s14_check,
        tags=["reasoning"]
    ),
    Scenario(
        name="Update via follow-up",
        turns=[
            "Create note about the design.",
            "Add the deadline tag to it."
        ],
        expected_tool_calls=["create_note", "update_note"],
        expected_outcome=s15_check,
        tags=["multi_turn"]
    ),
]
