import pytest
import tempfile
import os
from src.database import Database
from src.context import ConversationContext
from src.tools import NoteTools
from src.tool_schemas import CreateNoteInput, DeleteNoteInput, GetNoteInput, UpdateNoteInput, SearchNotesInput, ListNotesInput

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp()
    os.close(fd)
    db = Database(path)
    yield db
    db.close()
    try:
        os.unlink(path)
    except PermissionError:
        pass

@pytest.fixture
def note_tools(temp_db):
    context = ConversationContext(active_user_id="test_user")
    return NoteTools(temp_db, context)

def test_create_note(note_tools):
    result = note_tools.create_note(CreateNoteInput(
        title="Test Note",
        body="This is a test note body",
        tags=["test"]
    ))
    assert result.success is True
    assert result.data["title"] == "Test Note"
    assert note_tools.context.last_note_id == result.data["id"]

def test_delete_note_two_phase(note_tools):
    # 1. Create a note
    res_create = note_tools.create_note(CreateNoteInput(title="To Delete", body="foo"))
    note_id = res_create.data["id"]

    # 2. First delete call (should require confirmation)
    res_del1 = note_tools.delete_note(DeleteNoteInput(note_id=note_id))
    assert res_del1.success is True
    assert res_del1.requires_confirmation is True
    assert res_del1.confirmation_token is not None

    # 3. Second delete call (with token)
    res_del2 = note_tools.delete_note(DeleteNoteInput(
        note_id=note_id,
        confirmation_token=res_del1.confirmation_token
    ))
    assert res_del2.success is True
    assert res_del2.data["deleted"] is True

    # 4. Verify it's gone
    res_get = note_tools.get_note(GetNoteInput(note_id=note_id))
    assert res_get.success is False

def test_update_note(note_tools):
    # 1. Create
    res_create = note_tools.create_note(CreateNoteInput(title="Old", body="old body"))
    note_id = res_create.data["id"]

    # 2. Update
    res_upd = note_tools.update_note(UpdateNoteInput(note_id=note_id, title="New"))
    assert res_upd.success is True
    assert res_upd.data["title"] == "New"
    assert res_upd.data["body"] == "old body"  # unchanged

def test_tag_search_case_insensitive(note_tools):
    note_tools.create_note(CreateNoteInput(title="Shopping", body="Buy milk", tags=["Urgent"]))
    result = note_tools.search_notes(SearchNotesInput(tags=["urgent"]))
    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["title"] == "Shopping"

def test_list_notes_tag_case_insensitive(note_tools):
    note_tools.create_note(CreateNoteInput(title="Work item", body="Finish report", tags=["Meetings"]))
    result = note_tools.list_notes(ListNotesInput(tags=["meetings"]))
    assert result.success is True
    assert len(result.data) == 1

def test_keyword_search_case_insensitive(note_tools):
    note_tools.create_note(CreateNoteInput(title="API Notes", body="Details about the api endpoint", tags=[]))
    result = note_tools.search_notes(SearchNotesInput(query="api"))
    assert result.success is True
    assert len(result.data) == 1
