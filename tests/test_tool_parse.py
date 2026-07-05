from src.tool_parse import extract_tool_calls_from_text, strip_tool_json_from_text, parse_tool_arguments


def test_parse_tool_arguments_null():
    assert parse_tool_arguments(None) == {}
    assert parse_tool_arguments("") == {}
    assert parse_tool_arguments("{}") == {}
    assert parse_tool_arguments('{"limit": 10}') == {"limit": 10}


def test_extract_tool_call_from_embedded_json():
    text = (
        "It seems like the limit parameter is not correctly set. "
        'Let me try again with a smaller limit.\n\n'
        '{"name": "list_notes", "parameters": {"limit":"10"}}'
    )
    calls = extract_tool_calls_from_text(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["name"] == "list_notes"
    assert calls[0]["arguments"]["limit"] == "10"

    cleaned = strip_tool_json_from_text(text)
    assert cleaned is not None
    assert "list_notes" not in cleaned
    assert "limit parameter" in cleaned
