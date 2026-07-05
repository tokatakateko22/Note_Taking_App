"""Case-insensitive text matching helpers for notes search and filtering."""


def normalize(text: str) -> str:
    return text.casefold()


def tags_match(note_tags: list[str], filter_tags: list[str]) -> bool:
    """Return True if any filter tag matches any note tag (case-insensitive, OR logic)."""
    if not filter_tags:
        return True
    note_tags_lower = {normalize(t) for t in note_tags}
    return any(normalize(t) in note_tags_lower for t in filter_tags)


def category_matches(note_category: str | None, filter_category: str | None) -> bool:
    if filter_category is None:
        return True
    if note_category is None:
        return False
    return normalize(note_category) == normalize(filter_category)


def text_contains(haystack: str, needle: str) -> bool:
    return normalize(needle) in normalize(haystack)
