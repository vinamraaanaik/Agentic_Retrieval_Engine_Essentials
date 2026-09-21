import re

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize_user_text(text: str, max_length: int = 4000) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    cleaned = _CONTROL_RE.sub(" ", text).strip()
    if not cleaned:
        raise ValueError("text cannot be empty")
    if len(cleaned) > max_length:
        raise ValueError(f"text exceeds maximum length of {max_length} characters")
    return cleaned
