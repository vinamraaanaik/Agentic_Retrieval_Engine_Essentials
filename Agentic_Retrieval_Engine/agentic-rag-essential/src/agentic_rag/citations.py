import re

CITATION_RE = re.compile(r"\[SRC-[0-9a-f]{10}\]")


def extract_citations(text: str) -> set[str]:
    return set(CITATION_RE.findall(text or ""))
