"""Grep-like search over a documents folder.
"""

from pathlib import Path
from pydantic import BaseModel

GREP_SUFFIXES = {".pdf", ".txt", ".md"}


class GrepHit(BaseModel):
    path: str  # absolute path of the matching file
    page: int | None = None  # 1-indexed page number, set for PDFs
    line: int | None = None  # 1-indexed line number, set for .txt/.md
    match: str  # the actual matched substring
    snippet: str  # match with up to `context_chars` chars on each side


def grep_documents(
    pattern: str,
    root: Path,
    *,
    case_sensitive: bool = False,
    context_chars: int = 80,
    max_hits: int = 200,
) -> list[GrepHit]:
    """Regex-search documents under `root`. See module docstring for hints."""

    raise NotImplementedError("workshop: implement grep_documents")
