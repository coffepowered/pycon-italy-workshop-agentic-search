"""Minimal local MCP server for PDF page retrieval.

We start with a single tool: `get_page_for_llm(pdf_path, page, mode)`:
- mode="text":  returns the page text.
- mode="image": returns the page as an inline PNG image.

Transport: stdio. No auth.
"""

import base64
import logging
from pathlib import Path
from typing import Literal, List

import pymupdf
from fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from services.grep_docs import GrepHit
# from services.grep_docs import grep_documents as _grep_documents
# from solutions.bm25 import BM25Index, Hit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

mcp = FastMCP("search-over-my-documents")

# Resolve paths relative to this file, not to the current working directory.
# That keeps the server usable whether it is launched from the repo root,
# through MCP, or from an IDE.
BASE_DIR = Path(__file__).resolve().parent
MCP_DOCS_DIR = BASE_DIR / "documents"  # corpus to index/search


@mcp.tool()
def get_page_for_llm(
    pdf_path: str,
    page: int,
    mode: Literal["text", "image"] = "text",
) -> TextContent | ImageContent:
    """Return a single PDF page either as text or as an inline image.

    Good to retrieve the full context where text fragments occur.
    Caller can choose text (cheaper) or image (for complex pages).

    Args:
        pdf_path: Absolute path to the PDF file.
        page: 1-indexed page number.
    """
    logger.info("get_page_for_llm: pdf=%s page=%d", pdf_path, page)
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = pymupdf.open(path)
    try:
        if page < 1 or page > doc.page_count:
            raise ValueError(f"page {page} out of range (1..{doc.page_count})")
        page_obj = doc.load_page(page - 1)
        if mode == "text":
            return TextContent(type="text", text=page_obj.get_text())  # type: ignore

        pix = page_obj.get_pixmap()
        encoded = base64.b64encode(pix.tobytes("png")).decode("ascii")
        return ImageContent(type="image", data=encoded, mimeType="image/png")
    finally:
        doc.close()


'''
# Exercise 1: warmup with grep
@mcp.tool()
def grep_documents(
    pattern: str,
    case_sensitive: bool = False,
    context_chars: int = 80,
    max_hits: int = 200,
) -> List[GrepHit]:
    """Regex-search documents in the folder set by `MCP_DOCS_DIR`.

    Scans .pdf (per-page, returns `page`) and .txt/.md (per-line, returns `line`).

    Args:
        pattern: Python `re` pattern.
        case_sensitive: If False, match case-insensitively.
        context_chars: Characters of context to include on each side of the match
            in `snippet`.
        max_hits: Cap on total hits returned.
    """
    logger.info(
        "grep_documents: pattern=%r case_sensitive=%s context_chars=%d max_hits=%d",
        pattern,
        case_sensitive,
        context_chars,
        max_hits,
    )

    root = Path(MCP_DOCS_DIR).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"MCP_DOCS_DIR is not a directory: {root}")

    return _grep_documents(
        pattern,
        root,
        case_sensitive=case_sensitive,
        context_chars=context_chars,
        max_hits=max_hits,
    )  # type: ignore
'''

'''
# Exercise 2: BM25 search
MCP_INDEX_DIR = BASE_DIR / "index"  # where the LanceDB BM25 index lives
_bm25 = BM25Index(db_path=MCP_INDEX_DIR, docs_dir=MCP_DOCS_DIR)

@mcp.tool()
def search_documents_bm25(
    query: str, k: int = 5, context_chars: int = 300
) -> List[Hit]:
    """Full-text (BM25) search over indexed PDF pages.

    Returns the top-k pages ranked by lexical relevance, each with a text
    `snippet`. To read a full result, call `get_page_for_llm(pdf_path,
    page)` on a hit. The index uses an Italian analyzer (stemming, stopwords,
    accent folding), so "acciuga" matches "acciughe" and "ragu" matches "ragù".

    Args:
        query: Natural-language or keyword query.
        k: Number of pages to return.
    """
    logger.info("search_documents_bm25: query=%r k=%d", query, k)
    return _bm25.search(query, k=k, snippet_chars=context_chars)
'''

'''
# Exercise 3: multimodal (ColQwen2.5 multi-vector / late-interaction) search
from solutions.indexer import ColQwenIndex, Hit as MultiVectorHit

MCP_INDEX_DIR = BASE_DIR / "index"     # reuse the LanceDB dir
_colqwen = ColQwenIndex(db_path=MCP_INDEX_DIR, docs_dir=MCP_DOCS_DIR)
@mcp.tool()
def search_documents_multimodal(query: str, k: int = 5, context_chars: int = 300) -> list[MultiVectorHit]:
    """Multimodal search over rendered PDF pages.
    Use this when need to reference/search tables, images and plots from documents.

    Returns the top-k pages ranked by visual+textual late-interaction
    similarity, each with a text `snippet`.
    To then *see* a result, call `get_page_for_llm(pdf_path, page, mode="image")`.

    Requires the index to be built first:
    `uv run python -m solutions.indexer documents`.

    Args:
        query: Natural-language query.
        k: Number of pages to return.
    """
    logger.info("search_documents_multimodal: query=%r k=%d", query, k)
    return _colqwen.search(query, k=k, snippet_chars=context_chars)
'''


if __name__ == "__main__":
    logger.info("starting MCP server (MCP_DOCS_DIR=%s)", MCP_DOCS_DIR)
    mcp.run()
