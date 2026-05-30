"""BM25 (full-text) indexing over PDF pages, backed by LanceDB's native FTS.

Pure library (no MCP). Can be used:
- as an import (BM25Index class) — e.g. from the MCP server;
- as a CLI: `uv run python solutions/bm25_indexer.py documents`.

One row per PDF page, so a search hit composes 1:1 with the existing
`get_page_for_llm(pdf_path, page)` tool.

Schema (LanceDB table `pages_text`):
    id, pdf_path, pdf_name, page, text, indexed_at
Dedup table (`files`):
    path, mtime, size, n_pages, indexed_at
"""
import argparse
import datetime as dt
import logging
import time
from pathlib import Path
from typing import Iterable

import lancedb
import pyarrow as pa
import pymupdf
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("./index")
PAGES_TABLE = "pages_text"
FILES_TABLE = "files_text"

# Language analyzer for the FTS index. "Italian" turns on Italian stemming and
# stopwords; accent folding ("ragù" -> "ragu") is on by default. Set to
# "English" (or None) to feel how much recall the analyzer buys you.
FTS_LANGUAGE = "English"

# Pages with less text than this (after stripping) are skipped — usually blank
# pages or scanned images with no extractable text, which add only noise.
_MIN_PAGE_CHARS = 20


# ---------- public models ----------

class Hit(BaseModel):
    pdf_path: str
    page: int
    score: float  # BM25 score; higher is more relevant
    snippet: str


class BuildReport(BaseModel):
    files_seen: int
    files_indexed: int
    files_skipped: int
    pages_indexed: int
    pages_skipped_empty: int = 0
    index_rebuilt: bool = False
    duration_s: float
    errors: list[tuple[str, str]] = []


# ---------- schemas ----------

_PAGES_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("pdf_path", pa.string()),
    pa.field("pdf_name", pa.string()),
    pa.field("page", pa.int32()),
    pa.field("text", pa.string()),
    pa.field("indexed_at", pa.timestamp("ms")),
])

_FILES_SCHEMA = pa.schema([
    pa.field("path", pa.string()),
    pa.field("mtime", pa.float64()),
    pa.field("size", pa.int64()),
    pa.field("n_pages", pa.int32()),
    pa.field("indexed_at", pa.timestamp("ms")),
])


# ---------- helpers ----------

def _iter_pdfs(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"not a PDF: {path}")
        yield path
        return
    if path.is_dir():
        yield from sorted(path.rglob("*.pdf"))
        return
    raise FileNotFoundError(f"not found: {path}")


def _snippet(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ---------- index ----------

class BM25Index:
    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        docs_dir: Path | str = "documents",
    ) -> None:
        self.db_path = Path(db_path)
        self.docs_dir = Path(docs_dir)  # default corpus for build()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        names = set(self.db.list_tables().tables)
        if PAGES_TABLE not in names:
            self.db.create_table(PAGES_TABLE, schema=_PAGES_SCHEMA)
        if FILES_TABLE not in names:
            self.db.create_table(FILES_TABLE, schema=_FILES_SCHEMA)
        self.pages = self.db.open_table(PAGES_TABLE)
        self.files = self.db.open_table(FILES_TABLE)

    def _is_already_indexed(self, path: Path, mtime: float, size: int) -> bool:
        df = (
            self.files.search()
            .where(f"path = '{path}'", prefilter=True)
            .limit(1)
            .to_pandas()
        )
        if df.empty:
            return False
        row = df.iloc[0]
        return float(row["mtime"]) == mtime and int(row["size"]) == size

    def _index_one_pdf(self, pdf: Path) -> tuple[int, int]:
        """Extract per-page text for one PDF. Returns (pages_indexed, pages_skipped)."""
        # idempotent: drop any previous rows for this file
        self.pages.delete(f"pdf_path = '{pdf}'")

        now = dt.datetime.now()
        records: list[dict] = []
        skipped = 0

        doc = pymupdf.open(pdf)
        try:
            n_pages = doc.page_count
            for page_num in range(1, n_pages + 1):
                text = doc.load_page(page_num - 1).get_text() or ""
                if len(text.strip()) < _MIN_PAGE_CHARS:
                    skipped += 1
                    continue
                records.append({
                    "id": f"{pdf}#{page_num}",
                    "pdf_path": str(pdf),
                    "pdf_name": pdf.name,
                    "page": page_num,
                    "text": text,
                    "indexed_at": now,
                })
        finally:
            doc.close()

        if records:
            self.pages.add(records)

        self.files.delete(f"path = '{pdf}'")
        self.files.add([{
            "path": str(pdf),
            "mtime": pdf.stat().st_mtime,
            "size": pdf.stat().st_size,
            "n_pages": n_pages,
            "indexed_at": now,
        }])
        return len(records), skipped

    def _rebuild_fts_index(self) -> None:
        """(Re)create the BM25 full-text index over the `text` column.

        Native Lance FTS does not auto-include rows added after the index was
        built, so we rebuild (replace=True) whenever the corpus changed.
        """
        logger.info("building BM25 index (language=%s) over %d pages …",
                    FTS_LANGUAGE, self.pages.count_rows())
        t0 = time.perf_counter()
        self.pages.create_fts_index(
            "text",
            use_tantivy=False,      # native Lance FTS — no extra dependency
            language=FTS_LANGUAGE,  # Italian stemming + stopwords
            stem=True,
            remove_stop_words=True,
            ascii_folding=True,     # ragù ~ ragu, è ~ e
            lower_case=True,
            replace=True,
        )
        logger.info("BM25 index built in %.2fs", time.perf_counter() - t0)

    def build(self, path: str | Path | None = None, *, force: bool = False) -> BuildReport:
        target = Path(path if path is not None else self.docs_dir).expanduser().resolve()
        t0 = time.perf_counter()
        seen = indexed = skipped = pages_total = pages_skipped = 0
        errors: list[tuple[str, str]] = []

        for pdf in _iter_pdfs(target):
            seen += 1
            stat = pdf.stat()
            if not force and self._is_already_indexed(pdf, stat.st_mtime, stat.st_size):
                skipped += 1
                logger.info("skip (unchanged): %s", pdf)
                continue
            try:
                logger.info("indexing: %s", pdf)
                n_indexed, n_skipped = self._index_one_pdf(pdf)
                pages_total += n_indexed
                pages_skipped += n_skipped
                indexed += 1
            except Exception as e:  # one bad PDF must not abort the batch
                logger.exception("error indexing %s", pdf)
                errors.append((str(pdf), repr(e)))

        index_rebuilt = False
        if indexed and self.pages.count_rows():
            self._rebuild_fts_index()
            index_rebuilt = True

        return BuildReport(
            files_seen=seen,
            files_indexed=indexed,
            files_skipped=skipped,
            pages_indexed=pages_total,
            pages_skipped_empty=pages_skipped,
            index_rebuilt=index_rebuilt,
            duration_s=time.perf_counter() - t0,
            errors=errors,
        )

    def search(self, query: str, k: int = 5, snippet_chars: int = 300) -> list[Hit]:
        if not query.strip() or self.pages.count_rows() == 0:
            return []
        df = (
            self.pages.search(query, query_type="fts")
            .limit(k)
            .select(["pdf_path", "page", "text", "_score"])
            .to_pandas()
        )
        return [
            Hit(
                pdf_path=row["pdf_path"],
                page=int(row["page"]),
                score=float(row["_score"]),
                snippet=_snippet(row["text"], snippet_chars),
            )
            for _, row in df.iterrows()
        ]


# ---------- CLI ----------

def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="documents",
                        help="PDF file or directory to index (default: documents)")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--force", action="store_true", help="re-index even if unchanged")
    parser.add_argument("--query", help="run a BM25 query after indexing")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    index = BM25Index(db_path=args.db)
    report = index.build(args.path, force=args.force)
    print("=== build report ===")
    print(report.model_dump_json(indent=2))

    if args.query:
        print(f"\nquery: {args.query!r}\n")
        for i, hit in enumerate(index.search(args.query, k=args.k), 1):
            print(f"{i:>2}. {Path(hit.pdf_path).name}  page {hit.page}  score={hit.score:.3f}")
            print(f"    {hit.snippet}\n")


if __name__ == "__main__":
    _main()
