"""Multimodal (ColQwen2.5) indexing + search over PDF pages, backed by LanceDB.

Pure library (no MCP). Mirrors the shape of `bm25.BM25Index`: the *same*
object both builds the index and searches it.

- as an import (`ColQwenIndex` class) — e.g. from the MCP server;
- as a CLI: `uv run python -m solutions.indexer documents`.

One row per *visual* PDF page (text-only pages are skipped — they're cheap to
serve with the BM25 tool). A search hit composes 1:1 with the existing
`get_page_for_llm(pdf_path, page)` tool.

Schema:
    pages  : id, pdf_path, pdf_name, page, embedding (list<vector[128]>), indexed_at
    files  : path, mtime, size, n_pages, indexed_at  (dedup table)

The teaching point: search needs *no* ANN index here. LanceDB does an exact
flat (brute-force) MaxSim scan when no vector index exists — exact ground-truth
results, ideal for a small corpus. `build_search_index()` builds an on-disk
IVF_PQ index, but that only pays off at scale (thousands of pages) and is left
opt-in.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path
from typing import Iterable

import lancedb
import numpy as np
import pyarrow as pa
import pymupdf
from pydantic import BaseModel

from solutions.colqwen import EmbeddingModel, render_pdf_page

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("./index")
EMBEDDING_DIM = 128

# Persist embeddings to LanceDB every this many pages. Smaller → less work lost
# on interrupt (resume is page-granular); larger → fewer, bigger writes.
_FLUSH_EVERY = 8


# ---------- public models ----------

class Hit(BaseModel):
    pdf_path: str
    page: int
    score: float  # MaxSim cosine; higher is more relevant
    snippet: str


class BuildReport(BaseModel):
    files_seen: int
    files_indexed: int
    files_skipped: int
    pages_indexed: int
    pages_skipped_text_only: int = 0
    index_built: bool = False
    duration_s: float
    errors: list[tuple[str, str]] = []


# ---------- schemas ----------

_FILES_SCHEMA = pa.schema([
    pa.field("path", pa.string()),
    pa.field("mtime", pa.float64()),
    pa.field("size", pa.int64()),
    pa.field("n_pages", pa.int32()),
    pa.field("indexed_at", pa.timestamp("ms")),
])

_PAGES_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("pdf_path", pa.string()),
    pa.field("pdf_name", pa.string()),
    pa.field("page", pa.int32()),
    pa.field("embedding", pa.list_(pa.list_(pa.float16(), EMBEDDING_DIM))),
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


def _mx_to_list_of_vectors(emb) -> list[list[float]]:
    """Convert an mx.array of shape [n_tokens, dim] to a list of dim-length lists."""
    arr = np.asarray(emb, dtype=np.float16)
    return arr.tolist()


def _snippet(pdf_path: str, page: int, max_chars: int) -> str:
    """Load a page's text on the fly via PyMuPDF (snippets are not stored)."""
    if max_chars <= 0:
        return ""
    doc = pymupdf.open(pdf_path)
    try:
        text = doc.load_page(page - 1).get_text() or ""
    finally:
        doc.close()
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# A drawing thinner than this (points) on either axis is a hairline — a rule,
# underline or index leader-line, not real graphics. We ignore those.
_MIN_DRAW_DIM = 3.0
# A page is "visual" via drawings only when it has more than this many
# *significant* (non-hairline) shapes — enough to be a chart/diagram, not chrome.
_SIGNIFICANT_DRAWINGS = 5

# A page counts as "visual" only if a rendered image covers at least this
# fraction of the page — filters out logos, bullets and rules.
_MIN_IMAGE_AREA_FRAC = 0.05


def _page_is_visual(page: pymupdf.Page) -> bool:
    """Heuristic: True if the page contains a real image, a table, or a vector diagram.

    Pure-text pages are skipped from the multimodal index — they're cheap to
    serve with the BM25 tool later and the vision embedding wastes compute on them.

    Use `get_image_info()` (images actually *rendered* on the page), NOT
    `get_images()`: the latter lists shared/inherited resource XObjects and
    reports the same count on every page even when nothing is drawn.
    """
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    for im in page.get_image_info():
        x0, y0, x1, y1 = im["bbox"]
        if abs((x1 - x0) * (y1 - y0)) >= _MIN_IMAGE_AREA_FRAC * page_area:
            return True
    try:
        if list(page.find_tables()):
            return True
    except Exception:
        # find_tables can be flaky on some PDFs; treat as "no table"
        pass
    # Count only non-hairline shapes. A raw `len(get_drawings())` flags a text
    # index page with ~25 ruling/leader lines as "visual"; a real diagram instead
    # has many shapes with both width AND height.
    significant = sum(
        1 for d in page.get_drawings()
        if d["rect"].width >= _MIN_DRAW_DIM and d["rect"].height >= _MIN_DRAW_DIM
    )
    return significant > _SIGNIFICANT_DRAWINGS


# ---------- index ----------

class ColQwenIndex:
    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        docs_dir: Path | str = "documents",
        model: EmbeddingModel | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.docs_dir = Path(docs_dir)  # default corpus for build()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self.model = model or EmbeddingModel()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        names = set(self.db.list_tables().tables)
        if "files" not in names:
            self.db.create_table("files", schema=_FILES_SCHEMA)
        if "pages" not in names:
            self.db.create_table("pages", schema=_PAGES_SCHEMA)
        self.files = self.db.open_table("files")
        self.pages = self.db.open_table("pages")

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

    def _has_file_row(self, pdf: Path) -> bool:
        df = (
            self.files.search()
            .where(f"path = '{pdf}'", prefilter=True)
            .limit(1)
            .to_pandas()
        )
        return not df.empty

    def _indexed_pages(self, pdf: Path, n_pages: int) -> set[int]:
        """Page numbers already embedded for this PDF (for resuming)."""
        df = (
            self.pages.search()
            .where(f"pdf_path = '{pdf}'", prefilter=True)
            .limit(n_pages)  # explicit: LanceDB defaults to 10 otherwise
            .select(["page"])
            .to_pandas()
        )
        return {int(p) for p in df["page"]} if not df.empty else set()

    def _index_one_pdf(
        self, pdf: Path, *, fresh: bool, max_pages: int | None = None
    ) -> tuple[int, int]:
        """Index a single PDF, page by page. Returns (pages_embedded_now, pages_skipped_text_only).

        Resumable: embeddings are flushed every `_FLUSH_EVERY` pages (and on the
        way out, so an interrupt keeps what was done), and pages already present
        from a previous run are skipped. `fresh=True` wipes existing rows first
        (forced re-index, or the file changed since it was last completed).

        `max_pages` caps the number of *visual* pages embedded per document
        (counting any already embedded on a resume) — useful to keep very long
        PDFs cheap.
        """
        doc = pymupdf.open(pdf)
        try:
            n_pages = doc.page_count
        finally:
            doc.close()

        if fresh:
            self.pages.delete(f"pdf_path = '{pdf}'")
            done: set[int] = set()
        else:
            done = self._indexed_pages(pdf, n_pages)
            if done:
                logger.info("resuming %s: %d page(s) already embedded", pdf.name, len(done))

        now = dt.datetime.now()
        buffer: list[dict] = []
        indexed_now = 0
        skipped_text_only = 0

        def flush() -> None:
            nonlocal buffer
            if buffer:
                self.pages.add(buffer)
                buffer = []

        doc = pymupdf.open(pdf)
        try:
            for page_num in range(1, n_pages + 1):
                if page_num in done:
                    continue  # already embedded in a previous run
                page = doc.load_page(page_num - 1)
                if not _page_is_visual(page):
                    skipped_text_only += 1
                    logger.info("  page %d/%d skipped (text-only)", page_num, n_pages)
                    continue
                if max_pages is not None and page_num >= max_pages:
                    logger.info("  reached max-pages-per-doc=%d, stopping %s",
                                max_pages, pdf.name)
                    break
                img = render_pdf_page(pdf, page_num)
                emb = self.model.embed_image(img)
                buffer.append({
                    "id": f"{pdf}#{page_num}",
                    "pdf_path": str(pdf),
                    "pdf_name": pdf.name,
                    "page": page_num,
                    "embedding": _mx_to_list_of_vectors(emb),
                    "indexed_at": now,
                })
                indexed_now += 1
                logger.info("  page %d/%d embedded", page_num, n_pages)
                if len(buffer) >= _FLUSH_EVERY:
                    flush()
        finally:
            # persist whatever we embedded — including on KeyboardInterrupt
            flush()
            doc.close()

        # Mark the file complete only after every page is persisted. If we were
        # interrupted above, this line never runs, so the next invocation finds
        # no `files` row and resumes.
        self.files.delete(f"path = '{pdf}'")
        self.files.add([{
            "path": str(pdf),
            "mtime": pdf.stat().st_mtime,
            "size": pdf.stat().st_size,
            "n_pages": n_pages,
            "indexed_at": now,
        }])
        return indexed_now, skipped_text_only

    def build(
        self,
        path: str | Path | None = None,
        *,
        force: bool = False,
        max_pages_per_doc: int | None = None,
    ) -> BuildReport:
        """Ingest PDFs under `path` (default: `self.docs_dir`) into the `pages` table.

        Embeds every *visual* page with ColQwen2.5. Idempotent: unchanged files
        (same mtime+size) are skipped unless `force=True`, and a PDF interrupted
        mid-way resumes from the next un-embedded page. `max_pages_per_doc` caps
        the visual pages embedded per PDF (e.g. to keep long books cheap). No ANN
        index is built here — `search()` does an exact flat scan, which is plenty
        for a small corpus. Call `build_search_index()` separately only at scale.
        """
        target = Path(path if path is not None else self.docs_dir).expanduser().resolve()
        t0 = time.perf_counter()
        seen = indexed = skipped = pages_total = pages_skipped_text_only = 0
        errors: list[tuple[str, str]] = []

        for pdf in _iter_pdfs(target):
            seen += 1
            stat = pdf.stat()
            if not force and self._is_already_indexed(pdf, stat.st_mtime, stat.st_size):
                skipped += 1
                logger.info("skip (unchanged): %s", pdf)
                continue
            # Wipe first only if forced, or a completed row exists but the file
            # changed (mtime/size differ — we wouldn't be here otherwise). No row
            # means a prior run was interrupted: resume instead of wiping.
            fresh = force or self._has_file_row(pdf)
            try:
                logger.info("indexing: %s", pdf)
                n_indexed, n_skipped = self._index_one_pdf(
                    pdf, fresh=fresh, max_pages=max_pages_per_doc
                )
                pages_total += n_indexed
                pages_skipped_text_only += n_skipped
                indexed += 1
            except Exception as e:  # one bad PDF must not abort the batch
                logger.exception("error indexing %s", pdf)
                errors.append((str(pdf), repr(e)))

        return BuildReport(
            files_seen=seen,
            files_indexed=indexed,
            files_skipped=skipped,
            pages_indexed=pages_total,
            pages_skipped_text_only=pages_skipped_text_only,
            duration_s=time.perf_counter() - t0,
            errors=errors,
        )

    def build_search_index(self, *, replace: bool = False) -> None:
        """Create the on-disk multi-vector IVF_PQ index over `pages.embedding`.

        Optional and scale-only: IVF_PQ needs a few thousand rows to train
        usefully. For a workshop-sized corpus, skip this — `search()` falls back
        to an exact flat scan, which is both faster to set up and exact.
        """
        if not replace:
            try:
                existing = self.pages.list_indices()
                if existing:
                    logger.info("index already present: %s", existing)
                    return
            except Exception:
                pass
        n = self.pages.count_rows()
        logger.info("building multi-vector index over %d pages …", n)
        t0 = time.perf_counter()
        # LanceDB only supports cosine for multi-vector columns; ColQwen2.5
        # embeddings are already L2-normalized, so cosine == dot product.
        self.pages.create_index(
            metric="cosine",
            vector_column_name="embedding",
            replace=replace,
        )
        logger.info("index built in %.1fs", time.perf_counter() - t0)

    def search(self, query: str, k: int = 5, snippet_chars: int = 300) -> list[Hit]:
        """Multi-vector (MaxSim) search over indexed pages via LanceDB native search.

        With no IVF_PQ index present this is an exact flat scan — exact
        ground-truth top-k, ideal for a small corpus. Page text snippets are
        loaded on the fly via PyMuPDF.
        """
        if not query.strip() or self.pages.count_rows() == 0:
            return []

        self.model.load()
        q_emb = np.asarray(self.model.embed_query(query), dtype=np.float16)
        df = (
            self.pages.search(q_emb)
            .limit(k)
            # include `_distance` explicitly: LanceDB will stop auto-adding it to
            # projected queries in a future release.
            .select(["pdf_path", "page", "_distance"])
            .to_pandas()
        )

        hits: list[Hit] = []
        for _, row in df.iterrows():
            # cosine `_distance`: lower is better. Expose -_distance as score so
            # "higher is better" matches the BM25 tool's ergonomics.
            dist = float(row.get("_distance", 0.0))
            hits.append(
                Hit(
                    pdf_path=row["pdf_path"],
                    page=int(row["page"]),
                    score=-dist,
                    snippet=_snippet(row["pdf_path"], int(row["page"]), snippet_chars),
                )
            )
        return hits


# ---------- CLI ----------

def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="documents",
                        help="PDF file or directory to index (default: documents)")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="LanceDB path (default: ./index)")
    parser.add_argument("--force", action="store_true", help="re-index even if unchanged")
    parser.add_argument(
        "--max-pages-per-doc",
        type=int,
        default=None,
        help="cap visual pages embedded per document (e.g. 20) to keep long PDFs cheap",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="after ingestion, build the on-disk IVF_PQ index (only worth it at scale)",
    )
    parser.add_argument("--query", help="run a multimodal query")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="skip indexing and just query the existing index (requires --query)",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--snippet", type=int, default=300, help="snippet char length (0 to disable)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.no_build and not args.query:
        parser.error("--no-build only makes sense together with --query")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    # -v sets the *root* logger to INFO, which would also dump httpx/HF HTTP
    # chatter (the harmless 404→307→200 probing for processor_config.json).
    # Keep our own per-page progress, silence theirs.
    for noisy in ("httpx", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # transformers emits a WARNING because the checkpoint's model_type
    # ("colqwen2_5") isn't registered in the installed transformers — cosmetic
    # here, since the weights load via MLX, not transformers. Drop to ERROR.
    logging.getLogger("transformers").setLevel(logging.ERROR)

    index = ColQwenIndex(db_path=args.db)

    if not args.no_build:
        report = index.build(args.path, force=args.force, max_pages_per_doc=args.max_pages_per_doc)
        print("=== build report ===")
        print(report.model_dump_json(indent=2))
        if args.build_index:
            index.build_search_index(replace=True)

    if args.query:
        print(f"\nquery: {args.query!r}\n")
        for i, hit in enumerate(index.search(args.query, k=args.k, snippet_chars=args.snippet), 1):
            print(f"{i:>2}. {Path(hit.pdf_path).name}  page {hit.page}  score={hit.score:.3f}")
            print(f"    {hit.pdf_path}")
            if hit.snippet:
                print(f"    {hit.snippet}")
            print()


if __name__ == "__main__":
    _main()
