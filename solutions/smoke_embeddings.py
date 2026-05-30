"""Manual smoke test for the embedding layer. Prints progress + timings.

Usage:
    uv run python smoke_embeddings.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pymupdf

from solutions.colqwen import EmbeddingModel, render_pdf_page


def make_test_pdf(path: Path) -> None:
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 100), "hello world", fontsize=48)
    p1.insert_text((72, 180), "the quick brown fox jumps over the lazy dog", fontsize=18)
    p2 = doc.new_page()
    p2.insert_text((72, 100), "lorem ipsum", fontsize=48)
    p2.insert_text((72, 180), "dolor sit amet consectetur adipiscing elit", fontsize=18)
    doc.save(path)
    doc.close()


def stage(label: str) -> float:
    print(f"-> {label} ...", flush=True)
    return time.perf_counter()


def done(t0: float) -> None:
    print(f"   done in {time.perf_counter() - t0:.2f}s", flush=True)


def main() -> None:
    pdf_path = Path("/tmp/_smoke_embed.pdf")
    make_test_pdf(pdf_path)

    t = stage("instantiate EmbeddingModel (no load yet)")
    m = EmbeddingModel()
    done(t)
    print(f"   model name: {m.model_name}", flush=True)

    t = stage("load model")
    m.load()
    done(t)

    t = stage("render page 1")
    img1 = render_pdf_page(pdf_path, 1)
    done(t)

    t = stage("render page 2")
    img2 = render_pdf_page(pdf_path, 2)
    done(t)

    t = stage("embed page 1")
    p1 = m.embed_image(img1)
    done(t)
    print(f"   shape={tuple(p1.shape)} dtype={p1.dtype}", flush=True)

    t = stage("embed page 2")
    p2 = m.embed_image(img2)
    done(t)

    t = stage("embed query 'hello world'")
    q_hello = m.embed_query("hello world")
    done(t)

    t = stage("embed query 'lorem ipsum'")
    q_lorem = m.embed_query("lorem ipsum")
    done(t)

    t = stage("score matrix (MaxSim)")
    s_hello_p1 = m.score(q_hello, p1)
    s_hello_p2 = m.score(q_hello, p2)
    s_lorem_p1 = m.score(q_lorem, p1)
    s_lorem_p2 = m.score(q_lorem, p2)
    done(t)

    print()
    print("           page1   page2")
    print(f"  hello   {s_hello_p1:6.3f}  {s_hello_p2:6.3f}   (expect p1 > p2)")
    print(f"  lorem   {s_lorem_p1:6.3f}  {s_lorem_p2:6.3f}   (expect p2 > p1)")
    print()
    ok = s_hello_p1 > s_hello_p2 and s_lorem_p2 > s_lorem_p1
    print("RESULT:", "OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
