"""Tests for the embedding layer (MLX / ColQwen2.5)."""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pymupdf
import pytest

from solutions.colqwen import EmbeddingModel, render_pdf_page


@pytest.fixture(scope="module")
def test_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("pdfs") / "test.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 100), "hello world", fontsize=48)
    p1.insert_text((72, 180), "the quick brown fox jumps over the lazy dog", fontsize=18)
    p2 = doc.new_page()
    p2.insert_text((72, 100), "lorem ipsum", fontsize=48)
    p2.insert_text((72, 180), "dolor sit amet consectetur adipiscing elit", fontsize=18)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    m = EmbeddingModel()
    m.load()
    return m


def test_embedding_shape(model: EmbeddingModel, test_pdf: Path) -> None:
    img = render_pdf_page(test_pdf, page=1)
    emb = model.embed_image(img)
    assert isinstance(emb, mx.array)
    assert emb.ndim == 2, f"expected 2D tensor, got shape {emb.shape}"
    assert emb.shape[0] >= 16, f"too few tokens: {emb.shape}"
    assert emb.shape[1] >= 64, f"embedding dim suspiciously small: {emb.shape}"
    assert not mx.any(mx.isnan(emb)).item(), "NaN in embedding"
    assert mx.sum(mx.abs(emb)).item() > 0, "all-zero embedding"


def test_maxsim_sanity(model: EmbeddingModel, test_pdf: Path) -> None:
    img1 = render_pdf_page(test_pdf, page=1)
    img2 = render_pdf_page(test_pdf, page=2)
    p1 = model.embed_image(img1)
    p2 = model.embed_image(img2)
    q_hello = model.embed_query("hello world")
    q_lorem = model.embed_query("lorem ipsum")

    s_hello_p1 = model.score(q_hello, p1)
    s_hello_p2 = model.score(q_hello, p2)
    s_lorem_p1 = model.score(q_lorem, p1)
    s_lorem_p2 = model.score(q_lorem, p2)

    assert s_hello_p1 > s_hello_p2, (
        f"'hello world' should match page 1: p1={s_hello_p1:.3f} p2={s_hello_p2:.3f}"
    )
    assert s_lorem_p2 > s_lorem_p1, (
        f"'lorem ipsum' should match page 2: p1={s_lorem_p1:.3f} p2={s_lorem_p2:.3f}"
    )
