"""ColQwen2.5 multimodal embeddings via MLX.

No transformers/MPS path. Uses the pre-quantized MLX checkpoint and
mlx-embeddings' low-level API, following the pattern from the official
mlx-embeddings README.
"""

import logging
import threading
from pathlib import Path

import mlx.core as mx
import pymupdf
from mlx_embeddings import load
from mlx_embeddings.models.base import normalize_embeddings
from PIL import Image
from transformers import AutoImageProcessor

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qnguyen3/colqwen2_5-v0.2-mlx-4bit"


def render_pdf_page(path: Path | str, page: int, zoom: float = 1.0) -> Image.Image:
    """Render a PDF page (1-indexed) as a PIL RGB image."""
    p = Path(path)
    doc = pymupdf.open(p)
    try:
        if page < 1 or page > doc.page_count:
            raise ValueError(f"page {page} out of range (1..{doc.page_count})")
        pix = doc.load_page(page - 1).get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def _nonpad_rows(embeds: mx.array, attention_mask: mx.array) -> mx.array:
    """Drop padding rows from a [1, n_tokens, dim] embedding tensor."""
    indices = [i for i, v in enumerate(attention_mask[0].tolist()) if v != 0]
    return embeds[0, indices, :]


class EmbeddingModel:
    """Lazy-loading wrapper around an MLX ColQwen2.5 checkpoint."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._lock = threading.Lock()  # guards the one-time load()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        # Double-checked locking: the fast path (already loaded) takes no lock;
        # concurrent first calls (FastMCP runs sync tools in a thread pool) block
        # on the lock and the re-check ensures the weights load exactly once.
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            logger.info("loading %s (MLX)", self.model_name)
            self._model, self._tokenizer = load(self.model_name)
            self._image_processor = AutoImageProcessor.from_pretrained(self.model_name)

    def _prepare_query(self, text: str) -> dict[str, mx.array]:
        suffix = self._tokenizer.pad_token * 10
        query = "Query: " + text + suffix
        inputs = self._tokenizer([query], return_tensors="np", padding=True)
        return {
            "input_ids": mx.array(inputs["input_ids"]),
            "attention_mask": mx.array(inputs["attention_mask"]),
        }

    def _prepare_image(self, image: Image.Image) -> dict[str, mx.array]:
        image_inputs = self._image_processor(
            images=[image],
            return_tensors="np",
            data_format="channels_first",
            do_convert_rgb=True,
        )
        image_grid_thw = mx.array(image_inputs["image_grid_thw"])
        num_image_tokens = int(
            image_inputs["image_grid_thw"][0].prod()
            // (self._image_processor.merge_size ** 2)
        )
        prompt = (
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "Describe the image.<|im_end|><|endoftext|>"
        )
        prompt = prompt.replace("<|image_pad|>", "<|image_pad|>" * num_image_tokens)
        text_inputs = self._tokenizer([prompt], return_tensors="np", padding=True)
        return {
            "input_ids": mx.array(text_inputs["input_ids"]),
            "attention_mask": mx.array(text_inputs["attention_mask"]),
            "pixel_values": mx.array(image_inputs["pixel_values"]),
            "image_grid_thw": image_grid_thw,
        }

    def embed_query(self, text: str) -> mx.array:
        """Multi-vector embedding for a text query. Returns [n_tokens, dim] mx.array."""
        self.load()
        inputs = self._prepare_query(text)
        inputs_embeds = self._model.get_input_embeddings_batch(inputs["input_ids"])
        position_ids, _ = self._model.vlm.language_model.get_rope_index(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        hidden = self._model.vlm.language_model.model(
            None,
            inputs_embeds=inputs_embeds,
            mask=None,
            cache=None,
            position_ids=position_ids,
        )
        embeds = normalize_embeddings(self._model.embedding_proj_layer(hidden))
        embeds = embeds * inputs["attention_mask"][:, :, None]
        out = _nonpad_rows(embeds, inputs["attention_mask"])
        mx.eval(out)  # force materialization — MLX is lazy by default
        return out

    def embed_image(self, image: Image.Image) -> mx.array:
        """Multi-vector embedding for a single image. Returns [n_tokens, dim] mx.array."""
        self.load()
        inputs = self._prepare_image(image)
        inputs_embeds = self._model.get_input_embeddings_batch(
            inputs["input_ids"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )
        position_ids, _ = self._model.vlm.language_model.get_rope_index(
            inputs["input_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        )
        hidden = self._model.vlm.language_model.model(
            None,
            inputs_embeds=inputs_embeds,
            mask=None,
            cache=None,
            position_ids=position_ids,
        )
        embeds = normalize_embeddings(self._model.embedding_proj_layer(hidden))
        embeds = embeds * inputs["attention_mask"][:, :, None]
        out = _nonpad_rows(embeds, inputs["attention_mask"])
        mx.eval(out)  # force materialization — MLX is lazy by default
        return out

    def score(self, query_emb: mx.array, page_emb: mx.array) -> float:
        """MaxSim score between one query and one page (multi-vector each)."""
        sims = query_emb @ page_emb.T
        return float(mx.sum(mx.max(sims, axis=1)))
