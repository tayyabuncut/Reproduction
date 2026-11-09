"""Chunking strategies used by the two baselines.

* Baseline 2 (Li et al., 2024, Sec. 3.2): raw text split into chunks of 1000 words (no overlap stated).
* Baseline 1 (Guo & Wei, 2026, Sec. 3.2): fixed-size, strided passages, size in {256, 512, 768}
  tokens with overlap 128 (stride = size - 128).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


def word_chunks(doc_id: str, text: str, size: int = 1000, overlap: int = 0, meta: dict | None = None) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    out = []
    for i, start in enumerate(range(0, len(words), step)):
        piece = words[start : start + size]
        if not piece:
            break
        out.append(Chunk(f"{doc_id}::c{i}", doc_id, " ".join(piece), dict(meta or {})))
        if start + size >= len(words):
            break
    return out


def token_chunks(
    doc_id: str,
    text: str,
    size: int = 512,
    overlap: int = 128,
    tokenizer=None,
    meta: dict | None = None,
) -> list[Chunk]:
    """Strided chunking in model tokens when a HF tokenizer is given, else whitespace tokens.

    The paper does not say which tokenizer defines "size"; we use the dense retriever's tokenizer
    (e5-base-v2) when available, which is the most natural reading. Documented as an assumption.
    """
    if tokenizer is None:
        return word_chunks(doc_id, text, size=size, overlap=overlap, meta=meta)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return []
    step = max(1, size - overlap)
    out = []
    for i, start in enumerate(range(0, len(ids), step)):
        piece = ids[start : start + size]
        out.append(Chunk(f"{doc_id}::c{i}", doc_id, tokenizer.decode(piece), dict(meta or {})))
        if start + size >= len(ids):
            break
    return out
