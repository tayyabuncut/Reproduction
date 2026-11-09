"""Retrievers shared by both baselines.

* BM25Retriever   - Lucene-style BM25 (k1=1.2, b=0.75), pure Python/NumPy. Baseline 1 uses Pyserini/Lucene
                    0.22; Pyserini needs Java, which is awkward on Colab, so we re-implement the same scoring
                    function (Lucene BM25 IDF = log(1 + (N - df + 0.5)/(df + 0.5))). Documented deviation.
* DenseRetriever  - sentence-transformers bi-encoder, L2-normalised embeddings, cosine/inner-product search.
                    Uses FAISS HNSW (M=32, efSearch=64) when faiss is installed, otherwise exact search.
* mmr_select      - Maximal Marginal Relevance, as used by Baseline 2's LangChain retriever.
* CrossEncoderReranker - BAAI/bge-reranker-large reranker used by Baseline 2 ("BgeRerank").
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the of to in on for and or is are was were be been it this that with as by at from".split()
)


def bm25_tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.doc_tokens = [bm25_tokenize(c.text) for c in chunks]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(chunks) else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        # inverted index for speed
        self.postings: dict[str, list[int]] = {}
        for i, toks in enumerate(self.doc_tokens):
            for t in set(toks):
                self.postings.setdefault(t, []).append(i)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.chunks), dtype=np.float32)
        for t in set(bm25_tokenize(query)):
            if t not in self.postings:
                continue
            idf = self.idf[t]
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
                s[i] += idf * f * (self.k1 + 1) / denom
        return s

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        s = self.scores(query)
        if not len(s):
            return []
        k = min(k, len(s))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [(self.chunks[i], float(s[i])) for i in idx]


class DenseRetriever:
    """Bi-encoder retriever. `query_prefix` / `doc_prefix` follow each model card:
    e5-base-v2 -> 'query: ' / 'passage: '; mxbai-embed-large-v1 -> retrieval prompt on queries only;
    bge-large-en-v1.5 -> 'Represent this sentence for searching relevant passages: ' on queries.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        model_name_or_obj,
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 32,
        use_hnsw: bool = True,
        hnsw_m: int = 32,
        ef_search: int = 64,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = (
            SentenceTransformer(model_name_or_obj, device=device)
            if isinstance(model_name_or_obj, str)
            else model_name_or_obj
        )
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.emb = self.model.encode(
            [doc_prefix + c.text for c in chunks],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(chunks) > 500,
            convert_to_numpy=True,
        ).astype(np.float32)
        self.index = None
        if use_hnsw:
            try:
                import faiss

                self.index = faiss.IndexHNSWFlat(self.emb.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efSearch = ef_search
                self.index.add(self.emb)
            except ImportError:
                self.index = None  # exact search fallback (documented)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            [self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)[0]

    def _search_idx(self, q: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        k = min(k, len(self.chunks))
        if self.index is not None:
            d, i = self.index.search(q[None, :], k)
            pairs = [(int(j), float(s)) for j, s in zip(i[0], d[0]) if j >= 0]
        else:
            s = self.emb @ q
            idx = np.argsort(-s)[:k]
            pairs = [(int(j), float(s[j])) for j in idx]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        idx, sc = self._search_idx(self.encode_query(query), k)
        return [(self.chunks[j], s) for j, s in zip(idx, sc)]

    def search_mmr(self, query: str, k: int = 10, fetch_k: int = 20, lambda_mult: float = 0.5):
        """LangChain-style MMR: fetch `fetch_k` by similarity, then greedily pick `k` diverse ones."""
        q = self.encode_query(query)
        idx, sc = self._search_idx(q, fetch_k)
        picked = mmr_select(q, self.emb[idx], k=k, lambda_mult=lambda_mult)
        return [(self.chunks[idx[p]], sc[p]) for p in picked]


def mmr_select(query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lambda_mult: float = 0.5) -> list[int]:
    if len(cand_vecs) == 0:
        return []
    sim_q = cand_vecs @ query_vec
    selected = [int(np.argmax(sim_q))]
    while len(selected) < min(k, len(cand_vecs)):
        sim_sel = cand_vecs @ cand_vecs[selected].T  # (n, |S|)
        mmr = lambda_mult * sim_q - (1 - lambda_mult) * sim_sel.max(axis=1)
        mmr[selected] = -np.inf
        selected.append(int(np.argmax(mmr)))
    return selected


class CrossEncoderReranker:
    """BAAI/bge-reranker-large (CrossEncoder), keeps top_n of the candidates (paper: top-5 of 10)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str | None = None, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(self, query: str, cands: list[tuple[Chunk, float]], top_n: int = 5):
        if not cands:
            return []
        scores = self.model.predict([(query, c.text) for c, _ in cands], batch_size=16)
        order = np.argsort(-np.asarray(scores))[:top_n]
        return [(cands[i][0], float(scores[i])) for i in order]
