"""Baseline 2 RAG QA pipeline + ablation (Li, Yuan & Zhang, 2024, Sec. 4-5).

Pipeline (Fig. 3):
  mxbai-embed-large-v1 embeddings -> MMR top-10 retrieval -> bge-reranker-large keeps top-5
  -> Llama-2-7b-chat generates (prompt template from App. B.2).

Ablation (Table 1): four configurations scored on Recall / F1 / Cosine / BLEU over 4 random draws of 128 QA
pairs from the human-evaluated test set:
  (1) Baseline (no RAG)            - LLM answers from the question alone.
  (2) Raw RAG pipeline             - pretrained embedding + reranker + pretrained Llama-2.
  (3) + fine-tuned embedding       - embedding model fine-tuned on the QA train split.
  (4) + fine-tuned core model      - Llama-2 QLoRA-fine-tuned on the QA train split.
  (5) + both fine-tuned.
This module runs the retrieval+generation and scoring; fine-tuning lives in b2_finetune.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .chunking import Chunk
from .metrics import CosineScorer, qa_scores
from .utils import Timer

CORE_PROMPT_SYSTEM = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to "
    "answer the question. If you don't know the answer, just say that you don't know. Use 50 words maximum "
    "and keep the answer concise."
)


def build_core_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": CORE_PROMPT_SYSTEM},
        {"role": "user", "content": f"Question: {question}\nContext: {context}"},
    ]


@dataclass
class RagPipeline:
    """Wraps an embedding retriever (+optional reranker) and a generator LLM."""
    dense_retriever: object                 # DenseRetriever over the corpus chunks
    llm: object
    reranker: object | None = None
    retrieve_k: int = 10
    rerank_top_n: int = 5
    use_mmr: bool = True
    timer: Timer = field(default_factory=Timer)

    def retrieve(self, question: str) -> list[Chunk]:
        with self.timer.stage("retrieve"):
            if self.use_mmr:
                hits = self.dense_retriever.search_mmr(question, k=self.retrieve_k, fetch_k=self.retrieve_k * 2)
            else:
                hits = self.dense_retriever.search(question, k=self.retrieve_k)
            if self.reranker is not None:
                hits = self.reranker.rerank(question, hits, top_n=self.rerank_top_n)
            else:
                hits = hits[: self.rerank_top_n]
        return [c for c, _ in hits]

    def answer(self, question: str, use_rag: bool = True, max_new_tokens: int = 128) -> tuple[str, list[str]]:
        passages = [c.text for c in self.retrieve(question)] if use_rag else []
        context = "\n\n".join(passages) if passages else "(no external context provided)"
        prompt = self.llm.chat_to_prompt(build_core_messages(question, context))
        with self.timer.stage("generate"):
            ans = self.llm.generate([prompt], max_new_tokens=max_new_tokens, temperature=0.0)[0]
        return ans, passages

    def batch_answer(self, questions: list[str], use_rag: bool = True, max_new_tokens: int = 128,
                     batch_size: int = 8) -> list[str]:
        prompts = []
        for q in questions:
            passages = [c.text for c in self.retrieve(q)] if use_rag else []
            context = "\n\n".join(passages) if passages else "(no external context provided)"
            prompts.append(self.llm.chat_to_prompt(build_core_messages(q, context)))
        with self.timer.stage("generate"):
            return self.llm.generate(prompts, max_new_tokens=max_new_tokens, temperature=0.0,
                                     batch_size=batch_size)


def evaluate_config(pipeline: RagPipeline, qa_test: list[dict], use_rag: bool, cosine: CosineScorer | None,
                    n_draws: int = 4, draw_size: int = 128, seed: int = 42, max_new_tokens: int = 128) -> dict:
    """Reproduces Table 1's protocol: mean +/- std over `n_draws` random draws of `draw_size` QA pairs."""
    rng = random.Random(seed)
    pool = list(range(len(qa_test)))
    per_draw = {"recall": [], "f1": [], "cosine": [], "bleu": [], "precision": []}
    size = min(draw_size, len(qa_test))
    for d in range(n_draws):
        idx = rng.sample(pool, size) if len(pool) >= size else [rng.choice(pool) for _ in range(size)]
        sub = [qa_test[i] for i in idx]
        preds = pipeline.batch_answer([r["question"] for r in sub], use_rag=use_rag, max_new_tokens=max_new_tokens)
        golds = [r["answer"] for r in sub]
        sc = qa_scores(preds, golds, cosine=cosine)
        for key in per_draw:
            if key in sc:
                per_draw[key].append(sc[key])
    out = {}
    for key, vals in per_draw.items():
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)}
    out["n_draws"], out["draw_size"] = n_draws, size
    return out
