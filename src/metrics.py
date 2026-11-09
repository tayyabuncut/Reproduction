"""Evaluation metrics.

Baseline 2 (Table 1): Recall, F1, Cosine similarity (all-MiniLM-L6-v2), BLEU. The paper's Appendix D defines
P/R/F1 generically; for free-text QA the standard (and only computable) reading is SQuAD-style token overlap
between the generated and the reference answer. BLEU is averaged sentence-level BLEU-4 in [0, 1]
(sacrebleu's sentence_bleu with exponential smoothing when sacrebleu is installed, an equivalent
re-implementation otherwise). Both choices are documented assumptions.

Baseline 1: ASR, MRR@10, nDCG@10, 95% bootstrap CIs, paired Wilcoxon signed-rank tests.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------- text QA metrics (Baseline 2)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_prf(pred: str, gold: str) -> tuple[float, float, float]:
    p_toks, g_toks = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p_toks or not g_toks:
        same = float(p_toks == g_toks)
        return same, same, same
    common = Counter(p_toks) & Counter(g_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0, 0.0, 0.0
    prec, rec = n / len(p_toks), n / len(g_toks)
    return prec, rec, 2 * prec * rec / (prec + rec)


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def sentence_bleu(pred: str, gold: str, max_n: int = 4) -> float:
    try:
        import sacrebleu

        return sacrebleu.sentence_bleu(pred, [gold]).score / 100.0
    except ImportError:
        pass
    # Re-implementation of smoothed sentence BLEU (effective order, exp smoothing on higher orders).
    # If there is no unigram overlap at all, BLEU is 0 (matches the no-match corpus-BLEU limit and avoids
    # the inflated smoothed value short disjoint sentences would otherwise get).
    p, r = pred.split(), gold.split()
    if not p or not r:
        return 0.0
    if sum((_ngrams(p, 1) & _ngrams(r, 1)).values()) == 0:
        return 0.0
    log_prec, smooth, eff_order = 0.0, 1.0, 0
    for n in range(1, max_n + 1):
        pn, rn = _ngrams(p, n), _ngrams(r, n)
        total = sum(pn.values())
        if total == 0:  # hypothesis shorter than n: order ignored (effective order)
            continue
        eff_order += 1
        match = sum((pn & rn).values())
        if match == 0:
            smooth *= 2
            log_prec += math.log(1 / (smooth * total))
        else:
            log_prec += math.log(match / total)
    if eff_order == 0:
        return 0.0
    bp = 1.0 if len(p) > len(r) else math.exp(1 - len(r) / len(p))
    return bp * math.exp(log_prec / eff_order)


class CosineScorer:
    """Cosine similarity of sentence embeddings (sentence-transformers/all-MiniLM-L6-v2), as in B2 App. D."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, preds: Sequence[str], golds: Sequence[str]) -> np.ndarray:
        a = self.model.encode(list(preds), normalize_embeddings=True, convert_to_numpy=True)
        b = self.model.encode(list(golds), normalize_embeddings=True, convert_to_numpy=True)
        return (a * b).sum(axis=1)


def qa_scores(preds: Sequence[str], golds: Sequence[str], cosine: CosineScorer | None = None) -> dict:
    prf = np.array([token_prf(p, g) for p, g in zip(preds, golds)])
    out = {
        "precision": float(prf[:, 0].mean()),
        "recall": float(prf[:, 1].mean()),
        "f1": float(prf[:, 2].mean()),
        "bleu": float(np.mean([sentence_bleu(p, g) for p, g in zip(preds, golds)])),
    }
    if cosine is not None:
        out["cosine"] = float(np.mean(cosine(preds, golds)))
    return out


# ---------------------------------------------------------------- ranking metrics (Baseline 1)


def mrr_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked_ids[:k]):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def dedup_doc_ranking(chunk_hits: Iterable, key=lambda c: c.doc_id) -> list[str]:
    """Collapses a chunk ranking into a document ranking (first occurrence wins)."""
    seen, out = set(), []
    for c, _ in chunk_hits:
        d = key(c)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


# ---------------------------------------------------------------- statistics


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0}
    try:
        r = wilcoxon(a, b, zero_method="wilcox")
        return {"statistic": float(r.statistic), "p_value": float(r.pvalue)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": float("nan")}


def cohen_kappa(y1: Sequence[int], y2: Sequence[int]) -> float:
    y1, y2 = np.asarray(y1), np.asarray(y2)
    po = float((y1 == y2).mean())
    labels = np.union1d(y1, y2)
    pe = float(sum((y1 == l).mean() * (y2 == l).mean() for l in labels))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def prf_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
