"""Baseline 2 data preparation (Li, Yuan & Zhang, 2024).

The authors released code + data at github.com/anlp-team/LTI_Neural_Navigator (MIT). We reuse their artefacts
rather than re-crawling CMU:
  * knowledge-base text files (html/, pdf/, papers/) -> the retrieval corpus;
  * WizardLM-generated QA pairs (34,781; 27,824 train / 6,957 test) -> fine-tuning + evaluation data.

Because the repo layout and exact data filenames can drift, this module is defensive: it accepts either the
cloned repo path or a HuggingFace dataset, discovers the corpus/QA files by pattern, and—if neither is
available—can fall back to a tiny built-in CMU-style sample so the pipeline is runnable for a smoke test
(clearly flagged; never used for reported numbers).

Table 1's "human evaluated test set" (128-sample draws) is the subset the authors hand-checked. If that split
is not present in the release, we document it and sample from the random test split instead.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .chunking import Chunk, word_chunks

CORPUS_GLOBS = ["**/data/**/*.txt", "**/html/**/*.txt", "**/pdf/**/*.txt", "**/papers/**/*.txt", "**/*.txt"]
QA_GLOBS = ["**/*qa*pair*.json*", "**/*train*.json*", "**/*test*.json*", "**/*.jsonl", "**/*qa*.json"]


def load_corpus(repo_or_data_dir: str, max_docs: int | None = None, chunk_words: int = 1000) -> list[Chunk]:
    root = Path(repo_or_data_dir)
    # Try the specific directory globs first (html/pdf/papers/data); only fall back to the catch-all
    # "**/*.txt" if those find nothing. Specific globs are accumulated together, not stopped at the first.
    specific, catch_all = CORPUS_GLOBS[:-1], CORPUS_GLOBS[-1]
    files: list[Path] = []
    seen = set()
    for g in specific:
        for f in root.glob(g):
            if f.is_file() and f.suffix == ".txt" and f not in seen:
                seen.add(f)
                files.append(f)
    if not files:
        for f in root.glob(catch_all):
            if f.is_file() and f not in seen:
                seen.add(f)
                files.append(f)
    files = sorted(files)
    if max_docs:
        files = files[:max_docs]
    chunks: list[Chunk] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if len(text) < 200:  # paper drops <200-char files
            continue
        rel = str(f.relative_to(root))
        src = "paper" if "paper" in rel.lower() else ("pdf" if "pdf" in rel.lower() else "html")
        chunks.extend(word_chunks(rel, text, size=chunk_words, overlap=0, meta={"source_type": src}))
    return chunks


def _read_qa_file(f: Path) -> list[dict]:
    out = []
    txt = f.read_text(encoding="utf-8", errors="ignore")
    if f.suffix == ".jsonl":
        for line in txt.splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            out = data
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    out.extend(v)
    except json.JSONDecodeError:
        pass
    return out


def _normalise_qa(rows: list[dict]) -> list[dict]:
    norm = []
    for r in rows:
        q = r.get("question") or r.get("query") or r.get("instruction")
        a = r.get("answer") or r.get("output") or r.get("response")
        if not q or not a:
            continue
        norm.append({"question": str(q).strip(), "answer": str(a).strip(),
                     "context": str(r.get("context", "")).strip()})
    return norm


def load_qa_from_embedder_dataset(repo_or_data_dir: str) -> list[dict]:
    """Load all QA pairs from the authors' `embedder_dataset/` folder.

    Each file is {file_path, num_qa_pairs, doc_text, qa_list:[{question, answer, top_k_docs}]}. The released
    set holds 16,329 pairs across 272 files (a subset of the paper's 34,781). `top_k_docs` is the authors'
    retrieved context; we drop it and retrieve ourselves so the RAG pipeline is reproduced, not replayed.
    """
    root = Path(repo_or_data_dir)
    rows: list[dict] = []
    for fp in sorted(root.glob("**/embedder_dataset/*.json")) or sorted(root.glob("embedder_dataset/*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        for it in d.get("qa_list", []) if isinstance(d, dict) else []:
            q, a = it.get("question"), it.get("answer")
            if q and a:
                rows.append({"question": str(q).strip(), "answer": str(a).strip(), "context": ""})
    return rows


def load_qa_pairs(repo_or_data_dir: str, split: str = "test") -> list[dict]:
    """Returns normalised [{question, answer, context}].

    The authors' `embedder_dataset/` layout is one flat set with no train/test split, so it is returned only
    for an unsplit request (split in {"", "all", None}); the caller then splits it with train_test_split_qa.
    A split-named request ("train"/"test") against that layout returns [] so the caller falls through to the
    unsplit path. Split-named JSON/JSONL files (the offline sample, or other layouts) still match directly.
    """
    if split in ("", "all", None):
        emb = load_qa_from_embedder_dataset(repo_or_data_dir)
        if emb:
            return emb
    root = Path(repo_or_data_dir)
    candidates: list[Path] = []
    for g in QA_GLOBS:
        candidates.extend(p for p in root.glob(g) if p.is_file())
    split_files = [p for p in candidates if split in p.name.lower()]
    chosen = split_files or candidates
    rows: list[dict] = []
    for f in sorted(set(chosen)):
        rows.extend(_read_qa_file(f))
    return _normalise_qa(rows)


def train_test_split_qa(rows: list[dict], test_ratio: float = 0.2, seed: int = 42):
    r = random.Random(seed)
    idx = list(range(len(rows)))
    r.shuffle(idx)
    n_test = int(len(rows) * test_ratio)
    test_idx = set(idx[:n_test])
    train = [rows[i] for i in idx[n_test:]]
    test = [rows[i] for i in idx[:n_test]]
    return train, test


# --------------------------------------------------- tiny offline fallback (smoke test only)

_SAMPLE_DOCS = {
    "html/academic_calendar.txt": (
        "Carnegie Mellon University Academic Calendar. Classes for the Fall 2024 semester will begin on "
        "August 26, 2024. The add/drop deadline is September 9, 2024. Fall break is in mid October. The "
        "Language Technology Institute is part of the School of Computer Science at CMU in Pittsburgh."),
    "html/lti_overview.txt": (
        "The Language Technology Institute (LTI) at Carnegie Mellon University offers MLT, MIIS, MCDS and PhD "
        "programs. Graham Neubig is a faculty member whose research focuses on natural language processing and "
        "machine translation. The LTI is located in the Gates Hillman Center."),
    "papers/sama_meta_learning.txt": (
        "SAMA is a scalable meta-learning method. SAMA showcases a 1.7x and 4.8x increase in throughput and a "
        "2.0x and 3.8x decrease in memory consumption on single and multi GPU setups respectively, compared to "
        "baseline meta learning algorithms."),
    "html/andrew_project.txt": (
        "The Andrew Project was created by Carnegie Mellon University and IBM in the early 1980s. It built the "
        "first high-speed campus computer network and a GUI-based computing environment for students and "
        "faculty, making CMU one of the most wired campuses in the world."),
}
_SAMPLE_QA = [
    {"question": "When will classes begin in the Fall 2024 semester?",
     "answer": "Classes for the Fall 2024 semester will begin on August 26, 2024.", "context": ""},
    {"question": "Where is the Language Technology Institute located?",
     "answer": "The LTI is in the Gates Hillman Center at Carnegie Mellon University.", "context": ""},
    {"question": "What is Graham Neubig's research focus?",
     "answer": "Natural language processing and machine translation.", "context": ""},
    {"question": "How much does SAMA decrease memory consumption on a multi-GPU setup?",
     "answer": "SAMA decreases memory consumption by 3.8x on a multi-GPU setup.", "context": ""},
    {"question": "Which project created CMU's first high-speed computer network?",
     "answer": "The Andrew Project created CMU's first high-speed computer network.", "context": ""},
]


def build_sample_dataset(out_dir: str) -> str:
    """Writes the offline sample corpus + QA split; returns the directory. SMOKE TEST ONLY."""
    root = Path(out_dir)
    for rel, text in _SAMPLE_DOCS.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (root / "qa_test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _SAMPLE_QA), encoding="utf-8")
    (root / "qa_train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _SAMPLE_QA), encoding="utf-8")
    return str(root)
