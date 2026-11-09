# Assignment 02 — Reproducing Two RAG Papers

Reproduction of the two papers selected in Assignment 01:

| | Paper | Venue | arXiv | What we reproduce |
|---|---|---|---|---|
| **Baseline 1** | Guo & Wei, *Hidden-in-Plain-Text: A Benchmark for Social-Web Indirect Prompt Injection in RAG* (OpenRAG-Soc) | WWW '26 | [2601.10923](https://arxiv.org/abs/2601.10923) | IPI attack-success rate (Table 1), the sanitize/normalize/attribution defenses, retrieval-poisoning rank shift (Fig. 1), detector validation, latency |
| **Baseline 2** | Li, Yuan & Zhang, *Enhancing LLM Factual Accuracy with RAG to Counter Hallucinations* | 2024 | [2403.10446](https://arxiv.org/abs/2403.10446) | The RAG QA pipeline and the Table 1 ablation (no-RAG → raw RAG → +fine-tuned embedding → +fine-tuned core) on Recall/F1/Cosine/BLEU |

> **Compute note.** This code targets a **free Colab T4 (16 GB)**. Both papers used larger hardware, so the
> defaults here fit 7–8B models in 4-bit. Every deviation from the papers is listed in
> [`report/`](report/) §Deviations and in the module docstrings. Do **not** treat the numbers in the report as
> copied from the papers — they are placeholders until you run the notebook and paste in your own.

## Repository layout

```
rag-repro/
├── notebooks/Assignment2_Reproduction.ipynb   # <- run this on Colab (T4). Primary artifact.
├── src/
│   ├── chunking.py        # word / strided-token chunking
│   ├── retrieval.py       # BM25, dense bi-encoder (FAISS HNSW), MMR, cross-encoder reranker
│   ├── generation.py      # 4-bit HF generator (+ MockLLM for CPU tests)
│   ├── metrics.py         # Recall/F1/Cosine/BLEU, MRR@10, nDCG@10, bootstrap CI, Wilcoxon, kappa
│   ├── b1_corpus.py       # synthetic OpenRAG-Soc carrier corpus (hidden/off-screen/alt/aria/zero-width)
│   ├── b1_defenses.py     # ingest, DOMPurify-style sanitizer, NFKC normalizer, attribution gating
│   ├── b1_detector.py     # instruction-following (ASR) detector + human-audit-style validation
│   ├── b1_harness.py      # ingest→retrieve→generate→score; injection + poisoning evaluation
│   ├── b2_data.py         # load the authors' CMU corpus + WizardLM QA pairs (offline sample fallback)
│   ├── b2_pipeline.py     # mxbai→MMR→bge-reranker→Llama-2 pipeline + Table 1 evaluation
│   └── b2_finetune.py     # QLoRA core fine-tune + embedding fine-tune
├── scripts/
│   ├── run_baseline1.py   # CLI runner (also `--mock` CPU smoke mode)
│   ├── run_baseline2.py   # CLI runner (also `--mock`)
│   └── build_notebook.py  # regenerates the Colab notebook
├── configs/               # baseline1.yaml, baseline2.yaml (all hyperparameters)
├── tests/run_cpu_tests.py # 39 CPU-only checks, no GPU/network needed
├── report/                # the write-up (report.md) + results tables
└── requirements.txt
```

## Quick start (Colab)

1. Open `notebooks/Assignment2_Reproduction.ipynb` in Colab, set **Runtime → T4 GPU**.
2. Upload the `rag-repro/` folder (or `git clone` your copy).
3. Request access to the gated models, then paste your HF token when prompted:
   - <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>
   - <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>
4. Run top to bottom. Results are written to `results/` and zipped for download.

### Fitting a free T4 / staying in the session limit
- Lower `corpus.scale` in `configs/baseline1.yaml` (e.g. `0.25`) for a smaller OpenRAG-Soc corpus.
- Lower `finetune_core.max_steps` or `eval.draw_size` for Baseline 2.
- The notebook calls `llm.unload()` before loading the fine-tuned adapter so both don't sit in VRAM at once.

## Run locally / from the CLI

```bash
pip install -r requirements.txt
# Baseline 2 needs the authors' data:
git clone https://github.com/anlp-team/LTI_Neural_Navigator
python scripts/run_baseline2.py --config configs/baseline2.yaml --out results/b2 --hf-token $HF_TOKEN
python scripts/run_baseline1.py --config configs/baseline1.yaml --out results/b1 --hf-token $HF_TOKEN
```

## Verify the plumbing without a GPU

```bash
python tests/run_cpu_tests.py     # 39 checks on a MockLLM — should print "39 passed, 0 failed"
python scripts/run_baseline1.py --mock --scale 0.03 --out results/b1_smoke
python scripts/run_baseline2.py --mock --out results/b2_smoke
```

The `--mock` runs use a deterministic fake LLM and (for Baseline 2) a tiny built-in CMU-style sample, so they
exercise every code path but produce **meaningless numbers** — they are for checking the pipeline runs, nothing
else.

## Key deviations from the papers (see the report for the full list)

- **Baseline 1 corpus** is regenerated synthetically from the paper's description (no public data release). The
  zero-width carrier uses plain-text payloads beside zero-width markers rather than an invisible-character
  text encoding, so the Normalized defense is not expected to reproduce the paper's zero-width drop.
- **BM25** is a pure-Python Lucene-BM25 re-implementation (k1=1.2, b=0.75) instead of Pyserini, to avoid a Java
  dependency on Colab.
- **Sanitizer** is a BeautifulSoup policy mirroring DOMPurify (which is JavaScript).
- **Generators** run in 4-bit NF4 to fit a T4.
- **Baseline 2** reuses the authors' released corpus + QA pairs rather than re-crawling CMU.

## License / attribution
Baseline 2 data and some design choices derive from `anlp-team/LTI_Neural_Navigator` (MIT). Papers are cited
above. This reproduction code is for coursework.
