"""Generates notebooks/Assignment2_Reproduction.ipynb (valid nbformat v4 JSON, no nbformat dependency)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def _lines(s):
    # nbformat stores source as a list of lines, each keeping its trailing newline except the last
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": _lines(s)})

md("""# Assignment 02 — Reproducing Two RAG Papers (Colab, free T4)

**Baseline 1:** Guo & Wei (2026), *Hidden-in-Plain-Text: A Benchmark for Social-Web Indirect Prompt Injection in RAG* (OpenRAG-Soc) — arXiv:2601.10923
**Baseline 2:** Li, Yuan & Zhang (2024), *Enhancing LLM Factual Accuracy with RAG to Counter Hallucinations* — arXiv:2403.10446

This notebook runs both reproductions end-to-end. **Runtime → Change runtime type → T4 GPU** before you start.

Pipeline:
1. Install deps, set the HuggingFace token (both papers use **gated Llama** models — request access first).
2. Baseline 2: clone the authors' repo, build the RAG pipeline, fine-tune embedding + core (QLoRA), run the Table 1 ablation.
3. Baseline 1: build the OpenRAG-Soc carrier corpus, sweep the four defenses, measure ASR + poisoning.
4. Collect results into `results/` and render the tables/figures used in the report.

Every result cell writes JSON to `results/` so numbers can be copied straight into the report. Nothing here is
pre-filled — the tables show what *your* run produced.""")

md("## 0. Setup")
code("""# GPU check
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout or "NO GPU - set Runtime to T4")""")

code("""%pip install -q "transformers>=4.44" "accelerate>=0.33" "bitsandbytes>=0.43" "peft>=0.12" \\
    "trl>=0.9" "sentence-transformers>=3.0" "datasets>=2.20" faiss-cpu rank-bm25 sacrebleu beautifulsoup4 lxml \\
    pyyaml matplotlib
print("installed")""")

code("""# Get the reproduction code into the Colab session.
# Option A: upload the rag-repro/ folder (Files panel) and set REPRO_DIR to it.
# Option B: if you pushed it to GitHub, git clone it here.
import os, sys
REPRO_DIR = "rag-repro"   # <-- adjust if you uploaded elsewhere
assert os.path.isdir(REPRO_DIR), f"{REPRO_DIR} not found - upload the project folder first"
sys.path.insert(0, REPRO_DIR)
os.chdir(REPRO_DIR)
print("cwd:", os.getcwd())""")

code("""# HuggingFace login (needed for gated Llama-2 / Llama-3). Request access on the model pages first:
#   https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
#   https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
from huggingface_hub import login
from getpass import getpass
HF_TOKEN = getpass("HF token: ")
login(HF_TOKEN)""")

code("""from src.utils import environment_info, save_json, set_seed
set_seed(0)
env = environment_info(); save_json(env, "results/environment.json"); env""")

md("""---
## 1. Baseline 2 — CMU RAG factual-accuracy pipeline

We reuse the authors' released corpus + WizardLM QA pairs (MIT-licensed) instead of re-crawling CMU.""")
code("""# Clone the authors' repo (corpus + QA pairs live inside it)
import os
if not os.path.isdir("LTI_Neural_Navigator"):
    !git clone --depth 1 https://github.com/anlp-team/LTI_Neural_Navigator
# Inspect what data files are present so b2_data can locate the corpus + QA splits
!find LTI_Neural_Navigator -maxdepth 3 -name "*.txt" | head -5
!find LTI_Neural_Navigator -maxdepth 3 \\( -name "*.json" -o -name "*.jsonl" \\) | head -10""")

code("""from src.b2_data import load_corpus, load_qa_pairs, train_test_split_qa
DATA_DIR = "LTI_Neural_Navigator"
corpus = load_corpus(DATA_DIR, chunk_words=1000)
# The QA pairs live in LTI_Neural_Navigator/embedder_dataset/*.json under the key 'qa_list' (16,329 pairs in
# 272 files). load_qa_pairs(split='all') reads that flat set; we make our own 80/20 split.
allqa = load_qa_pairs(DATA_DIR, split="all")
qa_train, qa_test = train_test_split_qa(allqa, test_ratio=0.2, seed=42)
print(f"corpus chunks={len(corpus)}  train={len(qa_train)}  test={len(qa_test)}")
# Expect ~ corpus chunks=2163  train=13064  test=3265. If test is only a few dozen, the embedder_dataset
# folder was not found - check the repo path.""")

md("""### 1a. Build the RAG pipeline (pretrained models)
mxbai-embed-large-v1 embeddings → MMR top-10 → bge-reranker-large top-5 → Llama-2-7b-chat.""")
code("""from src.generation import HFGenerator
from src.retrieval import DenseRetriever, CrossEncoderReranker
from src.metrics import CosineScorer
from src.b2_pipeline import RagPipeline, evaluate_config

llm = HFGenerator("meta-llama/Llama-2-7b-chat-hf", load_in_4bit=True, hf_token=HF_TOKEN)
emb = DenseRetriever(corpus, "mixedbread-ai/mxbai-embed-large-v1",
                     query_prefix="Represent this sentence for searching relevant passages: ", doc_prefix="")
rerank = CrossEncoderReranker("BAAI/bge-reranker-large")
cosine = CosineScorer("sentence-transformers/all-MiniLM-L6-v2")
def make_pipe(retriever, reranker, model):
    return RagPipeline(retriever, model, reranker, retrieve_k=10, rerank_top_n=5, use_mmr=True)""")

code("""# Table 1, configs (1) no-RAG and (2) raw RAG. draw_size=128, 4 draws (paper protocol).
b2 = {}
b2["baseline_no_rag"] = evaluate_config(make_pipe(emb, rerank, llm), qa_test, use_rag=False, cosine=cosine)
b2["raw_rag"]        = evaluate_config(make_pipe(emb, rerank, llm), qa_test, use_rag=True,  cosine=cosine)
save_json(b2, "results/b2_partial.json"); {k: v.get("f1") for k, v in b2.items()}""")

md("""### 1b. Embedding fine-tuning — NOT reproduced (documented negative result)

Fine-tuning `mxbai-embed-large-v1` on the released QA pairs **collapses**: the MultipleNegativesRankingLoss
training loss drops to exactly 0 within one epoch, and held-out retrieval **recall@5 falls from ~0.785
(pretrained) to 0.0** — the fine-tuned model retrieves nothing relevant. Cause: ~16k narrow, auto-generated,
**templated** answers give the contrastive loss only trivial in-batch negatives, so a strong pretrained encoder
collapses rather than improves. This matches the authors' own caution about fine-tuning on small/skewed data.
**We use the pretrained embedding** and record this as a reproducibility finding. The cell below *demonstrates*
the collapse (so it's in your evidence), then keeps the pretrained retriever.""")
code("""# Evidence for the report: show that fine-tuning collapses retrieval, then keep the pretrained embedding.
import random
sample = random.Random(0).sample(qa_test, min(200, len(qa_test)))
def answer_recall_at_k(retriever, qa, k=5):
    hits = 0
    for r in qa:
        passages = " ".join(c.text.lower() for c, _ in retriever.search(r["question"], k=k))
        words = [w for w in r["answer"].lower().split() if len(w) > 4]
        if words and sum(w in passages for w in words) / len(words) >= 0.5:
            hits += 1
    return round(hits / len(qa), 3)

print("pretrained embedding  answer-recall@5:", answer_recall_at_k(emb, sample))

RUN_COLLAPSE_DEMO = True   # set False to skip the ~2-3 min demonstration
if RUN_COLLAPSE_DEMO:
    from src.b2_finetune import finetune_embedding
    emb_dir = finetune_embedding("mixedbread-ai/mxbai-embed-large-v1", qa_train, "ft_embedding_demo",
                                 epochs=3, batch_size=8, warmup_ratio=0.1)
    emb_bad = DenseRetriever(corpus, emb_dir,
                             query_prefix="Represent this sentence for searching relevant passages: ", doc_prefix="")
    print("fine-tuned embedding  answer-recall@5:", answer_recall_at_k(emb_bad, sample),
          "  <- collapsed; NOT used")

# Decision: keep the pretrained embedding for all downstream rows.
emb_ft = emb
b2["rag_ft_embedding"] = "not_reproduced_embedding_finetune_collapsed"
save_json(b2, "results/b2_partial.json")""")

md("""### 1c. Fine-tune the core model (Llama-2-7b, QLoRA r=16, 5 epochs / 1000 steps, lr 2e-4)

Watch the training loss: it should start ~1-2 and **decrease gradually**. If it snaps to 0 or shows `nan`, stop
and treat it like the embedding collapse. Expected result (reproducing the paper): core fine-tuning *raises
recall but lowers F1/BLEU* because the small, templated QA set hurts the generator's fluency — a lower F1 here
is the correct reproduction, not a bug. Since we did not fine-tune the embedding, "+ Both" is identical to
"+ Core" and is not run separately.""")
code("""from src.b2_finetune import finetune_core_qlora
core_dir = finetune_core_qlora("meta-llama/Llama-2-7b-chat-hf", qa_train, "ft_core_lora", hf_token=HF_TOKEN,
                               lora_r=16, lora_alpha=32, lr=2e-4, epochs=5, max_steps=1000, batch_size=8)
# free the base model before loading the adapter copy to stay within 16 GB
llm.unload()
llm_ft = HFGenerator("meta-llama/Llama-2-7b-chat-hf", load_in_4bit=True, adapter_path=core_dir, hf_token=HF_TOKEN)
# uses the PRETRAINED embedding (emb), since embedding fine-tuning collapsed (see 1b)
b2["rag_ft_core"] = evaluate_config(make_pipe(emb, rerank, llm_ft), qa_test, use_rag=True, cosine=cosine)
b2["rag_ft_both"] = "not_reproduced_same_as_core_no_embedding_finetune"
save_json(b2, "results/b2_ablation_results.json")
{k: (v.get("f1") if isinstance(v, dict) else v) for k, v in b2.items()}""")

code("""# Render Table 1 (rows that collapsed are shown as 'not reproduced')
import pandas as pd
labels = {"baseline_no_rag":"Baseline (w/o RAG)","raw_rag":"Raw RAG Pipeline","rag_ft_embedding":"+ Embedding",
          "rag_ft_core":"+ Core Model Finetune","rag_ft_both":"+ Embedding & Core"}
rows=[]
for k,l in labels.items():
    r=b2.get(k)
    if isinstance(r, dict):
        rows.append({"Configuration":l,
            **{m.capitalize(): round(r[m]["mean"],3) for m in ["recall","f1","cosine","bleu"] if m in r}})
    else:
        rows.append({"Configuration":l,"Recall":"not reproduced","F1":"","Cosine":"","Bleu":""})
df=pd.DataFrame(rows); df.to_csv("results/b2_table1.csv", index=False); df""")

md("""---
## 2. Baseline 1 — OpenRAG-Soc indirect prompt injection

We regenerate the 6,200-page carrier corpus from the paper's recipe (no public data release), then sweep the
four defenses with Llama-3-8B-Instruct + BM25.""")
code("""from src import b1_corpus
from src.b1_harness import run_injection_eval, run_poisoning_eval, CONFIGS, PAPER_CONFIG_ORDER
from src.b1_detector import validate_detector

pages = b1_corpus.build_corpus(seed=13, n_hard_negatives=300, n_svg=200, scale=1.0)  # scale<1.0 for a faster run
print(b1_corpus.corpus_stats(pages))""")

code("""# Llama-3-8B-Instruct generator (reuse one model across all four defense configs)
llm3 = HFGenerator("meta-llama/Meta-Llama-3-8B-Instruct", load_in_4bit=True, hf_token=HF_TOKEN)
configs = {k: CONFIGS[k] for k in ["Vanilla","Sanitized","Normalized","Attribution","Sanitized+Normalized","All Defenses"]}
log=[]
b1 = run_injection_eval(pages, llm3, configs=configs, retriever_kind="bm25", k=5,
                        chunk_size=512, overlap=128, max_new_tokens=128, temperature=0.2, log_rows=log, seed=0)
save_json(b1, "results/b1_injection_results.json")
from src.utils import save_jsonl; save_jsonl(log, "results/b1_generations.jsonl")""")

code("""# ASR table (Table 1 of Baseline 1)
import pandas as pd
carriers=["hidden_span","offscreen_css","alt_text","aria","zero_width"]
cfgs=[c for c in PAPER_CONFIG_ORDER if c in b1]
tab={c:{cr: round(b1[c][cr]["asr"],1) if cr in b1[c] else None for cr in carriers} for c in cfgs}
for c in cfgs: tab[c]["macro"]=round(b1[c]["macro"]["asr"],1)
df1=pd.DataFrame(tab).T; df1.to_csv("results/b1_asr_table.csv"); df1""")

code("""# Detector vs. gold obey-labels (Sec. 5.0.2): P / R / F1 / kappa
det = validate_detector(log); save_json(det, "results/b1_detector_validation.json"); det""")

code("""# Retrieval poisoning sweep (Figure 1): dMRR@10 / dnDCG@10 vs poison budget
poison = run_poisoning_eval(pages, poison_fractions=(0.0,0.01,0.02,0.05,0.10),
                            retriever_kinds=("bm25",), configs_to_compare=("Vanilla","Sanitized+Normalized"),
                            k=10, chunk_size=512, overlap=128, n_eval_queries=200)
save_json(poison, "results/b1_poisoning_results.json")
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
for rk,c in poison.items():
    for cfg,d in c.items():
        plt.plot([f*100 for f in d["fraction"]], d["dMRR@10"], marker="o", label=f"{rk} ({cfg})")
plt.xlabel("Poison fraction (%)"); plt.ylabel("ΔMRR@10 vs clean"); plt.legend(); plt.grid(alpha=.3)
plt.title("Poison budget vs rank impact"); plt.tight_layout(); plt.savefig("results/figures_poisoning.png", dpi=150)
plt.show()""")

code("""# Latency per stage + utility (Sec. 5.0.1 / 5.0.4)
import pandas as pd
rows=[]
for c in cfgs:
    lat=b1[c]["latency"]; u=b1[c]["utility"]
    rows.append({"config":c, "ingest_ms":round(lat["ingest"]["median_ms"],1),
                 "retrieve_ms":round(lat["retrieve"]["median_ms"],1),
                 "generate_ms":round(lat["generate"]["median_ms"],1),
                 "answerability":round(u["answerability"],3),
                 "attr_consistency":u["attribution_consistency"]})
dfl=pd.DataFrame(rows); dfl.to_csv("results/b1_latency_utility.csv", index=False); dfl""")

md("""---
## 3. Package results
Download the `results/` folder and paste the tables/figures into the report (`report/`). Also record the GPU,
model versions and corpus scale you used so the report's reproducibility section is accurate.""")
code("""import shutil
shutil.make_archive("assignment2_results", "zip", "results")
from google.colab import files
files.download("assignment2_results.zip")""")

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"},
                 "accelerator": "GPU", "colab": {"provenance": []}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebooks" / "Assignment2_Reproduction.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
