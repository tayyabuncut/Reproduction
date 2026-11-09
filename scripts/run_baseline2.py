#!/usr/bin/env python3
"""Run the full Baseline 2 (CMU RAG) reproduction: Table 1 ablation over 5 configurations.

Usage:
    # 1) clone the authors' repo first so the corpus + QA pairs are available:
    git clone https://github.com/anlp-team/LTI_Neural_Navigator
    # 2) run:
    python scripts/run_baseline2.py --config configs/baseline2.yaml --out results/b2 --hf-token $HF_TOKEN
    # offline smoke test (mock LLM + tiny sample corpus, plumbing only):
    python scripts/run_baseline2.py --config configs/baseline2.yaml --out results/b2_smoke --mock
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.b2_data import (build_sample_dataset, load_corpus, load_qa_pairs, train_test_split_qa)
from src.b2_pipeline import RagPipeline, evaluate_config
from src.utils import environment_info, load_config, save_json, set_seed

CONFIG_TABLE = ["baseline_no_rag", "raw_rag", "rag_ft_embedding", "rag_ft_core", "rag_ft_both"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline2.yaml")
    ap.add_argument("--out", default="results/b2")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--skip-finetune", action="store_true", help="evaluate only pretrained configs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data
    data_dir = cfg["data"]["repo_dir"]
    if args.mock or cfg["data"]["use_sample_fallback"] or not Path(data_dir).exists():
        if not (args.mock or cfg["data"]["use_sample_fallback"]):
            print(f"WARNING: {data_dir} not found; falling back to offline sample (SMOKE TEST, not reproduction).")
        data_dir = build_sample_dataset(str(out / "sample_data"))
    corpus = load_corpus(data_dir, max_docs=cfg["data"]["max_docs"], chunk_words=cfg["data"]["chunk_words"])
    # Prefer the authors' flat embedder_dataset set (16,329 pairs) and make our own split; only fall back to
    # split-named files (offline sample / other layouts) when that set is absent.
    allqa = load_qa_pairs(data_dir, split="all")
    if allqa:
        qa_train, qa_test = train_test_split_qa(allqa, test_ratio=0.2, seed=cfg.get("seed", 42))
    else:
        qa_test = load_qa_pairs(data_dir, split="test")
        qa_train = load_qa_pairs(data_dir, split="train")
        if not qa_test or not qa_train:
            qa_train, qa_test = train_test_split_qa(load_qa_pairs(data_dir, split=""))
    print(f"Corpus chunks: {len(corpus)} | train QA: {len(qa_train)} | test QA: {len(qa_test)}")

    # ---- models
    if args.mock:
        from src.generation import MockLLM
        from src.retrieval import BM25Retriever
        base_llm = MockLLM()
        base_retr = _bm25_wrap(BM25Retriever(corpus))
        reranker = None
    else:
        from src.generation import HFGenerator
        from src.retrieval import CrossEncoderReranker, DenseRetriever
        base_llm = HFGenerator(cfg["core"]["model_id"], load_in_4bit=cfg["core"]["load_in_4bit"],
                               hf_token=args.hf_token)
        base_retr = DenseRetriever(corpus, cfg["retrieval"]["embedding_model"],
                                   doc_prefix="", query_prefix="Represent this sentence for searching "
                                   "relevant passages: ")
        reranker = CrossEncoderReranker(cfg["retrieval"]["reranker_model"])

    cosine = None
    if not args.mock:
        from src.metrics import CosineScorer
        cosine = CosineScorer(cfg["eval"]["cosine_model"])

    results = {}

    def make_pipe(retriever, rerank, llm):
        return RagPipeline(dense_retriever=retriever, llm=llm, reranker=rerank,
                           retrieve_k=cfg["retrieval"]["retrieve_k"], rerank_top_n=cfg["retrieval"]["rerank_top_n"],
                           use_mmr=cfg["retrieval"]["use_mmr"])

    # (1) baseline no RAG
    results["baseline_no_rag"] = evaluate_config(make_pipe(base_retr, reranker, base_llm), qa_test,
                                                 use_rag=False, cosine=cosine,
                                                 n_draws=cfg["eval"]["n_draws"], draw_size=cfg["eval"]["draw_size"])
    print("baseline_no_rag:", results["baseline_no_rag"].get("f1"))

    # (2) raw RAG
    results["raw_rag"] = evaluate_config(make_pipe(base_retr, reranker, base_llm), qa_test, use_rag=True,
                                         cosine=cosine, n_draws=cfg["eval"]["n_draws"],
                                         draw_size=cfg["eval"]["draw_size"])
    print("raw_rag:", results["raw_rag"].get("f1"))

    if not args.mock and not args.skip_finetune:
        from src.b2_finetune import finetune_core_qlora, finetune_embedding
        from src.retrieval import CrossEncoderReranker, DenseRetriever
        from src.generation import HFGenerator

        # (3) + fine-tuned embedding
        if cfg["finetune_embedding"]["run"]:
            emb_dir = finetune_embedding(cfg["retrieval"]["embedding_model"], qa_train,
                                         cfg["finetune_embedding"]["output_dir"],
                                         epochs=cfg["finetune_embedding"]["epochs"],
                                         batch_size=cfg["finetune_embedding"]["batch_size"],
                                         max_steps=cfg["finetune_embedding"]["max_steps"],
                                         warmup_ratio=cfg["finetune_embedding"]["warmup_ratio"])
            ft_retr = DenseRetriever(corpus, emb_dir, doc_prefix="",
                                     query_prefix="Represent this sentence for searching relevant passages: ")
            results["rag_ft_embedding"] = evaluate_config(make_pipe(ft_retr, reranker, base_llm), qa_test,
                                                          use_rag=True, cosine=cosine,
                                                          n_draws=cfg["eval"]["n_draws"],
                                                          draw_size=cfg["eval"]["draw_size"])
            print("rag_ft_embedding:", results["rag_ft_embedding"].get("f1"))

        # (4) + fine-tuned core
        if cfg["finetune_core"]["run"]:
            core_dir = finetune_core_qlora(cfg["core"]["model_id"], qa_train, cfg["finetune_core"]["output_dir"],
                                           hf_token=args.hf_token, lora_r=cfg["finetune_core"]["lora_r"],
                                           lora_alpha=cfg["finetune_core"]["lora_alpha"],
                                           lr=cfg["finetune_core"]["lr"], epochs=cfg["finetune_core"]["epochs"],
                                           max_steps=cfg["finetune_core"]["max_steps"],
                                           batch_size=cfg["finetune_core"]["batch_size"])
            ft_llm = HFGenerator(cfg["core"]["model_id"], load_in_4bit=True, adapter_path=core_dir,
                                 hf_token=args.hf_token)
            results["rag_ft_core"] = evaluate_config(make_pipe(base_retr, reranker, ft_llm), qa_test,
                                                     use_rag=True, cosine=cosine, n_draws=cfg["eval"]["n_draws"],
                                                     draw_size=cfg["eval"]["draw_size"])
            print("rag_ft_core:", results["rag_ft_core"].get("f1"))

            # (5) + both
            if "rag_ft_embedding" in results:
                results["rag_ft_both"] = evaluate_config(make_pipe(ft_retr, reranker, ft_llm), qa_test,
                                                         use_rag=True, cosine=cosine,
                                                         n_draws=cfg["eval"]["n_draws"],
                                                         draw_size=cfg["eval"]["draw_size"])
                print("rag_ft_both:", results["rag_ft_both"].get("f1"))

    save_json(results, out / "ablation_results.json")
    save_json(environment_info(), out / "environment.json")
    print_table1(results)
    print(f"\nSaved to {out}")


def _bm25_wrap(retr):
    class W:
        def search(self, q, k=5):
            return retr.search(q, k)

        def search_mmr(self, q, k=10, fetch_k=20, lambda_mult=0.5):
            return retr.search(q, k)

    return W()


def print_table1(results):
    order = ["baseline_no_rag", "raw_rag", "rag_ft_embedding", "rag_ft_core", "rag_ft_both"]
    labels = {"baseline_no_rag": "Baseline (w/o RAG)", "raw_rag": "Raw RAG Pipeline",
              "rag_ft_embedding": "+ Embedding", "rag_ft_core": "+ Core Model Finetune",
              "rag_ft_both": "+ Embedding & Core"}
    print("\nTable 1 reproduction")
    print(f"{'Configuration':24}{'Recall':>10}{'F1':>10}{'Cosine':>10}{'BLEU':>10}")
    for key in order:
        if key not in results:
            continue
        r = results[key]
        def g(m):
            return f"{r[m]['mean']:.3f}" if m in r else "-"
        print(f"{labels[key]:24}{g('recall'):>10}{g('f1'):>10}{g('cosine'):>10}{g('bleu'):>10}")


if __name__ == "__main__":
    main()
