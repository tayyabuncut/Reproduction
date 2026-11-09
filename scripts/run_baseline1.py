#!/usr/bin/env python3
"""Run the full Baseline 1 (OpenRAG-Soc) reproduction and write results + figures.

Usage:
    python scripts/run_baseline1.py --config configs/baseline1.yaml --out results/b1
    # quick smoke run on CPU with the mock LLM (plumbing only, not reproduction numbers):
    python scripts/run_baseline1.py --config configs/baseline1.yaml --out results/b1_smoke --mock --scale 0.05
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import b1_corpus
from src.b1_harness import run_injection_eval, run_poisoning_eval, CONFIGS, PAPER_CONFIG_ORDER
from src.b1_detector import validate_detector
from src.utils import environment_info, load_config, save_json, save_jsonl, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline1.yaml")
    ap.add_argument("--out", default="results/b1")
    ap.add_argument("--mock", action="store_true", help="use MockLLM (CPU smoke test only)")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--scale", type=float, default=None, help="override corpus scale")
    ap.add_argument("--no-poison", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scale = args.scale if args.scale is not None else cfg["corpus"]["scale"]
    pages = b1_corpus.build_corpus(seed=cfg["corpus"]["seed"], n_hard_negatives=cfg["corpus"]["n_hard_negatives"],
                                   n_svg=cfg["corpus"]["n_svg"], scale=scale)
    stats = b1_corpus.corpus_stats(pages)
    print("Corpus:", stats)
    save_json(stats, out / "corpus_stats.json")
    save_jsonl([{k: v for k, v in b1_corpus.asdict(p).items() if k not in ("html_payload", "html_control")}
                for p in pages], out / "corpus_index.jsonl")

    # ---- LLM
    tokenizer = None
    if args.mock:
        from src.generation import MockLLM
        llm = MockLLM()
    else:
        from src.generation import HFGenerator
        llm = HFGenerator(cfg["llm"]["model_id"], load_in_4bit=cfg["llm"]["load_in_4bit"], hf_token=args.hf_token)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["dense_model"], token=args.hf_token) \
            if cfg["retriever"] == "dense" else None

    dense_model = None
    if cfg["retriever"] == "dense" and not args.mock:
        from sentence_transformers import SentenceTransformer
        dense_model = SentenceTransformer(cfg["dense_model"])

    # ---- injection eval
    configs = {k: CONFIGS[k] for k in cfg["configs"]}
    log_rows: list = []
    inj = run_injection_eval(pages, llm, configs=configs, retriever_kind=cfg["retriever"], dense_model=dense_model,
                             k=cfg["k"], chunk_size=cfg["chunk_size"], overlap=cfg["overlap"],
                             max_new_tokens=cfg["llm"]["max_new_tokens"], temperature=cfg["llm"]["temperature"],
                             attribution_retries=cfg["attribution_retries"], tokenizer=tokenizer,
                             seed=cfg.get("seed", 0), log_rows=log_rows)
    save_json(inj, out / "injection_results.json")
    save_jsonl(log_rows, out / "injection_generations.jsonl")
    print_asr_table(inj)

    # detector validation aggregated across all configs' logged rows
    det = validate_detector(log_rows)
    save_json(det, out / "detector_validation.json")
    print("Detector vs gold:", det)

    # ---- poisoning sweep
    if cfg["poisoning"]["run"] and not args.no_poison:
        poison = run_poisoning_eval(
            pages, poison_fractions=tuple(cfg["poisoning"]["fractions"]),
            retriever_kinds=tuple(cfg["poisoning"]["retrievers"]) if not args.mock else ("bm25",),
            dense_model=dense_model, configs_to_compare=tuple(cfg["poisoning"]["compare"]),
            k=cfg["poisoning"]["k"], chunk_size=cfg["chunk_size"], overlap=cfg["overlap"],
            tokenizer=tokenizer, n_eval_queries=cfg["poisoning"]["n_eval_queries"])
        save_json(poison, out / "poisoning_results.json")
        try:
            plot_poisoning(poison, out / "figures")
        except Exception as e:
            print("plot skipped:", e)

    save_json(environment_info(), out / "environment.json")
    print(f"\nSaved to {out}")


def print_asr_table(inj):
    carriers = ["hidden_span", "offscreen_css", "alt_text", "aria", "zero_width"]
    cfgs = [c for c in PAPER_CONFIG_ORDER if c in inj]
    print("\nIPI ASR (%)  [carrier x config]")
    header = f"{'carrier':16}" + "".join(f"{c:>14}" for c in cfgs)
    print(header)
    for cr in carriers:
        row = f"{cr:16}"
        for c in cfgs:
            cell = inj[c].get(cr)
            row += f"{cell['asr']:>13.1f} " if cell else f"{'-':>14}"
        print(row)
    row = f"{'macro':16}"
    for c in cfgs:
        row += f"{inj[c]['macro']['asr']:>13.1f} "
    print(row)


def plot_poisoning(poison, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for rk, cfgs in poison.items():
        for cfg, d in cfgs.items():
            plt.plot([f * 100 for f in d["fraction"]], d["dMRR@10"], marker="o", label=f"{rk} ({cfg})")
    plt.xlabel("Poison fraction (%)")
    plt.ylabel(r"$\Delta$MRR@10 vs. clean")
    plt.title("Poison budget vs. rank impact")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "poisoning_dmrr.png", dpi=150)
    print(f"Figure -> {figdir/'poisoning_dmrr.png'}")


if __name__ == "__main__":
    main()
