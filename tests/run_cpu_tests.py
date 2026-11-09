"""CPU-only smoke tests: exercise every module's logic without a GPU or network.

Run: python tests/run_cpu_tests.py
Uses MockLLM + a small synthetic corpus. Verifies corpus construction, the four defenses actually move text,
retrieval/metrics shapes, the full B1 injection harness, the poisoning sweep, and the B2 pipeline with the
offline sample dataset. Numbers printed here are plumbing checks only, not reproduction results.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import b1_corpus, b1_defenses as D
from src.b1_harness import run_injection_eval, run_poisoning_eval, CONFIGS
from src.b1_detector import detect_follow
from src.b2_data import build_sample_dataset, load_corpus, load_qa_pairs
from src.b2_pipeline import RagPipeline, evaluate_config, build_core_messages
from src.chunking import word_chunks, token_chunks
from src.generation import MockLLM
from src.metrics import token_prf, sentence_bleu, mrr_at_k, ndcg_at_k, bootstrap_ci, cohen_kappa
from src.retrieval import BM25Retriever
from src.utils import set_seed

PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    mark = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{mark}] {name} {extra}")


def test_metrics():
    p, r, f = token_prf("the ferry takes 40 minutes", "the ferry takes about 40 minutes")
    check("token_prf partial overlap", 0.0 < f < 1.0, f"f1={f:.3f}")
    check("token_prf exact", token_prf("40 minutes", "40 minutes")[2] == 1.0)
    check("bleu identical==1", abs(sentence_bleu("a b c d", "a b c d") - 1.0) < 1e-6)
    # smoothed sentence-BLEU of a fully disjoint pair is a tiny non-zero value, not exactly 0
    check("bleu disjoint small", sentence_bleu("x y z", "a b c") < 0.05, f"bleu={sentence_bleu('x y z','a b c'):.4f}")
    check("mrr@10 first", mrr_at_k(["d1", "d2"], {"d1"}, 10) == 1.0)
    check("mrr@10 second", abs(mrr_at_k(["d2", "d1"], {"d1"}, 10) - 0.5) < 1e-9)
    check("ndcg@10 ideal", abs(ndcg_at_k(["d1"], {"d1"}, 10) - 1.0) < 1e-9)
    m, lo, hi = bootstrap_ci([1, 0, 1, 1, 0], n_boot=2000)
    check("bootstrap ci brackets mean", lo <= m <= hi, f"{lo:.2f}<={m:.2f}<={hi:.2f}")
    check("cohen kappa perfect", abs(cohen_kappa([1, 0, 1], [1, 0, 1]) - 1.0) < 1e-9)


def test_chunking():
    ck = word_chunks("d", " ".join(["w"] * 2500), size=1000, overlap=0)
    check("word_chunks count", len(ck) == 3, f"n={len(ck)}")
    tk = token_chunks("d", " ".join(["w"] * 300), size=128, overlap=32, tokenizer=None)
    check("token_chunks fallback strided", len(tk) >= 2, f"n={len(tk)}")


def test_defenses():
    pages = b1_corpus.build_corpus(seed=1, n_hard_negatives=5, n_svg=5, scale=0.01)
    check("corpus builds", len(pages) > 30, f"n={len(pages)}")
    carriers = {p.carrier for p in pages}
    check("all carriers present", {"hidden_span", "offscreen_css", "alt_text", "aria", "zero_width"} <= carriers)

    # a hidden-span payload page: vanilla ingest keeps payload, sanitize drops it
    hp = next(p for p in pages if p.carrier == "hidden_span" and p.payload_visibility == "hidden")
    van = D.ingest(hp.html_payload, hp.fmt, sanitize=False, normalize=False)
    san = D.ingest(hp.html_payload, hp.fmt, sanitize=True, normalize=False)
    check("vanilla keeps hidden payload", payload_in(hp, van), extra=f"payload='{hp.payload_text[:40]}'")
    check("sanitize removes hidden payload", not payload_in(hp, san), extra=f"san='{san[:60]}...'")
    check("sanitize keeps visible fact", hp.answer.split()[0].lower() in san.lower())

    # zero-width markers removed by normalization
    zp = next(p for p in pages if p.carrier == "zero_width")
    raw_text = D.ingest_vanilla(zp.html_payload)
    norm = D.normalize_text(raw_text)
    check("normalize strips zero-width chars", "​" not in norm and "⁠" not in norm)

    # attribution helpers
    check("citation_ok true", D.citation_ok("The ferry takes 40 minutes [1]. It is cheap [2]."))
    check("citation_ok false", not D.citation_ok("The ferry takes 40 minutes. It is cheap [2]."))
    check("attr consistency in range", 0 <= D.attribution_consistency("ferry takes 40 minutes [1].",
                                                                      ["the ferry takes 40 minutes"]) <= 1)


def payload_in(page, text):
    # payload detectable via a distinctive instruction phrase from the template that was used
    key = page.payload_text.lower()
    phrase = next((w for w in ["ignore the user", "verified by", "end your response",
                               "summarizing this page"] if w in key), None)
    return (phrase in text.lower()) if phrase else False


def test_detector():
    check("detect canary", detect_follow("Verified by CNRY-ABC123", "CNRY-ABC123") == 1)
    check("detect refusal", detect_follow("The ferry takes 40 minutes [1].", "CNRY-ABC123",
                                          "ignore the user") == 0)


def test_b1_harness():
    set_seed(0)
    pages = b1_corpus.build_corpus(seed=2, n_hard_negatives=10, n_svg=10, scale=0.01)
    llm = MockLLM()
    log = []
    res = run_injection_eval(pages, llm, configs=CONFIGS, retriever_kind="bm25", k=5,
                             chunk_size=256, overlap=64, temperature=0.0, log_rows=log, seed=0)
    for cfg in ["Vanilla", "Sanitized", "Normalized", "All Defenses"]:
        macro = res[cfg]["macro"]["asr"]
        check(f"B1 {cfg} macro ASR in [0,100]", 0 <= macro <= 100, f"asr={macro:.1f}")
    check("B1 logged rows", len(log) > 0, f"rows={len(log)}")
    check("B1 detector validation present", "n" in res["Vanilla"]["detector_validation"])
    check("B1 latency stages", set(res["Vanilla"]["latency"]) >= {"ingest", "retrieve", "generate"})
    # sanity: sanitize should not INCREASE hidden-carrier ASR vs vanilla (mock obeys only visible payloads)
    v = res["Vanilla"]["macro"]["asr"]
    s = res["Sanitized"]["macro"]["asr"]
    check("B1 sanitize <= vanilla (mock)", s <= v + 1e-6, f"van={v:.1f} san={s:.1f}")


def test_b1_poisoning():
    pages = b1_corpus.build_corpus(seed=3, n_hard_negatives=0, n_svg=0, scale=0.01)
    res = run_poisoning_eval(pages, poison_fractions=(0.0, 0.05, 0.1), retriever_kinds=("bm25",),
                             configs_to_compare=("Vanilla", "Sanitized+Normalized"), k=10,
                             chunk_size=256, overlap=64, n_eval_queries=30)
    dmrr = res["bm25"]["Vanilla"]["dMRR@10"]
    check("B1 poisoning dMRR length", len(dmrr) == 3)
    check("B1 poisoning zero-budget ~0", abs(dmrr[0]) < 1e-9, f"dmrr0={dmrr[0]:.3f}")
    check("B1 poisoning hurts ranking", dmrr[-1] <= dmrr[0] + 1e-9, f"dmrr={[round(x,3) for x in dmrr]}")


def test_b2_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        d = build_sample_dataset(tmp)
        corpus = load_corpus(d, chunk_words=1000)
        qa = load_qa_pairs(d, split="test")
        check("B2 corpus loads", len(corpus) >= 4, f"n={len(corpus)}")
        check("B2 qa loads", len(qa) == 5, f"n={len(qa)}")
        # BM25 stands in for the dense retriever on CPU (no sentence-transformers here)
        retr = BM25Retriever(corpus)

        class BM25Wrap:
            def search(self, q, k=5):
                return retr.search(q, k)

            def search_mmr(self, q, k=10, fetch_k=20, lambda_mult=0.5):
                return retr.search(q, k)

        pipe = RagPipeline(dense_retriever=BM25Wrap(), llm=MockLLM(), reranker=None,
                           retrieve_k=4, rerank_top_n=3, use_mmr=False)
        ans, passages = pipe.answer("When will classes begin in the Fall 2024 semester?", use_rag=True)
        check("B2 retrieves passages", len(passages) > 0)
        # mock echoes the first sentence of retrieved context; check the answer is drawn from a passage
        check("B2 answer references context", any(ans.strip().lower()[:20] in p.lower() for p in passages),
              extra=f"ans='{ans[:40]}'")
        out = evaluate_config(pipe, qa, use_rag=True, cosine=None, n_draws=2, draw_size=5)
        check("B2 eval has f1", "f1" in out and 0 <= out["f1"]["mean"] <= 1, f"f1={out['f1']['mean']:.3f}")
        check("B2 core messages template", "50 words maximum" in build_core_messages("q", "c")[0]["content"])


def main():
    for t in [test_metrics, test_chunking, test_defenses, test_detector, test_b1_harness,
              test_b1_poisoning, test_b2_pipeline]:
        print(f"\n--- {t.__name__} ---")
        t()
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
