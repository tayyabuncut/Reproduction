# Results

This folder holds the experimental outputs and figures (an assignment deliverable).

- `figures/figures_poisoning.png` — Baseline 1 retrieval-poisoning: ΔMRR@10 vs. poison budget (Vanilla vs.
  Sanitized+Normalized). Extracted from the executed notebook.

## Add your run outputs here before submitting

Download the `results/` folder from the Colab session and drop these files in alongside the figure:

- `b2_table1.csv` — Baseline 2 ablation (Recall / F1 / Cosine / BLEU)
- `b2_ablation_results.json` — full Baseline 2 results
- `b1_asr_table.csv` — Baseline 1 attack-success rate per carrier × defense
- `b1_injection_results.json` — full Baseline 1 injection results
- `b1_detector_validation.json` — ASR detector P/R/F1/κ
- `b1_latency_utility.csv` — per-stage latency + utility
- `environment.json` — GPU / library versions captured at run time

The executed notebook (`notebooks/Assignment2_Reproduction.ipynb`) also shows every table and figure inline
in its cell outputs.
