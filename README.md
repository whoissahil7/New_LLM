# Faithfulness-Aware Reranking for Domain-Specific RAG — SciFact implementation

A complete, runnable implementation of the architecture "Faithfulness-Aware
Reranking for Domain-Specific RAG," instantiated on **SciFact**. It runs the
controlled three-system comparison (S1 baseline, S2 relevance reranker, S3
faithfulness-aware) and produces real retrieval, faithfulness, hallucination,
correctness, abstention, and latency measurements with paired significance
tests, an ablation grid, and an error taxonomy.

See `RESEARCH_DESIGN.md` for the hypothesis, dataset justification, and the
honest adaptations (notably: source-reliability is disabled on SciFact because
the corpus has no publisher/date/version metadata, and the weights are
renormalised — the code logs this).

## Honest status

- **Pure-logic core (`src/core.py`)** — rank fusion, all IR metrics, score
  fusion + gate, MMR/context ordering, the self-containment sub-signal, and the
  statistics — is **unit-tested and passing** (`tests/test_core.py`, 9 tests) on
  numpy/scipy alone.
- **Model-dependent stages** (dense encoder, BM25, cross-encoder, NLI, generator,
  data download) are implemented against **verified** model/dataset APIs but were
  **not executed** in the environment where this was written (no network, no GPU,
  no ML stack). Expect to shake out minor environment-specific issues on the
  first real run; nothing here is claimed to have produced results yet.
- **No experimental numbers appear anywhere** in the code or docs. Results exist
  only after you run `run_all.py`.

## Environment

- Python 3.10+
- A GPU is strongly recommended (the per-candidate NLI pass and the generator are
  the cost drivers).
- Network access on first run to download SciFact and the four models.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the offline tests (no downloads needed)

```bash
python tests/test_core.py
# or: python -m pytest tests/test_core.py
```

## Reproduce the experiment

```bash
# 1) (optional) tune alpha/beta, weights, gate on the DEV split
python tune.py --config config.yaml --trials 50
#    -> copy results/tuned_params.json values into config.yaml

# 2) run S1/S2/S3 on TEST, with stats + ablation + error taxonomy
python run_all.py --config config.yaml
#    -> writes results/results.json
#    quick smoke test on a few claims:  python run_all.py --limit 20 --skip-ablation
```

`run_all.py` first checks the retrieval gate (dev recall@K vs ≈0.85) and refuses
to treat the reranking comparison as valid if it fails — reranking cannot recover
evidence retrieval never returned.

## Layout

```
config.yaml            all params (architecture defaults; marked tunable/fixed)
run_all.py             end-to-end: gate -> S1/S2/S3 on test -> stats -> ablation -> errors
tune.py                Optuna dev-set tuning
src/core.py            dependency-light logic (unit-tested)
src/models.py          dense/BM25/reranker/NLI/generator wrappers (verified ids)
src/data.py            SciFact + BEIR loading, dev/test + answerable/unanswerable splits
src/faithfulness.py    the four sub-signals + weight renormalisation
src/pipeline.py        S1/S2/S3 runner, metric battery, error taxonomy
tests/test_core.py     9 passing sanity tests (toy inputs, not results)
RESEARCH_DESIGN.md     hypothesis, dataset rationale, caveats, metrics, protocol, refs
```

## What to report from a run

Primary: hallucination rate and faithfulness (S3 vs S2), with the McNemar /
Wilcoxon + bootstrap outputs and Holm-corrected p-values from `results.json`.
Diagnostic: Recall@K (gate), Precision@N, MRR, nDCG@10. Constraint: p50/p95
latency per stage. A null or latency-limited result is a valid finding — the
design commits to reporting it either way.
