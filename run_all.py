#!/usr/bin/env python
"""
run_all.py — reproduce the full S1/S2/S3 experiment on SciFact.

This orchestrates the experiment and writes real measurements to results/.
It does NOT contain any hard-coded metric values; every number it prints is
computed at run time from SciFact via the pipeline. Requires network + a GPU
(to download SciFact and the four models) — see README for the environment.

Usage:
    python run_all.py --config config.yaml [--limit 50] [--skip-ablation]
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import core                     # noqa: E402
import data as data_mod         # noqa: E402
import pipeline as pl           # noqa: E402


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_everything(cfg):
    from models import DenseIndex, BM25Index, CrossEncoderReranker, NLIModel, GroundedGenerator
    from faithfulness import FaithfulnessScorer

    print("[run] loading SciFact ...")
    sf = data_mod.load_scifact(cfg["cache_dir"])
    print(f"[run] corpus={len(sf.corpus)} dev_claims={len(sf.dev_claims)} "
          f"test_claims={len(sf.test_claims)}")

    dense = DenseIndex(cfg["dense_model"])
    bm25 = BM25Index()
    pl.build_indexes(sf, dense, bm25)

    reranker = CrossEncoderReranker(cfg["reranker_model"])
    nli = NLIModel(cfg["nli_model"])
    faith = FaithfulnessScorer(
        nli, weights={k: cfg[k] for k in ("w1", "w2", "w3", "w4")},
        use_source_reliability=cfg["use_source_reliability"],
        agreement_cap=cfg["agreement_cap"])
    gen = GroundedGenerator(cfg["generator_backend"], cfg["generator_model"],
                            cfg["temperature"], cfg["max_new_tokens"])

    pipe = pl.Pipeline(sf, dense, bm25, reranker, faith, gen, nli, cfg)
    return sf, pipe


def check_retrieval_gate(pipe, sf, cfg):
    """The architecture's precondition: recall@K on dev must clear the gate."""
    recs = []
    for cid, claim in sf.dev_claims.items():
        rel = set(sf.dev_qrels.get(cid, {}).keys())
        if not rel:
            continue
        cand = pl.hybrid_retrieve(pipe.dense, pipe.bm25, claim.text, cfg["K"], cfg["rrf_k"])
        recs.append(core.recall_at_k(cand, rel, cfg["K"]))
    r = float(np.mean(recs)) if recs else float("nan")
    gate = cfg["retrieval_gate_recall_at_K"]
    print(f"[gate] dev recall@{cfg['K']} = {r:.3f} (gate {gate}) -> "
          f"{'PASS' if r >= gate else 'FAIL — fix retrieval before comparing rerankers'}")
    return r


def paired_stats(test_results):
    """Paired tests S3 vs S2 on identical queries, Holm-corrected."""
    by_cid = {}
    for sysname in ("S2", "S3"):
        for q in test_results[sysname]["per_query"]:
            by_cid.setdefault(q["cid"], {})[sysname] = q
    common = [c for c, d in by_cid.items() if "S2" in d and "S3" in d]

    # continuous metrics -> Wilcoxon + bootstrap
    pvals, effects = {}, {}
    for metric in ("faith", "ndcg10", "precision_at_N"):
        a = [by_cid[c]["S3"][metric] for c in common]
        b = [by_cid[c]["S2"][metric] for c in common]
        a = np.nan_to_num(np.array(a, float)); b = np.nan_to_num(np.array(b, float))
        _, p = core.wilcoxon_signed_rank(a, b)
        pvals[f"{metric}_wilcoxon"] = p
        effects[metric] = core.paired_bootstrap_diff(a, b, seed=0)

    # hallucination (binary, answerable) -> McNemar
    ans = [c for c in common if by_cid[c]["S2"]["answerable"]]
    b_only = sum(1 for c in ans if not by_cid[c]["S2"]["halluc"] and by_cid[c]["S3"]["halluc"])
    c_only = sum(1 for c in ans if by_cid[c]["S2"]["halluc"] and not by_cid[c]["S3"]["halluc"])
    stat, p = core.mcnemar_test(b_only, c_only)
    pvals["hallucination_mcnemar"] = p

    corrected = core.holm_bonferroni(pvals)
    return {"n_common": len(common), "corrected": corrected, "effects": effects,
            "mcnemar_discordant": {"s3_worse": b_only, "s3_better": c_only}}


def run_ablation(pipe, sf, cfg):
    """Ablations on DEV: drop each sub-signal, gate soft vs hard, alpha sweep."""
    dev, qrels = sf.dev_claims, sf.dev_qrels
    out = {}

    base_w = {k: cfg[k] for k in ("w1", "w2", "w3", "w4")}
    for drop in (None, "w1", "w2", "w3"):
        w = dict(base_w)
        if drop:
            w[drop] = 0.0
        pipe.faith.w = w
        tag = "full" if drop is None else f"drop_{drop}"
        out[f"subsignal::{tag}"] = _slim(pl.evaluate(pipe, dev, qrels, "S3"))
    pipe.faith.w = base_w

    for gate in ("soft", "hard", "off"):
        pipe.cfg["gate"] = gate
        out[f"gate::{gate}"] = _slim(pl.evaluate(pipe, dev, qrels, "S3"))
    pipe.cfg["gate"] = cfg["gate"]

    for alpha in [round(a, 2) for a in np.linspace(0.0, 1.0, 11)]:
        pipe.cfg["alpha"], pipe.cfg["beta"] = alpha, round(1 - alpha, 2)
        out[f"alpha::{alpha:.2f}"] = _slim(pl.evaluate(pipe, dev, qrels, "S3"))
    pipe.cfg["alpha"], pipe.cfg["beta"] = cfg["alpha"], cfg["beta"]
    return out


def _slim(e):
    return {k: e[k] for k in ("faithfulness", "hallucination_rate", "ndcg10",
                              "precision_at_N", "answer_correctness")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="cap #test claims (debug)")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    np.random.seed(cfg["seed"])
    os.makedirs("results", exist_ok=True)

    sf, pipe = build_everything(cfg)
    gate = check_retrieval_gate(pipe, sf, cfg)

    test = dict(list(sf.test_claims.items())[: args.limit]) if args.limit else sf.test_claims

    results = {}
    for system in ("S1", "S2", "S3"):
        print(f"[run] evaluating {system} on {len(test)} test claims ...")
        results[system] = pl.evaluate(pipe, test, sf.test_qrels, system,
                                      cfg["faith_entail_threshold"])
        errs = pl.bucket_failures(results[system], test, cfg)
        results[system]["error_buckets"] = errs
        print(f"       {_slim(results[system])}")
        print(f"       errors: {errs}")

    stats = paired_stats(results)
    print(f"[stats] S3 vs S2 (Holm-corrected): "
          f"{json.dumps(stats['corrected'], indent=2)}")

    ablation = {} if args.skip_ablation else run_ablation(pipe, sf, cfg)

    # strip non-serialisable QueryResult objects before saving
    for s in results.values():
        for q in s["per_query"]:
            q.pop("result", None)
            q.pop("lat", None)

    payload = {"config": cfg, "dev_recall_gate": gate,
               "results": results, "stats": stats, "ablation": ablation}
    with open("results/results.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print("[run] wrote results/results.json")


if __name__ == "__main__":
    main()
