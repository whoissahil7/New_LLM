"""
pipeline.py — controlled S1/S2/S3 comparison and evaluation.

One code path expresses all three systems (the property that licenses causal
attribution):
  S1  retrieval -> select top-N by RRF order            (no rerank, no faith)
  S2  retrieval -> cross-encoder rerank -> top-N         (relevance only; beta=0)
  S3  retrieval -> rerank -> faithfulness -> fuse+gate -> top-N   (contribution)

Everything else (corpus, chunking, index, K, N, generator, prompt, decoding,
seeds, evaluation) is held identical.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import core
from data import SciFact, Claim, SUPPORTED, REFUTED, NEI


# ---------------------------------------------------------------------------
# Indexing (abstract/doc granularity, to match BEIR qrels)
# ---------------------------------------------------------------------------
def build_indexes(sf: SciFact, dense, bm25):
    ids = list(sf.corpus.keys())
    texts = [sf.doc_text(d) for d in ids]
    dense.build(ids, texts)
    bm25.build(ids, texts)
    return dense, bm25


def hybrid_retrieve(dense, bm25, query: str, k: int, rrf_k: int = 60) -> List[str]:
    dense_ids = dense.search(query, k)
    bm25_ids = bm25.search(query, k)
    fused = core.reciprocal_rank_fusion([dense_ids, bm25_ids], k=rrf_k)
    return core.top_k_from_scores(fused, k)


# ---------------------------------------------------------------------------
# Per-query record
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    cid: str
    candidates: List[str]                      # top-K after retrieval
    ranked: List[str]                          # candidate order after (re)ranking
    selected: List[str]                        # final top-N sent to the LLM
    generation: dict                           # {verdict, cited_ids, rationale}
    faith_entail_of_answer: float = 0.0        # NLI entailment of rationale by context
    latencies: Dict[str, float] = field(default_factory=dict)


_VERDICTS = {"SUPPORTED": SUPPORTED, "REFUTED": REFUTED, "NOT ENOUGH INFO": NEI}


def parse_generation(text: str) -> dict:
    """Robustly extract {verdict, cited_ids, rationale} from model output."""
    verdict, cited, rationale = NEI, [], ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).upper().strip()
            verdict = _VERDICTS.get(v, NEI)
            cited = [str(c) for c in obj.get("cited_ids", [])]
            rationale = str(obj.get("rationale", ""))
            return {"verdict": verdict, "cited_ids": cited, "rationale": rationale}
        except json.JSONDecodeError:
            pass
    up = text.upper()
    for k, v in _VERDICTS.items():
        if k in up:
            verdict = v
            break
    return {"verdict": verdict, "cited_ids": cited, "rationale": rationale or text[:200]}


# ---------------------------------------------------------------------------
# The three-systems runner
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, sf: SciFact, dense, bm25, reranker, faith_scorer, generator,
                 nli, cfg):
        self.sf = sf
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.faith = faith_scorer
        self.gen = generator
        self.nli = nli
        self.cfg = cfg

    def _select_context(self, ranked_ids: List[str], fused_scores: np.ndarray) -> List[str]:
        n = self.cfg["N"]
        cand = ranked_ids[: max(n * 4, n)]           # small pool for MMR
        vecs = self.dense.vecs_for(cand)
        rel = core.minmax(fused_scores[: len(cand)])
        picks = core.mmr(vecs, rel, k=n, lambda_mmr=self.cfg["lambda_mmr"])
        chosen = [cand[i] for i in picks]
        chosen = core.order_lost_in_the_middle(chosen)
        texts = [self.sf.doc_text(d) for d in chosen]
        keep = core.enforce_token_budget(texts, self.cfg["token_budget"])
        return [chosen[i] for i in keep]

    def run_query(self, claim: Claim, system: str) -> QueryResult:
        lat: Dict[str, float] = {}
        q = claim.text

        t = time.perf_counter()
        candidates = hybrid_retrieve(self.dense, self.bm25, q, self.cfg["K"],
                                     self.cfg["rrf_k"])
        lat["retrieval"] = time.perf_counter() - t
        texts = [self.sf.doc_text(d) for d in candidates]

        if system == "S1":
            ranked = candidates
            fused = np.arange(len(candidates), 0, -1, dtype=float)  # RRF order proxy
        else:
            t = time.perf_counter()
            rel = self.reranker.score(q, texts)
            lat["rerank"] = time.perf_counter() - t
            if system == "S2":
                fused, keep = core.fuse_and_gate(rel, None, alpha=1.0, beta=0.0)
            else:  # S3
                t = time.perf_counter()
                fsig = self.faith.score(q, texts)
                lat["faithfulness"] = time.perf_counter() - t
                fused, keep = core.fuse_and_gate(
                    rel, fsig["faith"], self.cfg["alpha"], self.cfg["beta"],
                    tau_f=self.cfg["tau_f"], lam=self.cfg["lam"], gate=self.cfg["gate"])
                if self.cfg["gate"] == "hard":
                    candidates = [c for c, k in zip(candidates, keep) if k]
                    fused = fused[keep]
            order = np.argsort(fused)[::-1]
            ranked = [candidates[i] for i in order]
            fused = fused[order]

        t = time.perf_counter()
        selected = self._select_context(ranked, np.asarray(fused, float))
        lat["select"] = time.perf_counter() - t

        # -- grounded generation --
        from models import build_prompt
        evidence = [(d, self.sf.doc_text(d)) for d in selected]
        t = time.perf_counter()
        raw = self.gen.generate(build_prompt(q, evidence))
        lat["generation"] = time.perf_counter() - t
        gen = parse_generation(raw)

        # -- answer faithfulness: is the rationale entailed by any selected passage? --
        faith_ans = 0.0
        if selected and gen["rationale"]:
            ent = self.nli.entailment([self.sf.doc_text(d) for d in selected],
                                      [gen["rationale"]] * len(selected))
            faith_ans = float(np.max(ent))

        return QueryResult(claim.cid, candidates, ranked, selected, gen,
                           faith_ans, lat)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _percentiles(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, float)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}


def evaluate(pipe: Pipeline, claims: Dict[str, Claim], qrels: Dict[str, Dict[str, int]],
             system: str, faith_entail_threshold: float = 0.5) -> dict:
    cfg = pipe.cfg
    per_query = []
    for cid, claim in claims.items():
        r = pipe.run_query(claim, system)
        rel_ids = set(qrels.get(cid, {}).keys())
        rel_gain = {d: float(g) for d, g in qrels.get(cid, {}).items()}

        rec_k = core.recall_at_k(r.candidates, rel_ids, cfg["K"]) if rel_ids else float("nan")
        p_n = core.precision_at_k(r.selected, rel_ids, cfg["N"]) if rel_ids else float("nan")
        mrr = core.reciprocal_rank(r.ranked, rel_ids) if rel_ids else float("nan")
        ndcg = core.ndcg_at_k(r.ranked, rel_gain, 10) if rel_gain else float("nan")

        correct = (r.generation["verdict"] == claim.gold_label)
        phantom = any(c not in set(r.selected) for c in r.generation["cited_ids"])
        halluc = (claim.gold_label != NEI) and (phantom or r.faith_entail_of_answer < faith_entail_threshold)

        per_query.append({
            "cid": cid, "gold": claim.gold_label, "verdict": r.generation["verdict"],
            "recall_at_K": rec_k, "precision_at_N": p_n, "mrr": mrr, "ndcg10": ndcg,
            "correct": correct, "faith": r.faith_entail_of_answer, "halluc": halluc,
            "answerable": claim.gold_label != NEI,
            "lat": r.latencies, "result": r,
        })

    ans = [q for q in per_query if q["answerable"]]
    unans = [q for q in per_query if not q["answerable"]]

    def _mean(rows, key):
        vals = [x[key] for x in rows if not (isinstance(x[key], float) and np.isnan(x[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    stages = ["retrieval", "rerank", "faithfulness", "select", "generation"]
    latency = {s: _percentiles([q["lat"][s] for q in per_query if s in q["lat"]])
               for s in stages if any(s in q["lat"] for q in per_query)}

    return {
        "system": system,
        "recall_at_K": _mean(per_query, "recall_at_K"),
        "precision_at_N": _mean(per_query, "precision_at_N"),
        "mrr": _mean(per_query, "mrr"),
        "ndcg10": _mean(per_query, "ndcg10"),
        "answer_correctness": _mean(per_query, "correct"),
        "faithfulness": _mean(ans, "faith"),
        "hallucination_rate": _mean(ans, "halluc"),
        "abstention_correctness": (float(np.mean([q["verdict"] == NEI for q in unans]))
                                   if unans else float("nan")),
        "latency": latency,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Error taxonomy (rule-based; one bucket per failing query)
# ---------------------------------------------------------------------------
BUCKETS = ["retrieval_miss", "ranking_error", "faithfulness_scoring_error",
           "fusion_error", "context_assembly_error", "generation_error",
           "annotation_error"]


def bucket_failures(eval_out: dict, claims: Dict[str, Claim], cfg: dict) -> Dict[str, int]:
    counts = {b: 0 for b in BUCKETS}
    system = eval_out["system"]
    for q in eval_out["per_query"]:
        failed = (not q["correct"]) or q["halluc"]
        if not failed:
            continue
        claim = claims[q["cid"]]
        gold_docs = set(claim.evidence_docs)
        r = q["result"]
        if not gold_docs:                       # NEI claim answered wrong -> generation
            counts["generation_error"] += 1
            continue
        in_cand = gold_docs & set(r.candidates)
        in_final = gold_docs & set(r.selected)
        if not in_cand:
            counts["retrieval_miss"] += 1
        elif not in_final:
            # gold retrieved but didn't survive to the context window
            counts["fusion_error" if system == "S3" else "ranking_error"] += 1
        else:
            counts["generation_error"] += 1
    return counts
