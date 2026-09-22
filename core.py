"""
core.py — dependency-light logic for the faithfulness-aware reranking pipeline.

Everything here runs on numpy/scipy only (no torch, transformers, faiss, or
network). These are the deterministic stages of the architecture and the
evaluation machinery, kept separate precisely so they can be unit-tested in
isolation from the model-dependent stages in models.py.

Sections
  1. Rank fusion (RRF)
  2. Retrieval metrics (recall@K, precision@N, MRR, nDCG@K)
  3. Score fusion + faithfulness gate
  4. Context selection (MMR + lost-in-the-middle ordering + token budget)
  5. Self-containment sub-signal (deterministic proxy)
  6. Statistics (McNemar, Wilcoxon, paired bootstrap, Holm-Bonferroni, Cliff's delta)
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
from scipy import stats as _st


# ---------------------------------------------------------------------------
# 1. Rank fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    rankings: list of ranked id-lists (e.g. [dense_ids, bm25_ids]), each ordered
              best-first.
    Returns {doc_id: fused_score}, higher is better. k is the RRF constant.
    """
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def top_k_from_scores(scores: Dict[str, float], k: int) -> List[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


# ---------------------------------------------------------------------------
# 2. Retrieval metrics (qrels-based)
# ---------------------------------------------------------------------------
def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return float("nan")
    hits = sum(1 for d in ranked_ids[:k] if d in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for d in ranked_ids[:k] if d in relevant_ids)
    return hits / k


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set) -> float:
    for i, d in enumerate(ranked_ids, start=1):
        if d in relevant_ids:
            return 1.0 / i
    return 0.0


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: Sequence[str], rel_gain: Dict[str, float], k: int) -> float:
    """nDCG@k with graded gains. rel_gain maps doc_id -> relevance gain (>=0)."""
    gains = [rel_gain.get(d, 0.0) for d in ranked_ids[:k]]
    dcg = _dcg(gains)
    ideal = _dcg(sorted(rel_gain.values(), reverse=True)[:k])
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# 3. Score fusion + faithfulness gate
# ---------------------------------------------------------------------------
def minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0,1] within the candidate set.

    Cross-encoder and NLI logits are uncalibrated, so the architecture
    normalises each score column within the candidate set before fusion.
    A degenerate (all-equal) column maps to 0.5 to avoid divide-by-zero.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def fuse_and_gate(
    rel_raw: np.ndarray,
    faith_raw: Optional[np.ndarray],
    alpha: float,
    beta: float,
    tau_f: float = 0.50,
    lam: float = 0.25,
    gate: str = "soft",  # 'soft' (penalty), 'hard' (drop), or 'off'
) -> Tuple[np.ndarray, np.ndarray]:
    """Implements S(c) = alpha*Rel + beta*Faith with the optional gate.

    Returns (fused_scores, keep_mask).
      - S2 (relevance-only) is exactly beta=0 with faith_raw=None.
      - S3 is beta>0 with a faith_raw column.
    'hard' gate drops candidates with normalised faith < tau_f (keep_mask False);
    'soft' gate subtracts lam from their score; 'off' applies neither.
    """
    assert abs((alpha + beta) - 1.0) < 1e-6, "alpha + beta must equal 1"
    rel_norm = minmax(rel_raw)
    n = len(rel_norm)
    keep = np.ones(n, dtype=bool)

    if beta == 0.0 or faith_raw is None:
        return rel_norm.copy(), keep  # reduces to S2

    faith_norm = minmax(faith_raw)
    fused = alpha * rel_norm + beta * faith_norm
    if gate == "soft":
        fused = np.where(faith_norm < tau_f, fused - lam, fused)
    elif gate == "hard":
        keep = faith_norm >= tau_f
    return fused, keep


# ---------------------------------------------------------------------------
# 4. Context selection
# ---------------------------------------------------------------------------
def mmr(
    doc_vecs: np.ndarray,
    relevance: np.ndarray,
    k: int,
    lambda_mmr: float = 0.5,
) -> List[int]:
    """Maximal Marginal Relevance selection over L2-normalised vectors.

    doc_vecs: (n, d) unit vectors; relevance: (n,) fused scores in [0,1].
    Returns indices of the selected k items (in selection order).
    """
    n = len(relevance)
    k = min(k, n)
    if n == 0:
        return []
    sim = doc_vecs @ doc_vecs.T  # cosine (vectors assumed unit-norm)
    selected: List[int] = []
    candidates = list(range(n))
    while candidates and len(selected) < k:
        if not selected:
            nxt = int(max(candidates, key=lambda i: relevance[i]))
        else:
            def score(i: int) -> float:
                max_sim = max(sim[i, j] for j in selected)
                return lambda_mmr * relevance[i] - (1 - lambda_mmr) * max_sim
            nxt = int(max(candidates, key=score))
        selected.append(nxt)
        candidates.remove(nxt)
    return selected


def order_lost_in_the_middle(items: List) -> List:
    """Reorder a score-descending list so the strongest items sit at the two
    ends and the weakest in the middle (Liu et al., 2023)."""
    left, right, toggle = [], [], True
    for it in items:  # items already sorted best-first
        (left if toggle else right).append(it)
        toggle = not toggle
    return left + right[::-1]


def enforce_token_budget(texts: List[str], budget_tokens: int) -> List[int]:
    """Greedy prefix that fits a whitespace-token budget. Returns kept indices."""
    kept, used = [], 0
    for i, t in enumerate(texts):
        cost = len(t.split())
        if used + cost > budget_tokens and kept:
            break
        kept.append(i)
        used += cost
    return kept


# ---------------------------------------------------------------------------
# 5. Self-containment sub-signal (deterministic proxy)
# ---------------------------------------------------------------------------
_DANGLING = re.compile(
    r"\b(as (shown|described|mentioned|discussed) (above|below|earlier|previously)"
    r"|the (aforementioned|former|latter|above|preceding)"
    r"|see (section|table|figure|fig\.?|appendix)"
    r"|as (previously|noted) (mentioned|stated|noted)"
    r"|respectively)\b",
    re.IGNORECASE,
)
_LEADING_ANAPHOR = re.compile(
    r"^\s*(it|they|this|that|these|those|he|she|such|its|their)\b", re.IGNORECASE
)
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_NUMBER = re.compile(r"\b\d+(\.\d+)?\b")
_UNIT = re.compile(
    r"\b(\d+(\.\d+)?)\s?(%|mg|kg|ml|mmol|mol|mm|cm|km|g|µg|ug|n=|p=|ci)\b",
    re.IGNORECASE,
)
_CAP_ENTITY = re.compile(r"\b([A-Z][a-z]{2,}|[A-Z]{2,})\b")


def self_containment_score(text: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Deterministic proxy for 'can this chunk be understood alone?'.

    Rewards local specificity (numbers, years, units, named entities) and
    penalises unresolved anaphora / dangling cross-references. This is a
    transparent stand-in for a model-based scorer; see RESEARCH_DESIGN.md for
    the validation protocol against human labels. Returns components + score.
    """
    w = weights or {"specificity": 0.5, "anaphora": 0.5}
    sents = re.split(r"(?<=[.!?])\s+", text.strip()) or [text]

    dangling_hits = len(_DANGLING.findall(text))
    leading_anaphora = sum(1 for s in sents if _LEADING_ANAPHOR.match(s))
    anaphora_raw = dangling_hits + leading_anaphora
    # squashing: 0 markers -> 0 penalty, grows toward 1
    anaphora_penalty = 1.0 - math.exp(-anaphora_raw)

    specificity = float(np.mean([
        bool(_NUMBER.search(text)),
        bool(_YEAR.search(text)),
        bool(_UNIT.search(text)),
        bool(_CAP_ENTITY.search(text)),
    ]))

    score = w["specificity"] * specificity + w["anaphora"] * (1.0 - anaphora_penalty)
    score = float(min(1.0, max(0.0, score)))
    return {
        "self_containment": score,
        "specificity": specificity,
        "anaphora_penalty": anaphora_penalty,
        "dangling_hits": float(dangling_hits),
        "leading_anaphora": float(leading_anaphora),
    }


# ---------------------------------------------------------------------------
# 6. Statistics
# ---------------------------------------------------------------------------
def mcnemar_test(b: int, c: int, exact: bool = True) -> Tuple[float, float]:
    """Paired binary test for two systems on the same queries.

    b = #queries where system A is correct/OK and B is not;
    c = #queries where B is correct/OK and A is not.
    Returns (statistic, p_value). Exact binomial for small discordant counts,
    else the chi-square approximation with continuity correction.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    if exact and n <= 25:
        p = 2.0 * _st.binom.cdf(min(b, c), n, 0.5)
        return float(min(b, c)), float(min(1.0, p))
    stat = (abs(b - c) - 1) ** 2 / n
    p = float(_st.chi2.sf(stat, df=1))
    return float(stat), p


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank on per-query continuous metrics."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    if np.allclose(diff, 0):
        return 0.0, 1.0
    try:
        stat, p = _st.wilcoxon(a, b, zero_method="wilcox")
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap CI for the mean paired difference (a - b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta effect size in [-1, 1] (non-parametric, paired-agnostic)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, dict]:
    """Holm-Bonferroni step-down correction across a family of tests."""
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    out, prev = {}, 0.0
    for rank, (name, p) in enumerate(items):
        thresh = alpha / (m - rank)
        adj = min(1.0, max(prev, p * (m - rank)))
        prev = adj
        out[name] = {"p_raw": p, "p_adj": adj, "threshold": thresh, "reject": p <= thresh}
    return out
