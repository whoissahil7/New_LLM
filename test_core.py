"""
Sanity tests for src/core.py on tiny, hand-constructed inputs.

IMPORTANT: the numbers here are unit-test fixtures, NOT experimental results.
They verify the logic is correct; they say nothing about SciFact performance.
Run: python -m pytest tests/test_core.py  (or: python tests/test_core.py)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import core  # noqa: E402


def test_rrf_orders_consensus_first():
    dense = ["a", "b", "c"]
    bm25 = ["b", "a", "d"]
    fused = core.reciprocal_rank_fusion([dense, bm25], k=60)
    order = core.top_k_from_scores(fused, 4)
    # 'a' and 'b' appear high in both lists -> ranked above 'c'/'d'
    assert set(order[:2]) == {"a", "b"}, order
    assert order[-2:] == ["c", "d"] or order[-2:] == ["d", "c"], order


def test_ir_metrics():
    ranked = ["d1", "d2", "d3", "d4"]
    rel = {"d2", "d4"}
    assert abs(core.recall_at_k(ranked, rel, 4) - 1.0) < 1e-9
    assert abs(core.recall_at_k(ranked, rel, 2) - 0.5) < 1e-9
    assert abs(core.precision_at_k(ranked, rel, 4) - 0.5) < 1e-9
    assert abs(core.reciprocal_rank(ranked, rel) - 0.5) < 1e-9  # first hit at rank 2
    ndcg = core.ndcg_at_k(ranked, {"d2": 1.0, "d4": 1.0}, 4)
    assert 0.0 < ndcg < 1.0, ndcg  # relevant items not at the very top


def test_fusion_reduces_to_s2_when_beta_zero():
    rel = np.array([2.0, 0.0, 1.0])
    s2, keep = core.fuse_and_gate(rel, None, alpha=1.0, beta=0.0)
    assert np.allclose(s2, core.minmax(rel))
    assert keep.all()


def test_soft_gate_penalises_low_faith():
    rel = np.array([1.0, 1.0])
    faith = np.array([1.0, 0.0])  # second candidate is unfaithful
    fused, keep = core.fuse_and_gate(rel, faith, 0.7, 0.3, tau_f=0.5, lam=0.25, gate="soft")
    assert fused[0] > fused[1], fused          # faithful candidate wins
    assert keep.all()                          # soft gate keeps both
    fused_h, keep_h = core.fuse_and_gate(rel, faith, 0.7, 0.3, tau_f=0.5, gate="hard")
    assert keep_h.tolist() == [True, False]    # hard gate drops the unfaithful one


def test_mmr_diversifies():
    # three unit vectors: 0 and 1 nearly identical, 2 orthogonal
    v = np.array([[1, 0.0], [0.98, 0.199], [0.0, 1.0]])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    rel = np.array([0.9, 0.85, 0.4])
    picks = core.mmr(v, rel, k=2, lambda_mmr=0.5)
    assert picks[0] == 0                        # most relevant first
    assert picks[1] == 2, picks                 # diversity beats the near-duplicate


def test_lost_in_middle_ordering():
    ordered = core.order_lost_in_the_middle([5, 4, 3, 2, 1])  # best-first
    assert ordered[0] == 5 and ordered[-1] == 4  # strongest at the ends
    assert 1 in ordered[1:-1]                     # weakest pushed inward


def test_self_containment_prefers_specific_text():
    vague = "It showed that they improved, as mentioned above."
    concrete = "Metformin reduced HbA1c by 1.2% in 312 patients (p=0.01) in 2019."
    sv = core.self_containment_score(vague)["self_containment"]
    sc = core.self_containment_score(concrete)["self_containment"]
    assert sc > sv, (sc, sv)


def test_mcnemar_and_holm():
    # A better than B: many queries where A ok / B not (b large), few the other way
    stat, p = core.mcnemar_test(b=18, c=3)
    assert p < 0.05, p
    corrected = core.holm_bonferroni({"halluc": 0.001, "faith": 0.02, "ndcg": 0.30})
    assert corrected["halluc"]["reject"] is True
    assert corrected["ndcg"]["reject"] is False


def test_bootstrap_and_cliffs():
    rng = np.random.default_rng(0)
    a = rng.normal(0.7, 0.1, 200)
    b = rng.normal(0.6, 0.1, 200)
    res = core.paired_bootstrap_diff(a, b, n_boot=2000, seed=1)
    assert res["ci_low"] > 0, res            # CI excludes 0 -> A > B
    assert core.cliffs_delta(a, b) > 0


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
