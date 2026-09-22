"""
faithfulness.py — the core contribution: Faith(c) from four sub-signals.

    Faith(c) = w1*Entailment + w2*SelfContainment + w3*CorpusAgreement + w4*SourceReliability

Per the architecture, if a sub-signal cannot be computed it is *excluded and the
remaining weights renormalised*, with the omission logged. On SciFact this
applies to Source Reliability: the BEIR/SciFact corpus ships title + abstract
but no publisher / effective-date / version metadata, so fabricating a
reliability prior would violate the integrity rule. We therefore disable w4 on
SciFact by default and renormalise (w1,w2,w3), and log it. See RESEARCH_DESIGN.md.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import core


class FaithfulnessScorer:
    def __init__(self, nli, weights: Optional[Dict[str, float]] = None,
                 use_source_reliability: bool = False, agreement_cap: int = 10):
        self.nli = nli
        self.w = weights or {"w1": 0.45, "w2": 0.20, "w3": 0.20, "w4": 0.15}
        self.use_source = use_source_reliability
        self.agreement_cap = agreement_cap          # bound the O(K^2) pairwise cost
        self._logged_omission = False

    # -- sub-signals --------------------------------------------------------
    def _entailment(self, claim: str, texts: List[str]) -> np.ndarray:
        return self.nli.entailment(texts, [claim] * len(texts))

    def _self_containment(self, texts: List[str]) -> np.ndarray:
        return np.array([core.self_containment_score(t)["self_containment"] for t in texts])

    def _corpus_agreement(self, texts: List[str]) -> np.ndarray:
        """1 - max contradiction of each candidate against the other top
        candidates. Capped to the top `agreement_cap` to bound cost."""
        n = len(texts)
        ref = list(range(min(n, self.agreement_cap)))
        agree = np.ones(n)
        for i in range(n):
            others = [j for j in ref if j != i]
            if not others:
                continue
            con = self.nli.contradiction([texts[j] for j in others], [texts[i]] * len(others))
            agree[i] = 1.0 - float(np.max(con))
        return agree

    def _source_reliability(self, meta: Optional[List[dict]], n: int) -> Optional[np.ndarray]:
        if not self.use_source or not meta:
            if not self._logged_omission:
                print("[faithfulness] source-reliability disabled (no corpus metadata); "
                      "renormalising w1,w2,w3.")
                self._logged_omission = True
            return None
        # If real metadata is supplied (e.g. via a Semantic Scholar join), compute
        # a deterministic prior here. Not available for vanilla SciFact.
        raise NotImplementedError("supply a metadata-based reliability prior")

    # -- combine ------------------------------------------------------------
    def score(self, claim: str, texts: List[str],
              meta: Optional[List[dict]] = None) -> Dict[str, np.ndarray]:
        n = len(texts)
        signals = {
            "entailment": (self._entailment(claim, texts), self.w["w1"]),
            "self_containment": (self._self_containment(texts), self.w["w2"]),
            "corpus_agreement": (self._corpus_agreement(texts), self.w["w3"]),
        }
        src = self._source_reliability(meta, n)
        if src is not None:
            signals["source_reliability"] = (src, self.w["w4"])

        total_w = sum(w for _, w in signals.values())
        faith = np.zeros(n)
        out: Dict[str, np.ndarray] = {}
        for name, (vals, w) in signals.items():
            out[name] = vals
            faith += (w / total_w) * vals          # renormalised weights
        out["faith"] = faith
        return out
